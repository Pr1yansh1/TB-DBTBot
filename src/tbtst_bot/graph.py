# src/tbtst_bot/graph.py
from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, TypedDict, TypeVar, cast

from pydantic import BaseModel, Field, ValidationError

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

# RemoveMessage import location varies by langchain version
try:
    from langchain_core.messages import RemoveMessage  # type: ignore
except Exception:  # pragma: no cover
    from langchain.messages import RemoveMessage  # type: ignore

from .config import get_llm
from .prompts import load_prompt
from .rag_utils import retrieve_tb_docs

# =========================================================
# Logging / tracing knobs (env)
# =========================================================
LOG_LEVEL = os.getenv("TBTST_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("tbtst.graph")

DEBUG_METRICS = os.getenv("TBTST_DEBUG_METRICS", "0") == "1"
LOG_PROMPT_SIZES = os.getenv("TBTST_LOG_PROMPT_SIZES", "0") == "1"

# Truncation recovery (finish cut-off replies automatically)
CONTINUE_ON_TRUNCATION = os.getenv("TBTST_CONTINUE_ON_TRUNCATION", "1") == "1"
MAX_CONTINUATION_CALLS = int(os.getenv("TBTST_MAX_CONTINUATION_CALLS", "1"))

# =========================================================
# Prompts (loaded from files)
# =========================================================
CLASSIFY_SYSTEM = load_prompt("classify_route.txt")
CLASSIFY_USER_TMPL = load_prompt("classify_route_user.txt")
CRISIS_TEXT = load_prompt("crisis.txt")

# Global response rules applied to all user-facing responses (should enforce Spanish)
GLOBAL_RULES = load_prompt("global_response_rules.txt")

# DBT mini
DBT_SYSTEM = load_prompt("dbt_system.txt")

# DBT full
DBT_BRAIN_ROUTER_SYSTEM = load_prompt("dbt_brain.txt")
DBT_DT_SYSTEM = load_prompt("DBT/dt_prompt.txt")
DBT_MIND_SYSTEM = load_prompt("DBT/mind_prompt.txt")
DBT_ER_SYSTEM = load_prompt("DBT/er_prompt.txt")
DBT_IE_SYSTEM = load_prompt("DBT/ie_prompt.txt")

# TB RAG
TB_RAG_ANSWER_SYSTEM = load_prompt("tb_rag_answer_system.txt")
TB_RAG_ANSWER_USER_TMPL = load_prompt("tb_rag_answer_user.txt")

# Misc (user-facing content is Spanish)
MISC_SYSTEM = """
Eres un asistente de apoyo para una app de TB y bienestar.
Tu tarea aquí es responder mensajes "misc": saludos, dudas sobre el sistema, o cosas fuera de TB/DBT.
Responde SIEMPRE en español, con calidez y brevedad.
Si el usuario pide algo concreto pero es ambiguo, haz EXACTAMENTE 1 pregunta breve para orientar.
No uses jerga. No inventes datos médicos.
""".strip()

# =========================================================
# State types
# =========================================================
RiskLevel = Literal["none", "passive", "active_no_plan", "active_with_plan", "uncertain"]
Route = Literal["faq", "dbt", "misc"]

# For TB FAQ routing: whether latent TB is explicitly in-scope this turn.
TBTopic = Literal["general", "latent"]

DBTModule = Literal["DT", "MIND", "ER", "IE"]
DBTMode = Literal["connect", "offer", "coach"]
DBTContinuity = Literal["same", "new", "unclear"]
DBTSkillStatus = Literal["none", "offered", "accepted", "declined", "paused", "completed"]

RECENT_SKILLS_MAX = 4


class RecentSkillEvent(TypedDict, total=False):
    skill: str
    status: DBTSkillStatus
    trace_id: str
    note: str  # optional internal note


class State(MessagesState, total=False):
    # main routing
    safety_risk_level: RiskLevel
    safety_triggers: List[str]
    has_protective: bool
    safety_route: Literal["ok", "crisis"]
    route: Route
    route_source: str
    classifier_raw: str
    classifier_parse_ok: bool

    # NEW: For TB FAQ route only
    tb_topic: TBTopic

    # dbt router outputs
    dbt_module: DBTModule
    dbt_confidence: float
    dbt_rationale_brief: str
    dbt_signals: Dict[str, Any]
    dbt_router_raw: str

    # DBT conversational control (router-controlled)
    dbt_mode: DBTMode
    dbt_continuity: DBTContinuity

    # DBT skill thread memory (agent-updated)
    dbt_active_skill: str
    dbt_skill_status: DBTSkillStatus
    dbt_recent_skills: List[RecentSkillEvent]

    # short-term memory management (rolling summary of conversation only)
    summary: str

    # debugging / tracing
    last_trace_id: str


# =========================================================
# Memory / context management settings
# =========================================================
MAX_TOKENS_BEFORE_SUMMARY = 4500
KEEP_LAST_MESSAGES = 8
LLM_RECENT_MESSAGES_MAX_TOKENS = 1800
MIN_RECENT_MESSAGES = 10

SUMMARY_SYSTEM = """You maintain a compact, factual rolling summary of a multi-turn conversation.
Goal: preserve details that prevent repeated questions and keep continuity.

Include:
- Stable user facts/preferences (if any)
- Key situation/context the user shared
- What has been tried/decided
- Constraints or safety-relevant details (only what the user said)

IMPORTANT:
- Do NOT invent or infer internal DBT control state (mode/skill/status). Only summarize what is explicitly in user-visible messages.
Output ONLY the updated summary text."""


# =========================================================
# Trace + metrics helpers
# =========================================================
def _trace_id_for_state(state: State) -> str:
    msgs: List[BaseMessage] = state.get("messages", []) or []
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            mid = getattr(m, "id", None)
            if mid:
                return str(mid)
            break
    return uuid.uuid4().hex[:10]


def _tokens(messages: List[BaseMessage]) -> int:
    return int(count_tokens_approximately(messages))


@dataclass
class CallMetrics:
    name: str
    trace_id: str
    latency_ms: float
    in_tokens: int
    out_chars: int
    stop_reason: str = ""
    usage: Dict[str, Any] = None


def _is_throttle_error(e: Exception) -> bool:
    msg = str(e)
    return ("ThrottlingException" in msg) or ("Too many requests" in msg) or ("Rate exceeded" in msg)


def _extract_llm_meta(resp: Any) -> Tuple[str, Dict[str, Any]]:
    meta = getattr(resp, "response_metadata", None) or {}
    usage = meta.get("usage") or meta.get("token_usage") or meta.get("usage_metadata") or {}
    stop_reason = (
        meta.get("stop_reason")
        or meta.get("stopReason")
        or meta.get("finish_reason")
        or meta.get("completion_reason")
        or meta.get("completionReason")
        or ""
    )
    if not isinstance(usage, dict):
        usage = {"usage": usage}
    return str(stop_reason), usage


def _timed_invoke(name: str, llm: Any, messages: List[BaseMessage], *, trace_id: str) -> Tuple[Any, CallMetrics]:
    t0 = time.perf_counter()
    last_exc: Exception | None = None

    for attempt in range(6):
        try:
            resp = llm.invoke(messages)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            stop_reason, usage = _extract_llm_meta(resp)

            m = CallMetrics(
                name=name,
                trace_id=trace_id,
                latency_ms=dt_ms,
                in_tokens=_tokens(messages),
                out_chars=len((getattr(resp, "content", "") or "")),
                stop_reason=stop_reason,
                usage=usage or {},
            )

            if DEBUG_METRICS:
                logger.info(
                    "[TRACE %s] [LLM] %s latency=%.1fms in_tokens~%d out_chars=%d stop_reason=%s usage=%s",
                    m.trace_id,
                    m.name,
                    m.latency_ms,
                    m.in_tokens,
                    m.out_chars,
                    m.stop_reason,
                    m.usage,
                )
            return resp, m

        except Exception as e:
            last_exc = e
            if _is_throttle_error(e) and attempt < 5:
                sleep_s = min(10.0, (2**attempt) * 0.6) + random.uniform(0.0, 0.35)
                logger.warning(
                    "[TRACE %s] [LLM] %s throttled; retrying in %.2fs (attempt %d/6)",
                    trace_id,
                    name,
                    sleep_s,
                    attempt + 1,
                )
                time.sleep(sleep_s)
                continue
            raise

    raise last_exc or RuntimeError("LLM invoke failed")


def _timed_tool(name: str, fn: Any, args: Dict[str, Any], *, trace_id: str) -> Any:
    t0 = time.perf_counter()
    out = fn.invoke(args)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if DEBUG_METRICS:
        n_sources = 0
        if isinstance(out, dict) and isinstance(out.get("sources"), list):
            n_sources = len(out["sources"])
        logger.info("[TRACE %s] [TOOL] %s latency=%.1fms sources=%d", trace_id, name, dt_ms, n_sources)
    return out


def _log_prompt_sizes_once() -> None:
    if not LOG_PROMPT_SIZES:
        return
    prompts: Dict[str, str] = {
        "GLOBAL_RULES": GLOBAL_RULES,
        "CLASSIFY_SYSTEM": CLASSIFY_SYSTEM,
        "DBT_SYSTEM": DBT_SYSTEM,
        "DBT_BRAIN_ROUTER_SYSTEM": DBT_BRAIN_ROUTER_SYSTEM,
        "DBT_DT_SYSTEM": DBT_DT_SYSTEM,
        "DBT_MIND_SYSTEM": DBT_MIND_SYSTEM,
        "DBT_ER_SYSTEM": DBT_ER_SYSTEM,
        "DBT_IE_SYSTEM": DBT_IE_SYSTEM,
        "TB_RAG_ANSWER_SYSTEM": TB_RAG_ANSWER_SYSTEM,
    }
    for name, text in prompts.items():
        lines = len((text or "").splitlines())
        chars = len(text or "")
        toks = _tokens([SystemMessage(content=text or "")])
        logger.info("[PROMPT_FILE] %s lines=%d chars=%d tokens~%d", name, lines, chars, toks)


# =========================================================
# Continuation helpers
# =========================================================
def _looks_truncated(text: str) -> bool:
    t = (text or "").rstrip()
    if not t:
        return False
    if t[-1].isalnum():
        return True
    if t.endswith(("¿", "¡", "(", "[", "{", "“", '"', "'", "…")):
        return True
    return False


def _should_continue(stop_reason: str, text: str) -> bool:
    sr = (stop_reason or "").lower().strip()
    if sr in {"length", "max_tokens", "token_limit"}:
        return True
    if ("max" in sr and "token" in sr) or ("length" in sr):
        return True
    return _looks_truncated(text)


def _invoke_with_auto_continue(
    name: str,
    llm: Any,
    messages: List[BaseMessage],
    *,
    trace_id: str,
) -> AIMessage:
    resp, m = _timed_invoke(name, llm, messages, trace_id=trace_id)
    cur_text = (getattr(resp, "content", "") or "")

    if not CONTINUE_ON_TRUNCATION:
        return AIMessage(content=cur_text)

    if not _should_continue(m.stop_reason, cur_text):
        return AIMessage(content=cur_text)

    for i in range(MAX_CONTINUATION_CALLS):
        cont_prompt = (
            "Continúa EXACTAMENTE donde te cortaste.\n"
            "- Empieza por terminar la última frase incompleta.\n"
            "- NO repitas lo que ya dijiste.\n"
            "- Mantén el mismo tono y formato.\n"
        )
        cont_msgs: List[BaseMessage] = [*messages, AIMessage(content=cur_text), HumanMessage(content=cont_prompt)]
        resp2, m2 = _timed_invoke(f"{name}:cont{i+1}", llm, cont_msgs, trace_id=trace_id)
        add = (getattr(resp2, "content", "") or "").lstrip()

        if not add:
            break

        joiner = "" if cur_text.endswith(("\n", " ")) else " "
        cur_text = f"{cur_text}{joiner}{add}"

        if not _should_continue(m2.stop_reason, cur_text):
            break

    return AIMessage(content=cur_text)


# =========================================================
# Core helpers
# =========================================================
def _latest_user_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _safe_fill_user_text(template: str, user_text: str) -> str:
    return template.replace("{user_text}", user_text)


_msg_token_cache: Dict[str, int] = {}


def _message_cache_key(m: BaseMessage) -> str:
    mid = getattr(m, "id", None)
    if mid:
        return f"id:{mid}"
    return f"obj:{id(m)}"


def _message_tokens(m: BaseMessage) -> int:
    k = _message_cache_key(m)
    t = _msg_token_cache.get(k)
    if t is not None:
        return t
    t = int(count_tokens_approximately([m]))
    _msg_token_cache[k] = t
    return t


def _select_recent_messages(
    messages: List[BaseMessage],
    *,
    max_tokens: int,
    min_messages: int = MIN_RECENT_MESSAGES,
) -> List[BaseMessage]:
    if not messages:
        return []

    hard_min = max(2, min_messages)
    selected_rev: List[BaseMessage] = []
    total = 0

    for m in reversed(messages):
        mt = _message_tokens(m)

        if len(selected_rev) >= hard_min and (total + mt) > max_tokens:
            break

        selected_rev.append(m)
        total += mt

        if total >= max_tokens and len(selected_rev) >= hard_min:
            break

    return list(reversed(selected_rev))


def _system_and_summary_messages(system_prompt: str, summary: str) -> List[BaseMessage]:
    combined_system = f"{GLOBAL_RULES.strip()}\n\n---\n\n{system_prompt.strip()}".strip()
    msgs: List[BaseMessage] = [SystemMessage(content=combined_system)]
    if summary.strip():
        msgs.append(
            SystemMessage(
                content=(
                    "Conversation memory (compact & factual). Use this to maintain continuity and avoid repeating questions:\n"
                    f"{summary.strip()}"
                )
            )
        )
    return msgs


def _llm_messages_with_summary(system_prompt: str, state: State, *, trace_id: str, label: str) -> List[BaseMessage]:
    summary = (state.get("summary") or "").strip()
    prefix = _system_and_summary_messages(system_prompt, summary)

    base_tokens = _tokens(prefix)
    recent_budget = max(256, LLM_RECENT_MESSAGES_MAX_TOKENS - base_tokens)

    recent = _select_recent_messages(state.get("messages", []) or [], max_tokens=recent_budget)

    if DEBUG_METRICS:
        sys_tokens = _tokens([prefix[0]])
        summ_tokens = _tokens(prefix[1:]) if len(prefix) > 1 else 0
        recent_tokens = _tokens(recent)
        total = sys_tokens + summ_tokens + recent_tokens
        logger.info(
            "[TRACE %s] [PROMPT:%s] sys~%d summary~%d recent_msgs=%d recent~%d recent_budget~%d total~%d",
            trace_id,
            label,
            sys_tokens,
            summ_tokens,
            len(recent),
            recent_tokens,
            int(recent_budget),
            total,
        )

    return [*prefix, *recent]


# =========================================================
# Nodes: classify + safety
# =========================================================
class ClassifyOut(BaseModel):
    safety_risk_level: RiskLevel = Field(...)
    safety_triggers: List[str] = Field(default_factory=list)
    has_protective: bool = Field(False)
    route: Route = Field(...)

    # NEW: only meaningful when route == "faq"
    tb_topic: TBTopic = Field(default="general")


def _coerce_and_validate_classifier(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    try:
        obj = ClassifyOut.model_validate(data)
        triggers = [t for t in (obj.safety_triggers or []) if isinstance(t, str)]
        out = {
            "safety_risk_level": obj.safety_risk_level,
            "safety_triggers": triggers[:5],
            "has_protective": bool(obj.has_protective),
            "route": obj.route,
            "tb_topic": obj.tb_topic,
        }
        return out, True
    except ValidationError:
        return {
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "route": "misc",
            "tb_topic": "general",
        }, False


def _parse_classifier_json(raw: str) -> Tuple[Dict[str, Any], bool]:
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return _coerce_and_validate_classifier({})
        return _coerce_and_validate_classifier(data)
    except Exception:
        return _coerce_and_validate_classifier({})


def classify_node(state: State) -> Dict[str, Any]:
    trace_id = _trace_id_for_state(state)
    state["last_trace_id"] = trace_id

    user_text = _latest_user_text(state.get("messages", []))
    if not user_text.strip():
        return {
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "safety_route": "ok",
            "route": "misc",
            "tb_topic": "general",
            "route_source": "empty_input_default",
            "classifier_raw": "",
            "classifier_parse_ok": True,
        }

    user_prompt = _safe_fill_user_text(CLASSIFY_USER_TMPL, user_text)
    llm = get_llm(temperature=0.0, max_tokens=240)

    raw_msg, _ = _timed_invoke(
        "classify:first",
        llm,
        [SystemMessage(content=CLASSIFY_SYSTEM), HumanMessage(content=user_prompt)],
        trace_id=trace_id,
    )
    raw1 = (raw_msg.content or "").strip()
    parsed1, ok1 = _parse_classifier_json(raw1)

    need_retry = (not ok1) or (
        parsed1.get("safety_risk_level") == "uncertain" and not (parsed1.get("safety_triggers") or [])
    )

    raw = raw1
    parsed = parsed1
    ok = ok1

    if need_retry:
        retry_system = (
            CLASSIFY_SYSTEM.strip()
            + "\n\nREMINDER: Output EXACTLY one line of STRICT JSON with only the required keys. No extra text."
        )
        raw_msg2, _ = _timed_invoke(
            "classify:retry",
            llm,
            [SystemMessage(content=retry_system), HumanMessage(content=user_prompt)],
            trace_id=trace_id,
        )
        raw2 = (raw_msg2.content or "").strip()
        parsed2, ok2 = _parse_classifier_json(raw2)

        if ok2:
            raw, parsed, ok = raw2, parsed2, ok2
        else:
            raw, parsed, ok = (
                raw1,
                {
                    "safety_risk_level": "none",
                    "safety_triggers": [],
                    "has_protective": False,
                    "route": "misc",
                    "tb_topic": "general",
                },
                False,
            )

    risk: RiskLevel = parsed["safety_risk_level"]
    route: Route = parsed["route"]
    tb_topic: TBTopic = parsed.get("tb_topic", "general")

    safety_route: Literal["ok", "crisis"] = (
        "crisis" if risk in ("passive", "active_no_plan", "active_with_plan") else "ok"
    )

    if DEBUG_METRICS:
        logger.info(
            "[TRACE %s] classify result risk=%s safety_route=%s route=%s tb_topic=%s parse_ok=%s triggers=%s",
            trace_id,
            risk,
            safety_route,
            route,
            tb_topic,
            ok,
            parsed.get("safety_triggers", []),
        )

    return {
        "safety_risk_level": risk,
        "safety_triggers": parsed["safety_triggers"],
        "has_protective": parsed["has_protective"],
        "safety_route": safety_route,
        "route": route,
        "tb_topic": tb_topic,
        "route_source": "llm",
        "classifier_raw": raw,
        "classifier_parse_ok": ok,
    }


def crisis_node(state: State) -> Dict[str, Any]:
    return {"messages": [AIMessage(content=CRISIS_TEXT)]}


# =========================================================
# Memory manager node (summarize + delete)
# =========================================================
def memory_manager_node(state: State) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)

    messages: List[BaseMessage] = state.get("messages", [])
    if not messages:
        return {}

    approx_tokens = int(count_tokens_approximately(messages))
    if DEBUG_METRICS:
        logger.info("[TRACE %s] memory check messages=%d tokens~%d", trace_id, len(messages), approx_tokens)

    if approx_tokens < MAX_TOKENS_BEFORE_SUMMARY:
        return {}

    existing_summary = (state.get("summary") or "").strip()
    if existing_summary:
        summary_message = (
            f"Existing summary:\n{existing_summary}\n\n"
            "Update it by incorporating ONLY new information from the recent messages:"
        )
    else:
        summary_message = "Create a compact summary of the conversation so far:"

    summarizer = get_llm(temperature=0.0, max_tokens=320)
    summary_input = [SystemMessage(content=SUMMARY_SYSTEM), *messages, HumanMessage(content=summary_message)]

    resp, _ = _timed_invoke("summarize", summarizer, summary_input, trace_id=trace_id)
    new_summary = (resp.content or "").strip()

    to_remove: List[Any] = []
    if len(messages) > KEEP_LAST_MESSAGES:
        for msg in messages[:-KEEP_LAST_MESSAGES]:
            mid = getattr(msg, "id", None)
            if mid:
                to_remove.append(RemoveMessage(id=mid))

    out: Dict[str, Any] = {"summary": new_summary}
    if to_remove:
        out["messages"] = to_remove
    return out


# =========================================================
# DBT MINI node
# =========================================================
def dbt_mini_node(state: State) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)
    llm = get_llm(temperature=0.25, max_tokens=650)
    msgs = _llm_messages_with_summary(DBT_SYSTEM, state, trace_id=trace_id, label="dbt_mini")
    reply = _invoke_with_auto_continue("dbt:mini", llm, msgs, trace_id=trace_id)
    return {"messages": [AIMessage(content=reply.content or "")]}


# =========================================================
# TB-info specialist: citation-safe RAG + clarify gate
# =========================================================
class TBAnswerJSON(BaseModel):
    answer: str = Field(...)
    citations_used: List[int] = Field(default_factory=list)


class TBGateOut(BaseModel):
    action: Literal["clarify", "answer"] = Field(...)
    clarifying_question: str = Field(default="")


def _format_sources_block(sources: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for s in sources:
        sid = s.get("id")
        title = s.get("title", "Unknown")
        filename = s.get("filename", "unknown.txt")
        excerpt = (s.get("excerpt") or "").strip()
        parts.append(
            f"Fuente {sid}:\n"
            f"Título: {title}\n"
            f"Archivo: {filename}\n"
            f"Extracto:\n{excerpt}\n"
        )
    return "\n".join(parts).strip()


def _render_answer_with_references(answer_text: str, sources: List[Dict[str, Any]], citations_used: List[int]) -> str:
    src_by_id: Dict[int, Dict[str, Any]] = {}
    for s in sources:
        try:
            sid = int(s.get("id"))
            src_by_id[sid] = s
        except Exception:
            continue

    used: Set[int] = set()
    for x in citations_used or []:
        if isinstance(x, int) and x in src_by_id:
            used.add(x)

    if not used:
        return (answer_text or "").strip()

    lines = [(answer_text or "").strip(), "", "Referencias:"]
    for sid in sorted(used):
        s = src_by_id[sid]
        title = s.get("title", "Unknown")
        filename = s.get("filename", "unknown.txt")
        lines.append(f"[{sid}] {title} ({filename})")
    return "\n".join(lines).strip()


def _tb_should_clarify(user_text: str, summary: str) -> TBGateOut:
    """
    LLM-based clarify gate (no keyword heuristics):
    - If the user’s request is vague / not a real question → clarify (ONE question).
    - Otherwise → answer.
    """
    system = (
        "Eres un asistente de información sobre tuberculosis.\n"
        "Decide si el mensaje del usuario es lo suficientemente específico para responder ahora, "
        "o si es demasiado vago y necesitas UNA sola pregunta de aclaración.\n\n"
        "Devuelve SOLO JSON con:\n"
        '  action: "clarify" o "answer"\n'
        "  clarifying_question: string (vacío si action=answer)\n\n"
        "Reglas:\n"
        "- Responde SIEMPRE en español.\n"
        "- Si action=clarify: haz EXACTAMENTE 1 pregunta breve (sin listas, sin explicaciones médicas todavía).\n"
        "- Si el usuario pegó texto largo pero no preguntó nada concreto: action=clarify.\n"
    )

    human = (
        f"Resumen (si existe):\n{summary.strip()}\n\n"
        f"Mensaje del usuario:\n{user_text.strip()}\n"
    )

    llm = get_llm(temperature=0.0, max_tokens=100).with_structured_output(TBGateOut)
    out = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])

    q = (out.clarifying_question or "").strip()
    if out.action == "clarify" and not q:
        q = "¿Qué quieres saber específicamente sobre la tuberculosis (síntomas, pruebas, tratamiento, contagio u otra cosa)?"
    return TBGateOut(action=out.action, clarifying_question=q)


def tb_info_node(state: State) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)

    user_text = _latest_user_text(state.get("messages", []))
    if not user_text.strip():
        return {"messages": [AIMessage(content="")]}

    summary = (state.get("summary") or "").strip()

    # --- NEW: Clarify gate (pre-RAG) ---
    gate = _tb_should_clarify(user_text, summary)
    if gate.action == "clarify":
        return {"messages": [AIMessage(content=gate.clarifying_question)]}

    # --- Retrieval: allow latent ONLY if classifier/router said so ---
    tb_topic: TBTopic = state.get("tb_topic", "general")
    allow_latent = tb_topic == "latent"

    tool_out = _timed_tool(
        "retrieve_tb_docs",
        retrieve_tb_docs,
        {"query": user_text, "allow_latent": allow_latent},
        trace_id=trace_id,
    )
    sources = tool_out.get("sources", []) if isinstance(tool_out, dict) else []

    sources_block = _format_sources_block(sources)
    user_prompt = (
        TB_RAG_ANSWER_USER_TMPL.replace("{user_text}", user_text)
        .replace("{sources_block}", sources_block)
    )

    prefix = _system_and_summary_messages(TB_RAG_ANSWER_SYSTEM, summary)

    llm = get_llm(temperature=0.2, max_tokens=700).with_structured_output(TBAnswerJSON)

    t0 = time.perf_counter()
    result = llm.invoke([*prefix, HumanMessage(content=user_prompt)])
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if DEBUG_METRICS:
        stop_reason, usage = _extract_llm_meta(result)
        logger.info(
            "[TRACE %s] [LLM] tb_info latency=%.1fms in_tokens~%d out_answer_chars=%d cited=%s stop_reason=%s usage=%s allow_latent=%s",
            trace_id,
            dt_ms,
            _tokens([*prefix, HumanMessage(content=user_prompt)]),
            len((result.answer or "")),
            result.citations_used,
            stop_reason,
            usage,
            allow_latent,
        )

    final_text = _render_answer_with_references(
        answer_text=result.answer,
        sources=sources,
        citations_used=result.citations_used,
    )
    return {"messages": [AIMessage(content=final_text)]}


# =========================================================
# MISC node
# =========================================================
def misc_node(state: State) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)
    llm = get_llm(temperature=0.2, max_tokens=420)
    msgs = _llm_messages_with_summary(MISC_SYSTEM, state, trace_id=trace_id, label="misc")
    reply = _invoke_with_auto_continue("misc", llm, msgs, trace_id=trace_id)
    return {"messages": [AIMessage(content=reply.content or "")]}


# =========================================================
# DBT FULL subgraph (unchanged below, keep your existing code)
# =========================================================
# (… everything after this point stays exactly as in your file …)

# --- KEEP YOUR EXISTING DBT FULL SECTION HERE UNCHANGED ---
# I am not repeating it to avoid accidental diffs.
# In your repo, paste this patched file over your current graph.py,
# because this block includes only the changed sections above.
#
# If you want, I can re-paste the entire DBT section too, but it’s long
# and easy to introduce mistakes by hand-merging.
#
# =========================================================

# =========================================================
# Top-level routing
# =========================================================
def branch_after_classify(state: State) -> Literal["crisis", "tb_info", "dbt", "misc"]:
    if state.get("safety_route") == "crisis":
        return "crisis"

    route = state.get("route", "misc")
    if route == "faq":
        return "tb_info"
    if route == "dbt":
        return "dbt"
    return "misc"


def build_graph(*, dbt_node: Any):
    g = StateGraph(State)
    g.add_node("classify", classify_node)
    g.add_node("memory", memory_manager_node)

    g.add_node("crisis", crisis_node)
    g.add_node("tb_info", tb_info_node)
    g.add_node("dbt", dbt_node)
    g.add_node("misc", misc_node)

    g.add_edge(START, "classify")
    g.add_edge("classify", "memory")

    g.add_conditional_edges(
        "memory",
        branch_after_classify,
        {"crisis": "crisis", "tb_info": "tb_info", "dbt": "dbt", "misc": "misc"},
    )

    g.add_edge("crisis", END)
    g.add_edge("tb_info", END)
    g.add_edge("dbt", END)
    g.add_edge("misc", END)

    return g.compile(checkpointer=MemorySaver())


# Public compiled graphs
GRAPH_DBT_MINI = build_graph(dbt_node=dbt_mini_node)

# If you use DBT_FULL_SUBGRAPH in this file, keep your existing definition and compile:
# GRAPH_DBT_FULL = build_graph(dbt_node=DBT_FULL_SUBGRAPH)

_log_prompt_sizes_once()
