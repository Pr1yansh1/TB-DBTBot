# src/tbtst_bot/graph.py
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Set, Tuple

from pydantic import BaseModel, Field, ValidationError

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

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
# Turn on per-call trace logs:
#   TBTST_DEBUG_METRICS=1
# Log prompt file token/line sizes once at startup:
#   TBTST_LOG_PROMPT_SIZES=1
# Choose log level:
#   TBTST_LOG_LEVEL=INFO|DEBUG
LOG_LEVEL = os.getenv("TBTST_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("tbtst.graph")

DEBUG_METRICS = os.getenv("TBTST_DEBUG_METRICS", "0") == "1"
LOG_PROMPT_SIZES = os.getenv("TBTST_LOG_PROMPT_SIZES", "0") == "1"


# =========================================================
# Prompts (loaded from files)
# =========================================================
CLASSIFY_SYSTEM = load_prompt("classify_route.txt")
CLASSIFY_USER_TMPL = load_prompt("classify_route_user.txt")
CRISIS_TEXT = load_prompt("crisis.txt")

# Global response rules applied to all user-facing responses
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

# Misc (meta / greetings / off-topic)
MISC_SYSTEM = """
Sos un asistente de apoyo para una app de TB y bienestar.
Tu tarea acá es responder mensajes "misc": saludos, dudas sobre el sistema, o cosas fuera de TB/DBT.
Respondé en español (vos), con calidez y brevedad.
Si el usuario pide algo concreto, pedí 1 pregunta breve para orientar. No uses jerga. No inventes datos médicos.
""".strip()


# =========================================================
# State
# =========================================================
RiskLevel = Literal["none", "passive", "active_no_plan", "active_with_plan", "uncertain"]
Route = Literal["faq", "dbt", "misc"]

# For DBT full router
DBTModule = Literal["DT", "MIND", "ER", "IE"]


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

    # dbt full router outputs (only set when route == "dbt")
    dbt_module: DBTModule
    dbt_confidence: float
    dbt_rationale_brief: str
    dbt_signals: Dict[str, Any]
    dbt_router_raw: str

    # short-term memory management (rolling summary)
    summary: str

    # optional: stored for debugging if you want it
    last_trace_id: str


# =========================================================
# Memory / context management settings
# =========================================================
MAX_TOKENS_BEFORE_SUMMARY = 2200
KEEP_LAST_MESSAGES = 8

# Target max tokens for recent messages sent to LLM; we dynamically subtract system+summary.
LLM_RECENT_MESSAGES_MAX_TOKENS = 1400

SUMMARY_SYSTEM = """You maintain a compact, factual rolling summary of a multi-turn conversation for a coaching chatbot.
Goal: preserve details that prevent repeated questions and keep continuity.
Include:
- Stable user facts/preferences (if any)
- Key situation/context the user shared
- What has been tried/decided
- Constraints or safety-relevant details (only what the user said)
Be concise. No speculation. Output ONLY the updated summary text."""


# =========================================================
# Trace + metrics helpers
# =========================================================
def _trace_id_for_state(state: State) -> str:
    """
    Best-effort per-turn trace id to correlate logs across nodes.
    Uses the most recent human message id if available, otherwise a random uuid.
    """
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


def _timed_invoke(name: str, llm: Any, messages: List[BaseMessage], *, trace_id: str) -> Tuple[Any, CallMetrics]:
    t0 = time.perf_counter()
    resp = llm.invoke(messages)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    m = CallMetrics(
        name=name,
        trace_id=trace_id,
        latency_ms=dt_ms,
        in_tokens=_tokens(messages),
        out_chars=len((getattr(resp, "content", "") or "")),
    )

    if DEBUG_METRICS:
        logger.info(
            "[TRACE %s] [LLM] %s latency=%.1fms in_tokens~%d out_chars=%d",
            m.trace_id,
            m.name,
            m.latency_ms,
            m.in_tokens,
            m.out_chars,
        )
    return resp, m


def _timed_tool(name: str, fn: Any, args: Dict[str, Any], *, trace_id: str) -> Any:
    t0 = time.perf_counter()
    out = fn.invoke(args)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if DEBUG_METRICS:
        n_sources = 0
        if isinstance(out, dict) and isinstance(out.get("sources"), list):
            n_sources = len(out["sources"])
        logger.info(
            "[TRACE %s] [TOOL] %s latency=%.1fms sources=%d",
            trace_id,
            name,
            dt_ms,
            n_sources,
        )
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
# Core helpers
# =========================================================
def _latest_user_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _safe_fill_user_text(template: str, user_text: str) -> str:
    return template.replace("{user_text}", user_text)


class ClassifyOut(BaseModel):
    safety_risk_level: RiskLevel = Field(...)
    safety_triggers: List[str] = Field(default_factory=list)
    has_protective: bool = Field(False)
    route: Route = Field(...)


def _coerce_and_validate_classifier(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Returns (normalized_dict, parse_ok).
    Normalization is conservative: if required fields are missing/invalid, parse_ok=False.
    """
    try:
        obj = ClassifyOut.model_validate(data)
        triggers = [t for t in (obj.safety_triggers or []) if isinstance(t, str)]
        out = {
            "safety_risk_level": obj.safety_risk_level,
            "safety_triggers": triggers[:5],
            "has_protective": bool(obj.has_protective),
            "route": obj.route,
        }
        return out, True
    except ValidationError:
        return {
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "route": "misc",
        }, False


def _parse_classifier_json(raw: str) -> Tuple[Dict[str, Any], bool]:
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return _coerce_and_validate_classifier({})
        return _coerce_and_validate_classifier(data)
    except Exception:
        return _coerce_and_validate_classifier({})


def _format_sources_block(sources: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for s in sources:
        sid = s.get("id")
        title = s.get("title", "Unknown")
        filename = s.get("filename", "unknown.txt")
        excerpt = (s.get("excerpt") or "").strip()
        parts.append(
            f"Source {sid}:\n"
            f"Title: {title}\n"
            f"Filename: {filename}\n"
            f"Excerpt:\n{excerpt}\n"
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


def _trim_for_llm(messages: List[BaseMessage], *, max_tokens: int) -> List[BaseMessage]:
    if not messages:
        return []
    return trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        end_on=("human", "tool"),
    )


def _system_and_summary_messages(system_prompt: str, summary: str) -> List[BaseMessage]:
    combined_system = f"{GLOBAL_RULES.strip()}\n\n---\n\n{system_prompt.strip()}".strip()
    msgs: List[BaseMessage] = [SystemMessage(content=combined_system)]
    if summary.strip():
        msgs.append(
            SystemMessage(
                content=(
                    "Memoria de conversación (concisa y factual). "
                    "Usala para mantener continuidad y no repetir preguntas:\n"
                    f"{summary.strip()}"
                )
            )
        )
    return msgs


def _llm_messages_with_summary(system_prompt: str, state: State, *, trace_id: str, label: str) -> List[BaseMessage]:
    """
    Construct model input:
      - System (GLOBAL_RULES + module prompt)
      - Optional summary as SystemMessage
      - Trimmed recent messages, with headroom
    Also logs token breakdown when DEBUG_METRICS=1.
    """
    summary = (state.get("summary") or "").strip()
    prefix = _system_and_summary_messages(system_prompt, summary)

    base_tokens = _tokens(prefix)
    recent_budget = max(256, LLM_RECENT_MESSAGES_MAX_TOKENS - base_tokens)
    recent = _trim_for_llm(state.get("messages", []), max_tokens=recent_budget)

    if DEBUG_METRICS:
        sys_tokens = _tokens([prefix[0]])
        summ_tokens = _tokens(prefix[1:]) if len(prefix) > 1 else 0
        recent_tokens = _tokens(recent)
        total = sys_tokens + summ_tokens + recent_tokens
        logger.info(
            "[TRACE %s] [PROMPT:%s] sys~%d summary~%d recent~%d recent_budget~%d total~%d",
            trace_id,
            label,
            sys_tokens,
            summ_tokens,
            recent_tokens,
            int(recent_budget),
            total,
        )

    return [*prefix, *recent]


# =========================================================
# Nodes: classify + safety
# =========================================================
def classify_node(state: State) -> Dict[str, Any]:
    trace_id = _trace_id_for_state(state)
    state["last_trace_id"] = trace_id  # optional, helps debug in state dumps

    user_text = _latest_user_text(state.get("messages", []))
    if not user_text.strip():
        if DEBUG_METRICS:
            logger.info("[TRACE %s] classify empty input -> route=misc", trace_id)
        return {
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "safety_route": "ok",
            "route": "misc",
            "route_source": "empty_input_default",
            "classifier_raw": "",
            "classifier_parse_ok": True,
        }

    user_prompt = _safe_fill_user_text(CLASSIFY_USER_TMPL, user_text)
    llm = get_llm(temperature=0.0, max_tokens=220)

    # First attempt
    raw_msg, _ = _timed_invoke(
        "classify:first",
        llm,
        [SystemMessage(content=CLASSIFY_SYSTEM), HumanMessage(content=user_prompt)],
        trace_id=trace_id,
    )
    raw1 = (raw_msg.content or "").strip()
    parsed1, ok1 = _parse_classifier_json(raw1)

    # Retry when parse failed OR uncertain without triggers (often formatting/behavioral drift).
    need_retry = (not ok1) or (parsed1.get("safety_risk_level") == "uncertain" and not (parsed1.get("safety_triggers") or []))

    raw = raw1
    parsed = parsed1
    ok = ok1

    if need_retry:
        retry_system = CLASSIFY_SYSTEM.strip() + "\n\nREMINDER: Output EXACTLY one line of STRICT JSON with only the required keys. No extra text."
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
            # Non-escalatory fallback: do not force crisis on parse failure.
            raw, parsed, ok = raw1, {"safety_risk_level": "none", "safety_triggers": [], "has_protective": False, "route": "misc"}, False

    risk: RiskLevel = parsed["safety_risk_level"]
    route: Route = parsed["route"]

    # IMPORTANT: crisis only for explicit passive/active categories (not uncertain).
    safety_route: Literal["ok", "crisis"] = "crisis" if risk in ("passive", "active_no_plan", "active_with_plan") else "ok"

    if DEBUG_METRICS:
        logger.info(
            "[TRACE %s] classify result risk=%s safety_route=%s route=%s parse_ok=%s triggers=%s",
            trace_id,
            risk,
            safety_route,
            route,
            ok,
            parsed.get("safety_triggers", []),
        )

    return {
        "safety_risk_level": risk,
        "safety_triggers": parsed["safety_triggers"],
        "has_protective": parsed["has_protective"],
        "safety_route": safety_route,
        "route": route,
        "route_source": "llm",
        "classifier_raw": raw,
        "classifier_parse_ok": ok,
    }


def crisis_node(state: State) -> Dict[str, Any]:
    # crisis.txt is already user-facing.
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
            f"Resumen existente:\n{existing_summary}\n\n"
            "Actualizalo incorporando SOLO la información nueva de los mensajes recientes:"
        )
    else:
        summary_message = "Creá un resumen conciso de la conversación anterior:"

    summarizer = get_llm(temperature=0.0, max_tokens=256)
    summary_input = [SystemMessage(content=SUMMARY_SYSTEM), *messages, HumanMessage(content=summary_message)]

    resp, m = _timed_invoke("summarize", summarizer, summary_input, trace_id=trace_id)
    new_summary = (resp.content or "").strip()

    # Delete older messages; keep last K verbatim.
    to_remove: List[Any] = []
    if len(messages) > KEEP_LAST_MESSAGES:
        for msg in messages[:-KEEP_LAST_MESSAGES]:
            mid = getattr(msg, "id", None)
            if mid:
                to_remove.append(RemoveMessage(id=mid))

    if DEBUG_METRICS:
        logger.info(
            "[TRACE %s] memory summarized -> new_summary_chars=%d removed=%d (kept=%d)",
            trace_id,
            len(new_summary),
            len(to_remove),
            KEEP_LAST_MESSAGES,
        )

    out: Dict[str, Any] = {"summary": new_summary}
    if to_remove:
        out["messages"] = to_remove
    return out


# =========================================================
# DBT MINI node
# =========================================================
def dbt_mini_node(state: State) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)
    llm = get_llm(temperature=0.25, max_tokens=900)
    msgs = _llm_messages_with_summary(DBT_SYSTEM, state, trace_id=trace_id, label="dbt_mini")
    reply, _ = _timed_invoke("dbt:mini", llm, msgs, trace_id=trace_id)
    return {"messages": [AIMessage(content=reply.content or "")]}


# =========================================================
# TB-info specialist: citation-safe RAG
# =========================================================
class TBAnswerJSON(BaseModel):
    answer: str = Field(...)
    citations_used: List[int] = Field(default_factory=list)


def tb_info_node(state: State) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)

    user_text = _latest_user_text(state.get("messages", []))
    if not user_text.strip():
        return {"messages": [AIMessage(content="")]}

    tool_out = _timed_tool("retrieve_tb_docs", retrieve_tb_docs, {"query": user_text}, trace_id=trace_id)
    sources = tool_out.get("sources", []) if isinstance(tool_out, dict) else []

    sources_block = _format_sources_block(sources)
    user_prompt = TB_RAG_ANSWER_USER_TMPL.replace("{user_text}", user_text).replace("{sources_block}", sources_block)

    # IMPORTANT: summary is a SystemMessage, not injected into HumanMessage.
    summary = (state.get("summary") or "").strip()
    prefix = _system_and_summary_messages(TB_RAG_ANSWER_SYSTEM, summary)

    if DEBUG_METRICS:
        sys_tokens = _tokens(prefix)
        human_tokens = _tokens([HumanMessage(content=user_prompt)])
        logger.info(
            "[TRACE %s] [PROMPT:tb_info] prefix~%d human~%d total~%d sources=%d",
            trace_id,
            sys_tokens,
            human_tokens,
            sys_tokens + human_tokens,
            len(sources),
        )

    llm = get_llm(temperature=0.2, max_tokens=700).with_structured_output(TBAnswerJSON)

    t0 = time.perf_counter()
    result = llm.invoke([*prefix, HumanMessage(content=user_prompt)])
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if DEBUG_METRICS:
        logger.info(
            "[TRACE %s] [LLM] tb_info latency=%.1fms in_tokens~%d out_answer_chars=%d cited=%s",
            trace_id,
            dt_ms,
            _tokens([*prefix, HumanMessage(content=user_prompt)]),
            len((result.answer or "")),
            result.citations_used,
        )

    final_text = _render_answer_with_references(answer_text=result.answer, sources=sources, citations_used=result.citations_used)
    return {"messages": [AIMessage(content=final_text)]}


# =========================================================
# MISC node (meta, greetings, off-topic)
# =========================================================
def misc_node(state: State) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)
    llm = get_llm(temperature=0.2, max_tokens=350)
    msgs = _llm_messages_with_summary(MISC_SYSTEM, state, trace_id=trace_id, label="misc")
    reply, _ = _timed_invoke("misc", llm, msgs, trace_id=trace_id)
    return {"messages": [AIMessage(content=reply.content or "")]}


# =========================================================
# DBT FULL subgraph (router -> module agent)
# =========================================================
class DBTBrainSignals(BaseModel):
    emotion_intensity: Literal["low", "medium", "high"]
    impulse_or_urge_present: bool
    problem_solvable_now: bool
    interpersonal_context: bool
    attention_or_judgment_issue: bool


class DBTBrainOut(BaseModel):
    module: DBTModule
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_brief: str
    signals: DBTBrainSignals


def dbt_brain_router_node(state: State) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)

    user_text = _latest_user_text(state.get("messages", []))
    if not user_text.strip():
        return {
            "dbt_module": "MIND",
            "dbt_confidence": 0.0,
            "dbt_rationale_brief": "Empty input.",
            "dbt_signals": {},
            "dbt_router_raw": "",
        }

    llm = get_llm(temperature=0.0, max_tokens=220).with_structured_output(DBTBrainOut)

    msgs = [SystemMessage(content=DBT_BRAIN_ROUTER_SYSTEM), HumanMessage(content=user_text)]
    t0 = time.perf_counter()
    result = llm.invoke(msgs)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if DEBUG_METRICS:
        logger.info(
            "[TRACE %s] [LLM] dbt_brain latency=%.1fms in_tokens~%d module=%s conf=%.2f",
            trace_id,
            dt_ms,
            _tokens(msgs),
            result.module,
            float(result.confidence),
        )

    return {
        "dbt_module": result.module,
        "dbt_confidence": float(result.confidence),
        "dbt_rationale_brief": result.rationale_brief,
        "dbt_signals": result.signals.model_dump(),
        "dbt_router_raw": result.model_dump_json(),
    }


def _dbt_module_agent(system_prompt: str, state: State, *, label: str) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)
    llm = get_llm(temperature=0.25, max_tokens=900)
    msgs = _llm_messages_with_summary(system_prompt, state, trace_id=trace_id, label=label)
    reply, _ = _timed_invoke(f"dbt:{label}", llm, msgs, trace_id=trace_id)
    return {"messages": [AIMessage(content=reply.content or "")]}


def dbt_dt_node(state: State) -> Dict[str, Any]:
    return _dbt_module_agent(DBT_DT_SYSTEM, state, label="DT")


def dbt_mind_node(state: State) -> Dict[str, Any]:
    return _dbt_module_agent(DBT_MIND_SYSTEM, state, label="MIND")


def dbt_er_node(state: State) -> Dict[str, Any]:
    return _dbt_module_agent(DBT_ER_SYSTEM, state, label="ER")


def dbt_ie_node(state: State) -> Dict[str, Any]:
    return _dbt_module_agent(DBT_IE_SYSTEM, state, label="IE")


def branch_after_dbt_brain(state: State) -> DBTModule:
    return state.get("dbt_module", "MIND")


def build_dbt_full_subgraph():
    sg = StateGraph(State)
    sg.add_node("dbt_brain", dbt_brain_router_node)
    sg.add_node("DT", dbt_dt_node)
    sg.add_node("MIND", dbt_mind_node)
    sg.add_node("ER", dbt_er_node)
    sg.add_node("IE", dbt_ie_node)

    sg.add_edge(START, "dbt_brain")
    sg.add_conditional_edges("dbt_brain", branch_after_dbt_brain, {"DT": "DT", "MIND": "MIND", "ER": "ER", "IE": "IE"})
    sg.add_edge("DT", END)
    sg.add_edge("MIND", END)
    sg.add_edge("ER", END)
    sg.add_edge("IE", END)

    return sg.compile()


DBT_FULL_SUBGRAPH = build_dbt_full_subgraph()


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

    g.add_conditional_edges("memory", branch_after_classify, {"crisis": "crisis", "tb_info": "tb_info", "dbt": "dbt", "misc": "misc"})

    g.add_edge("crisis", END)
    g.add_edge("tb_info", END)
    g.add_edge("dbt", END)
    g.add_edge("misc", END)

    return g.compile(checkpointer=MemorySaver())


# Public compiled graphs
GRAPH_DBT_MINI = build_graph(dbt_node=dbt_mini_node)
GRAPH_DBT_FULL = build_graph(dbt_node=DBT_FULL_SUBGRAPH)


# Emit prompt size report once at import (if enabled)
_log_prompt_sizes_once()

