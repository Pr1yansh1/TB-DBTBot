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
# TBTST_DEBUG_METRICS=1
# TBTST_LOG_PROMPT_SIZES=1
# TBTST_LOG_LEVEL=INFO|DEBUG
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
Sos un asistente de apoyo para una app de TB y bienestar.
Tu tarea acá es responder mensajes "misc": saludos, dudas sobre el sistema, o cosas fuera de TB/DBT.
Respondé en español (vos), con calidez y brevedad.
Si el usuario pide algo concreto, pedí 1 pregunta breve para orientar.
No uses jerga. No inventes datos médicos.
""".strip()

# =========================================================
# State types
# =========================================================
RiskLevel = Literal["none", "passive", "active_no_plan", "active_with_plan", "uncertain"]
Route = Literal["faq", "dbt", "misc"]

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
    dbt_skill_status: DBTSkillStatus  # status of active skill/thread
    dbt_recent_skills: List[RecentSkillEvent]  # last 3-4 skill events, capped

    # short-term memory management (rolling summary of conversation only)
    summary: str

    # debugging / tracing
    last_trace_id: str


# =========================================================
# Memory / context management settings
# =========================================================
MAX_TOKENS_BEFORE_SUMMARY = 4500
KEEP_LAST_MESSAGES = 8  # what remains verbatim after summarization

# Budget for "recent messages" appended after system+summary.
# If you increase system prompts, raise this too.
LLM_RECENT_MESSAGES_MAX_TOKENS = 1800

# This is the "hard floor" for coherence: keep at least N recent messages even under token pressure.
# Typical: 8–14 messages (~4–7 turns).
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
    """
    Best-effort per-turn trace id to correlate logs across nodes.
    Uses most recent human message id if available; otherwise random short uuid.
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


def _is_throttle_error(e: Exception) -> bool:
    msg = str(e)
    return ("ThrottlingException" in msg) or ("Too many requests" in msg) or ("Rate exceeded" in msg)


def _timed_invoke(name: str, llm: Any, messages: List[BaseMessage], *, trace_id: str) -> Tuple[Any, CallMetrics]:
    t0 = time.perf_counter()
    last_exc: Exception | None = None

    for attempt in range(6):  # 1 initial + 5 retries
        try:
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
# Core helpers
# =========================================================
def _latest_user_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _safe_fill_user_text(template: str, user_text: str) -> str:
    return template.replace("{user_text}", user_text)


# --- Robust recent-context selection (fixes continuity for good) ---
_msg_token_cache: Dict[str, int] = {}


def _message_cache_key(m: BaseMessage) -> str:
    # Prefer stable IDs (LangChain often sets them); otherwise fall back to object id.
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
    """
    Deterministic, role-preserving recent context selection.

    Key properties:
    - Never uses start_on="human" (which can drop AI framing messages).
    - Always keeps at least `min_messages` from the tail (or as many as exist).
    - Expands backwards until token budget is reached.
    - If token budget is too small, still keeps at least the last 2 messages.
    """
    if not messages:
        return []

    # Always keep a small tail to preserve the immediate conversational frame.
    hard_min = max(2, min_messages)
    selected_rev: List[BaseMessage] = []
    total = 0

    for m in reversed(messages):
        mt = _message_tokens(m)

        # If we already satisfied the minimum and adding this would exceed budget, stop.
        if len(selected_rev) >= hard_min and (total + mt) > max_tokens:
            break

        selected_rev.append(m)
        total += mt

        # If we crossed budget but haven't met minimum, keep going (coherence > budget),
        # but don't go crazy: once we meet minimum, we allow early stop on next iteration.
        if total >= max_tokens and len(selected_rev) >= hard_min:
            break

    selected = list(reversed(selected_rev))
    return selected


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

    # Robust selection: keeps assistant+user frames together.
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


def _ensure_dbt_defaults(state: State) -> None:
    state.setdefault("dbt_mode", "connect")
    state.setdefault("dbt_continuity", "new")
    state.setdefault("dbt_active_skill", "")
    state.setdefault("dbt_skill_status", "none")
    state.setdefault("dbt_recent_skills", [])


def _dbt_state_system_note(state: State) -> SystemMessage:
    _ensure_dbt_defaults(state)

    active_skill = (state.get("dbt_active_skill") or "").strip()
    skill_status: DBTSkillStatus = state.get("dbt_skill_status", "none")
    recent = state.get("dbt_recent_skills", []) or []

    recent_lines: List[str] = []
    for ev in recent[-RECENT_SKILLS_MAX:]:
        s = (ev.get("skill") or "").strip()
        st = ev.get("status", "none")
        if s:
            recent_lines.append(f"- {s}: {st}")
    recent_block = "\n".join(recent_lines) if recent_lines else "- (none)"

    return SystemMessage(
        content=(
            "INTERNAL DBT STATE (for continuity; do not show to the user):\n"
            f"- active_skill: {active_skill or '(none)'}\n"
            f"- active_skill_status: {skill_status}\n"
            f"- recent_skill_events (most recent last; max {RECENT_SKILLS_MAX}):\n{recent_block}\n"
            "\n"
            "Hard rules:\n"
            "- If active_skill_status is 'declined' or 'paused': DO NOT offer new skills. Use mode=connect.\n"
            "- If active_skill_status is 'accepted' and user replies briefly/affirmatively: continue mode=coach on the same skill.\n"
        )
    )


def _dbt_mode_instructions(state: State) -> SystemMessage:
    _ensure_dbt_defaults(state)

    mode: DBTMode = state.get("dbt_mode", "connect")
    active_skill = (state.get("dbt_active_skill") or "").strip()

    if mode == "connect":
        rules = (
            "MODE=CONNECT:\n"
            "- Respond in Spanish.\n"
            "- Validate/empathize in 1–2 sentences.\n"
            "- Ask EXACTLY 1 brief question.\n"
            "- Do NOT teach DBT skills, do NOT list steps, do NOT offer multiple options.\n"
            "- Keep it short (roughly <= 120–180 words).\n"
        )
    elif mode == "offer":
        rules = (
            "MODE=OFFER:\n"
            "- Respond in Spanish.\n"
            "- Offer EXACTLY ONE DBT skill (name + 1-line definition).\n"
            "- Provide at most 2–3 bullet steps.\n"
            "- End with 1 consent question (ask if they want to try it now).\n"
            "- Do NOT add a second skill.\n"
            "- Keep it short (roughly <= 180–250 words).\n"
        )
    else:  # coach
        rules = (
            "MODE=COACH:\n"
            "- Respond in Spanish.\n"
            f"- Continue ONLY the active skill thread: {active_skill or '(choose one and keep it consistent)'}.\n"
            "- Provide ONE next step or mini-exercise (2–4 bullets).\n"
            "- End with 1 brief check-in question.\n"
            "- Keep it short (roughly <= 220–320 words).\n"
        )

    return SystemMessage(content=rules)


def _dbt_max_tokens_for_mode(state: State) -> int:
    _ensure_dbt_defaults(state)
    mode: DBTMode = state.get("dbt_mode", "connect")
    if mode == "connect":
        return 260
    if mode == "offer":
        return 420
    return 520


def _append_recent_skill_event(
    state: State,
    *,
    skill: str,
    status: DBTSkillStatus,
    trace_id: str,
    note: str = "",
) -> None:
    _ensure_dbt_defaults(state)
    skill = (skill or "").strip()
    if not skill:
        return

    ev: RecentSkillEvent = {"skill": skill, "status": status, "trace_id": trace_id}
    if note:
        ev["note"] = note

    recent: List[RecentSkillEvent] = state.get("dbt_recent_skills", []) or []
    recent = [*recent, ev]
    if len(recent) > RECENT_SKILLS_MAX:
        recent = recent[-RECENT_SKILLS_MAX:]
    state["dbt_recent_skills"] = recent


def _apply_dbt_hard_overrides(
    state: State, router_mode: DBTMode, router_continuity: DBTContinuity
) -> Tuple[DBTMode, DBTContinuity]:
    _ensure_dbt_defaults(state)
    prev_status: DBTSkillStatus = state.get("dbt_skill_status", "none")
    if prev_status in ("declined", "paused"):
        return "connect", "same"
    return router_mode, router_continuity


def _coerce_agent_skill_state(
    *,
    prev_active_skill: str,
    prev_skill_status: DBTSkillStatus,
    mode: DBTMode,
    agent_skill: str,
    agent_status: DBTSkillStatus,
) -> Tuple[str, DBTSkillStatus]:
    prev_active_skill = (prev_active_skill or "").strip()
    agent_skill = (agent_skill or "").strip()

    if prev_skill_status in ("declined", "paused"):
        if agent_status in ("accepted", "completed"):
            return (agent_skill or prev_active_skill), agent_status
        return prev_active_skill, prev_skill_status

    if mode == "connect":
        return prev_active_skill, prev_skill_status

    if mode == "offer":
        if agent_status in ("declined", "paused"):
            return prev_active_skill, agent_status
        return (agent_skill or prev_active_skill), "offered"

    if mode == "coach":
        if agent_status == "completed":
            return (agent_skill or prev_active_skill), "completed"
        if agent_status == "accepted":
            return (agent_skill or prev_active_skill), "accepted"
        return (agent_skill or prev_active_skill), "accepted"

    return prev_active_skill, prev_skill_status


T = TypeVar("T", bound=BaseModel)


def _invoke_structured_with_retry(
    *,
    llm_structured: Any,
    messages: List[BaseMessage],
    trace_id: str,
    label: str,
    retry_hint: str,
) -> Tuple[Optional[T], str]:
    """
    Best-effort structured-output invocation.

    Returns (result_or_none, raw_text).
    - If structured parsing fails twice, returns (None, raw_text_last).
    """
    raw_last = ""
    try:
        result = llm_structured.invoke(messages)
        # Many structured wrappers expose the raw via .raw, but not consistently.
        raw_last = getattr(result, "model_dump_json", lambda: "")()
        return cast(T, result), raw_last
    except Exception as e1:
        if DEBUG_METRICS:
            logger.warning("[TRACE %s] structured parse failed for %s: %s", trace_id, label, str(e1))

    # Retry with stricter hint appended
    stricter = [*messages, SystemMessage(content=retry_hint)]
    try:
        result2 = llm_structured.invoke(stricter)
        raw_last = getattr(result2, "model_dump_json", lambda: "")()
        return cast(T, result2), raw_last
    except Exception as e2:
        if DEBUG_METRICS:
            logger.warning("[TRACE %s] structured retry failed for %s: %s", trace_id, label, str(e2))
        # Try to capture some raw content if the underlying object has it
        raw_last = ""
        return None, raw_last


# =========================================================
# Nodes: classify + safety
# =========================================================
class ClassifyOut(BaseModel):
    safety_risk_level: RiskLevel = Field(...)
    safety_triggers: List[str] = Field(default_factory=list)
    has_protective: bool = Field(False)
    route: Route = Field(...)


def _coerce_and_validate_classifier(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
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


def classify_node(state: State) -> Dict[str, Any]:
    trace_id = _trace_id_for_state(state)
    state["last_trace_id"] = trace_id

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
            raw, parsed, ok = raw1, {
                "safety_risk_level": "none",
                "safety_triggers": [],
                "has_protective": False,
                "route": "misc",
            }, False

    risk: RiskLevel = parsed["safety_risk_level"]
    route: Route = parsed["route"]

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
    llm = get_llm(temperature=0.25, max_tokens=600)
    msgs = _llm_messages_with_summary(DBT_SYSTEM, state, trace_id=trace_id, label="dbt_mini")
    reply, _ = _timed_invoke("dbt:mini", llm, msgs, trace_id=trace_id)
    return {"messages": [AIMessage(content=reply.content or "")]}


# =========================================================
# TB-info specialist: citation-safe RAG
# =========================================================
class TBAnswerJSON(BaseModel):
    answer: str = Field(...)
    citations_used: List[int] = Field(default_factory=list)


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


def tb_info_node(state: State) -> Dict[str, Any]:
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)

    user_text = _latest_user_text(state.get("messages", []))
    if not user_text.strip():
        return {"messages": [AIMessage(content="")]}

    tool_out = _timed_tool("retrieve_tb_docs", retrieve_tb_docs, {"query": user_text}, trace_id=trace_id)
    sources = tool_out.get("sources", []) if isinstance(tool_out, dict) else []

    sources_block = _format_sources_block(sources)
    user_prompt = TB_RAG_ANSWER_USER_TMPL.replace("{user_text}", user_text).replace("{sources_block}", sources_block)

    summary = (state.get("summary") or "").strip()
    prefix = _system_and_summary_messages(TB_RAG_ANSWER_SYSTEM, summary)

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
    mode: DBTMode
    continuity: DBTContinuity
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_brief: str
    signals: DBTBrainSignals


class DBTAgentOut(BaseModel):
    """
    Structured output from the DBT module agent.
    - message: user-facing Spanish
    - skill: short name of the skill used/offered/coached (if applicable)
    - skill_status: internal status update
    """
    message: str = Field(..., description="User-facing message in Spanish.")
    skill: str = Field(default="", description="Short DBT skill name (e.g., STOP, TIP, DEAR MAN).")
    skill_status: DBTSkillStatus = Field(default="none", description="Internal skill status update.")


def dbt_brain_router_node(state: State) -> Dict[str, Any]:
    """
    DBT router: chooses module + mode + continuity for THIS turn.
    It sees summary + recent messages + internal DBT state note.
    """
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)
    _ensure_dbt_defaults(state)

    user_text = _latest_user_text(state.get("messages", []))
    if not user_text.strip():
        return {
            "dbt_module": "MIND",
            "dbt_confidence": 0.0,
            "dbt_rationale_brief": "Empty input.",
            "dbt_signals": {},
            "dbt_router_raw": "",
            "dbt_mode": "connect",
            "dbt_continuity": "unclear",
        }

    llm_structured = get_llm(temperature=0.0, max_tokens=280).with_structured_output(DBTBrainOut)

    base_msgs = _llm_messages_with_summary(DBT_BRAIN_ROUTER_SYSTEM, state, trace_id=trace_id, label="dbt_brain")
    msgs = [base_msgs[0], _dbt_state_system_note(state), *base_msgs[1:]]

    retry_hint = (
        "STRICT OUTPUT REMINDER:\n"
        "Return ONLY valid JSON that matches the schema exactly. No extra keys. No prose.\n"
        "If uncertain, choose the best module and set confidence low."
    )

    t0 = time.perf_counter()
    result, raw_json = _invoke_structured_with_retry(
        llm_structured=llm_structured,
        messages=msgs,
        trace_id=trace_id,
        label="dbt_brain",
        retry_hint=retry_hint,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if result is None:
        # Fail-safe: do not crash or emit random state; default to connect + MIND.
        if DEBUG_METRICS:
            logger.warning("[TRACE %s] dbt_brain fallback (structured failed)", trace_id)
        return {
            "dbt_module": "MIND",
            "dbt_confidence": 0.0,
            "dbt_rationale_brief": "Router parse failed; fallback to connect.",
            "dbt_signals": {},
            "dbt_router_raw": raw_json or "",
            "dbt_mode": "connect",
            "dbt_continuity": "unclear",
        }

    final_mode, final_cont = _apply_dbt_hard_overrides(state, result.mode, result.continuity)

    if DEBUG_METRICS:
        logger.info(
            "[TRACE %s] [LLM] dbt_brain latency=%.1fms in_tokens~%d module=%s mode=%s cont=%s active_status=%s",
            trace_id,
            dt_ms,
            _tokens(msgs),
            result.module,
            final_mode,
            final_cont,
            state.get("dbt_skill_status", "none"),
        )

    return {
        "dbt_module": result.module,
        "dbt_confidence": float(result.confidence),
        "dbt_rationale_brief": result.rationale_brief,
        "dbt_signals": result.signals.model_dump(),
        "dbt_router_raw": raw_json or result.model_dump_json(),
        "dbt_mode": final_mode,
        "dbt_continuity": final_cont,
    }


def _dbt_module_agent(system_prompt: str, state: State, *, label: str) -> Dict[str, Any]:
    """
    Module agent receives:
    - GLOBAL_RULES + module prompt
    - summary
    - internal DBT state note
    - mode constraints (connect/offer/coach)

    Returns DBTAgentOut:
    - message (Spanish)
    - skill + skill_status for internal state updates
    """
    trace_id = state.get("last_trace_id") or _trace_id_for_state(state)
    _ensure_dbt_defaults(state)

    max_tokens = _dbt_max_tokens_for_mode(state)
    llm_structured = get_llm(temperature=0.25, max_tokens=max_tokens).with_structured_output(DBTAgentOut)

    base_msgs = _llm_messages_with_summary(system_prompt, state, trace_id=trace_id, label=label)

    output_contract = SystemMessage(
        content=(
            "OUTPUT CONTRACT (internal; do not show JSON to user):\n"
            "Return an object with fields:\n"
            "- message: Spanish user-facing text.\n"
            "- skill: short skill name (if applicable; else empty).\n"
            "- skill_status: one of [none, offered, accepted, declined, paused, completed].\n"
            "\n"
            "Rules:\n"
            "- If mode=connect: skill_status must be 'none' and skill must be empty.\n"
            "- If mode=offer: skill_status must be 'offered' and skill must be set.\n"
            "- If mode=coach: skill_status should be 'accepted' (or 'completed' if wrapping up) and skill must be set.\n"
            "- If user explicitly says 'no', 'not now', 'just listen', or similar: set skill_status to 'paused' or 'declined'.\n"
        )
    )

    msgs: List[BaseMessage] = [
        base_msgs[0],
        _dbt_mode_instructions(state),
        _dbt_state_system_note(state),
        output_contract,
        *base_msgs[1:],
    ]

    retry_hint = (
        "STRICT OUTPUT REMINDER:\n"
        "Return ONLY valid JSON matching the schema exactly. No prose outside JSON.\n"
        "The 'message' field must be Spanish user-facing text."
    )

    t0 = time.perf_counter()
    result, _raw = _invoke_structured_with_retry(
        llm_structured=llm_structured,
        messages=msgs,
        trace_id=trace_id,
        label=f"dbt:{label}",
        retry_hint=retry_hint,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if result is None:
        # Fail-safe response: keep continuity, do not mutate DBT state
        if DEBUG_METRICS:
            logger.warning("[TRACE %s] dbt:%s fallback (structured failed)", trace_id, label)
        fallback_text = (
            "Entiendo. Gracias por contármelo. "
            "¿Qué fue lo más difícil de ese momento para vos: el comentario en sí, la vergüenza, o la sensación de quedarte congelado/a?"
        )
        return {"messages": [AIMessage(content=fallback_text)]}

    if DEBUG_METRICS:
        logger.info(
            "[TRACE %s] [LLM] dbt:%s latency=%.1fms in_tokens~%d out_chars=%d skill=%s status=%s",
            trace_id,
            label,
            dt_ms,
            _tokens(msgs),
            len(result.message or ""),
            (result.skill or ""),
            result.skill_status,
        )

    mode: DBTMode = state.get("dbt_mode", "connect")
    prev_skill = (state.get("dbt_active_skill") or "").strip()
    prev_status: DBTSkillStatus = state.get("dbt_skill_status", "none")

    new_skill, new_status = _coerce_agent_skill_state(
        prev_active_skill=prev_skill,
        prev_skill_status=prev_status,
        mode=mode,
        agent_skill=(result.skill or ""),
        agent_status=result.skill_status,
    )

    updates: Dict[str, Any] = {"messages": [AIMessage(content=(result.message or "").strip())]}

    if mode != "connect":
        if new_skill:
            updates["dbt_active_skill"] = new_skill
        updates["dbt_skill_status"] = new_status

        event_skill = (result.skill or "").strip() or new_skill
        if event_skill and new_status != "none":
            _append_recent_skill_event(
                state,
                skill=event_skill,
                status=new_status,
                trace_id=trace_id,
                note=f"module={label} mode={mode}",
            )
            updates["dbt_recent_skills"] = state.get("dbt_recent_skills", [])

    return updates


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
    sg.add_conditional_edges(
        "dbt_brain",
        branch_after_dbt_brain,
        {"DT": "DT", "MIND": "MIND", "ER": "ER", "IE": "IE"},
    )
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
GRAPH_DBT_FULL = build_graph(dbt_node=DBT_FULL_SUBGRAPH)

# Emit prompt size report once at import (if enabled)
_log_prompt_sizes_once()

