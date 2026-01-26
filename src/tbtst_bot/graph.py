# src/tbtst_bot/graph.py
from __future__ import annotations

import json
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


# =========================================================
# Memory / context management settings
# =========================================================
# When the thread gets longer than this (approx tokens), we update summary + delete older messages.
MAX_TOKENS_BEFORE_SUMMARY = 2200

# After summarizing, keep the last K messages verbatim
KEEP_LAST_MESSAGES = 8

# Target max tokens for what we SEND to the model (recent messages only).
# We keep headroom for system prompts + summary by dynamically reducing this further per call.
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
# Helpers
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
    Normalization is conservative: if required fields are missing or invalid, parse_ok=False.
    """
    try:
        obj = ClassifyOut.model_validate(data)
        # ensure triggers is list of short strings
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


def _render_answer_with_references(
    answer_text: str,
    sources: List[Dict[str, Any]],
    citations_used: List[int],
) -> str:
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
    """
    Trim what we SEND to the model (doesn't mutate graph state).
    Keeps the most recent content and ensures we start on a human message.
    """
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


def _llm_messages_with_summary(system_prompt: str, state: State) -> List[BaseMessage]:
    """
    Construct model input:
      - System (GLOBAL_RULES + module prompt)
      - Optional summary as SystemMessage
      - Trimmed recent messages, with headroom.
    """
    summary = (state.get("summary") or "").strip()
    prefix = _system_and_summary_messages(system_prompt, summary)

    # Budget headroom: compute tokens taken by system+summary, reduce budget for recent messages.
    base_tokens = count_tokens_approximately(prefix)
    recent_budget = max(256, LLM_RECENT_MESSAGES_MAX_TOKENS - base_tokens)

    recent = _trim_for_llm(state.get("messages", []), max_tokens=recent_budget)
    return [*prefix, *recent]


# =========================================================
# Nodes: classify + safety
# =========================================================
def classify_node(state: State) -> Dict[str, Any]:
    user_text = _latest_user_text(state.get("messages", []))
    if not user_text.strip():
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
    raw_msg = llm.invoke([SystemMessage(content=CLASSIFY_SYSTEM), HumanMessage(content=user_prompt)])
    raw1 = (raw_msg.content or "").strip()
    parsed1, ok1 = _parse_classifier_json(raw1)

    # Retry ONLY when:
    # - parse failed OR
    # - model returned "uncertain" without clear self-harm/death triggers (often a sign of confusion)
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
        raw_msg2 = llm.invoke([SystemMessage(content=retry_system), HumanMessage(content=user_prompt)])
        raw2 = (raw_msg2.content or "").strip()
        parsed2, ok2 = _parse_classifier_json(raw2)

        # Prefer a successful parse; otherwise keep the first attempt but make it non-escalatory.
        if ok2:
            raw, parsed, ok = raw2, parsed2, ok2
        else:
            # Non-escalatory safe fallback
            raw, parsed, ok = raw1, {"safety_risk_level": "none", "safety_triggers": [], "has_protective": False, "route": "misc"}, False

    risk: RiskLevel = parsed["safety_risk_level"]
    route: Route = parsed["route"]

    # IMPORTANT FIX:
    # crisis is only for explicit passive/active categories.
    # "uncertain" stays "ok" and can be handled by downstream content route.
    safety_route: Literal["ok", "crisis"] = (
        "crisis" if risk in ("passive", "active_no_plan", "active_with_plan") else "ok"
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
    # crisis.txt is already a fully-formed user-facing message.
    return {"messages": [AIMessage(content=CRISIS_TEXT)]}


# =========================================================
# Memory manager node (summarize + delete)
# =========================================================
def memory_manager_node(state: State) -> Dict[str, Any]:
    messages: List[BaseMessage] = state.get("messages", [])
    if not messages:
        return {}

    approx_tokens = count_tokens_approximately(messages)
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
    resp = summarizer.invoke(summary_input)
    new_summary = (resp.content or "").strip()

    # Delete older messages from state; keep last K verbatim.
    to_remove: List[Any] = []
    if len(messages) > KEEP_LAST_MESSAGES:
        for m in messages[:-KEEP_LAST_MESSAGES]:
            mid = getattr(m, "id", None)
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
    llm = get_llm(temperature=0.25, max_tokens=900)
    reply = llm.invoke(_llm_messages_with_summary(DBT_SYSTEM, state))
    return {"messages": [AIMessage(content=reply.content or "")]}


# =========================================================
# TB-info specialist: citation-safe RAG
# =========================================================
class TBAnswerJSON(BaseModel):
    answer: str = Field(...)
    citations_used: List[int] = Field(default_factory=list)


def tb_info_node(state: State) -> Dict[str, Any]:
    user_text = _latest_user_text(state.get("messages", []))
    if not user_text.strip():
        return {"messages": [AIMessage(content="")]}

    tool_out = retrieve_tb_docs.invoke({"query": user_text})
    sources = tool_out.get("sources", []) if isinstance(tool_out, dict) else []

    sources_block = _format_sources_block(sources)
    user_prompt = (
        TB_RAG_ANSWER_USER_TMPL.replace("{user_text}", user_text).replace("{sources_block}", sources_block)
    )

    # IMPORTANT FIX: summary is a SystemMessage, not injected into HumanMessage.
    summary = (state.get("summary") or "").strip()
    prefix = _system_and_summary_messages(TB_RAG_ANSWER_SYSTEM, summary)

    llm = get_llm(temperature=0.2, max_tokens=700)
    result = llm.with_structured_output(TBAnswerJSON).invoke(
        [
            *prefix,
            HumanMessage(content=user_prompt),
        ]
    )

    final_text = _render_answer_with_references(
        answer_text=result.answer, sources=sources, citations_used=result.citations_used
    )
    return {"messages": [AIMessage(content=final_text)]}


# =========================================================
# MISC node (meta, greetings, off-topic)
# =========================================================
def misc_node(state: State) -> Dict[str, Any]:
    llm = get_llm(temperature=0.2, max_tokens=350)
    reply = llm.invoke(_llm_messages_with_summary(MISC_SYSTEM, state))
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
    """Classifies one DBT module. Not user-facing."""
    user_text = _latest_user_text(state.get("messages", []))
    if not user_text.strip():
        return {
            "dbt_module": "MIND",
            "dbt_confidence": 0.0,
            "dbt_rationale_brief": "Empty input.",
            "dbt_signals": {},
            "dbt_router_raw": "",
        }

    llm = get_llm(temperature=0.0, max_tokens=220)
    result = llm.with_structured_output(DBTBrainOut).invoke(
        [SystemMessage(content=DBT_BRAIN_ROUTER_SYSTEM), HumanMessage(content=user_text)]
    )

    return {
        "dbt_module": result.module,
        "dbt_confidence": float(result.confidence),
        "dbt_rationale_brief": result.rationale_brief,
        "dbt_signals": result.signals.model_dump(),
        "dbt_router_raw": result.model_dump_json(),
    }


def _dbt_module_agent(system_prompt: str, state: State) -> Dict[str, Any]:
    llm = get_llm(temperature=0.25, max_tokens=900)
    reply = llm.invoke(_llm_messages_with_summary(system_prompt, state))
    return {"messages": [AIMessage(content=reply.content or "")]}


def dbt_dt_node(state: State) -> Dict[str, Any]:
    return _dbt_module_agent(DBT_DT_SYSTEM, state)


def dbt_mind_node(state: State) -> Dict[str, Any]:
    return _dbt_module_agent(DBT_MIND_SYSTEM, state)


def dbt_er_node(state: State) -> Dict[str, Any]:
    return _dbt_module_agent(DBT_ER_SYSTEM, state)


def dbt_ie_node(state: State) -> Dict[str, Any]:
    return _dbt_module_agent(DBT_IE_SYSTEM, state)


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

    # Route AFTER memory management
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

