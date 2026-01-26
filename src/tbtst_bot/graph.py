
# src/tbtst_bot/graph.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Set

from pydantic import BaseModel, Field

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

# NEW: Global response rules applied to ALL user-facing responses
GLOBAL_RULES = load_prompt("global_response_rules.txt")

# DBT mini (existing)
DBT_SYSTEM = load_prompt("dbt_system.txt")

# DBT full (new)
DBT_BRAIN_ROUTER_SYSTEM = load_prompt("dbt_brain.txt")
DBT_DT_SYSTEM = load_prompt("DBT/dt_prompt.txt")
DBT_MIND_SYSTEM = load_prompt("DBT/mind_prompt.txt")
DBT_ER_SYSTEM = load_prompt("DBT/er_prompt.txt")
DBT_IE_SYSTEM = load_prompt("DBT/ie_prompt.txt")

# TB RAG (existing)
TB_RAG_ANSWER_SYSTEM = load_prompt("tb_rag_answer_system.txt")
TB_RAG_ANSWER_USER_TMPL = load_prompt("tb_rag_answer_user.txt")


# =========================================================
# State
# =========================================================
RiskLevel = Literal["none", "passive", "active_no_plan", "active_with_plan", "uncertain"]
Route = Literal["faq", "dbt", "psychoed"]

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

# After summarizing, keep the last K messages verbatim (prevents the model from feeling “amnesiac”).
KEEP_LAST_MESSAGES = 8

# For each LLM call, we still trim the messages we pass in to stay within a safe window.
# (This does NOT delete state; it only controls what goes to the model.)
LLM_INPUT_MAX_TOKENS = 1800

# Summary update prompt (short, factual, avoids repetition)
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


def _parse_classifier_json(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "safety_risk_level": "uncertain",
            "safety_triggers": [],
            "has_protective": False,
            "route": "dbt",
            "raw": raw,
        }

    risk = data.get("safety_risk_level", "uncertain")
    if risk not in ["none", "passive", "active_no_plan", "active_with_plan", "uncertain"]:
        risk = "uncertain"

    triggers = data.get("safety_triggers", [])
    if not isinstance(triggers, list):
        triggers = []

    has_protective = bool(data.get("has_protective", False))

    route = data.get("route", "dbt")
    if route not in ["faq", "dbt", "psychoed"]:
        route = "dbt"

    return {
        "safety_risk_level": risk,
        "safety_triggers": triggers,
        "has_protective": has_protective,
        "route": route,
        "raw": raw,
    }


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


def _trim_for_llm(messages: List[BaseMessage], *, max_tokens: int = LLM_INPUT_MAX_TOKENS) -> List[BaseMessage]:
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
        # No tools in this graph right now, but leaving end_on is harmless.
        end_on=("human", "tool"),
    )


def _llm_messages_with_summary(system_prompt: str, state: State) -> List[BaseMessage]:
    """
    Construct model input:
      System prompt (combined: GLOBAL_RULES + module-specific system prompt)
      + optional rolling summary (as additional system context)
      + trimmed recent messages
    """
    recent = _trim_for_llm(state.get("messages", []), max_tokens=LLM_INPUT_MAX_TOKENS)
    summary = (state.get("summary") or "").strip()

    combined_system = f"{GLOBAL_RULES.strip()}\n\n---\n\n{system_prompt.strip()}".strip()

    if summary:
        summary_msg = SystemMessage(
            content=(
                "Conversation memory (authoritative, concise). "
                "Use it to avoid repeating questions and keep continuity:\n"
                f"{summary}"
            )
        )
        return [SystemMessage(content=combined_system), summary_msg, *recent]

    return [SystemMessage(content=combined_system), *recent]


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
            "route": "faq",
            "route_source": "empty_input_default",
            "classifier_raw": "",
        }

    user_prompt = _safe_fill_user_text(CLASSIFY_USER_TMPL, user_text)

    llm = get_llm(temperature=0.0, max_tokens=220)
    raw_msg = llm.invoke([SystemMessage(content=CLASSIFY_SYSTEM), HumanMessage(content=user_prompt)])
    raw = raw_msg.content or ""
    parsed = _parse_classifier_json(raw)

    risk: RiskLevel = parsed["safety_risk_level"]
    route: Route = parsed["route"]

    safety_route: Literal["ok", "crisis"] = (
        "crisis" if risk in ("passive", "active_no_plan", "active_with_plan", "uncertain") else "ok"
    )

    return {
        "safety_risk_level": risk,
        "safety_triggers": parsed["safety_triggers"],
        "has_protective": parsed["has_protective"],
        "safety_route": safety_route,
        "route": route,
        "route_source": "llm",
        "classifier_raw": raw,
    }


def crisis_node(state: State) -> Dict[str, Any]:
    # crisis.txt is already a fully-formed user-facing message, so we do not wrap it with system rules here.
    # If you want global language/pronoun rules here too, rewrite crisis.txt itself to comply.
    return {"messages": [AIMessage(content=CRISIS_TEXT)]}


# =========================================================
# Memory manager node (LangGraph docs pattern: summarize + delete)
# Called after classify, before routing to response nodes.
# =========================================================
def memory_manager_node(state: State) -> Dict[str, Any]:
    messages: List[BaseMessage] = state.get("messages", [])
    if not messages:
        return {}

    # Only summarize when we’re getting “too big”
    approx_tokens = count_tokens_approximately(messages)
    if approx_tokens < MAX_TOKENS_BEFORE_SUMMARY:
        return {}

    existing_summary = (state.get("summary") or "").strip()

    if existing_summary:
        summary_message = (
            f"This is a summary of the conversation to date:\n{existing_summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a concise summary of the conversation above:"

    # We summarize using the full messages (Bedrock can handle it), but keep output small.
    summarizer = get_llm(temperature=0.0, max_tokens=256)
    summary_input = [SystemMessage(content=SUMMARY_SYSTEM), *messages, HumanMessage(content=summary_message)]
    resp = summarizer.invoke(summary_input)
    new_summary = (resp.content or "").strip()

    # Delete older messages from state (per docs) to keep thread small.
    # Keep last K messages verbatim so the agent feels consistent.
    if len(messages) > KEEP_LAST_MESSAGES:
        to_remove = [RemoveMessage(id=m.id) for m in messages[:-KEEP_LAST_MESSAGES] if getattr(m, "id", None)]
    else:
        to_remove = []

    out: Dict[str, Any] = {"summary": new_summary}
    if to_remove:
        out["messages"] = to_remove
    return out


# =========================================================
# DBT MINI node (existing behavior, but with bounded context + larger output)
# =========================================================
def dbt_mini_node(state: State) -> Dict[str, Any]:
    llm = get_llm(temperature=0.25, max_tokens=900)
    reply = llm.invoke(_llm_messages_with_summary(DBT_SYSTEM, state))
    return {"messages": [AIMessage(content=reply.content or "")]}


# =========================================================
# TB-info specialist: citation-safe RAG (existing)
# (Not strongly affected by long history, but we still include summary for continuity.)
# =========================================================
class TBAnswerJSON(BaseModel):
    answer: str = Field(...)
    citations_used: List[int] = Field(default_factory=list)


def tb_info_node(state: State) -> Dict[str, Any]:
    messages: List[BaseMessage] = state.get("messages", [])
    user_text = _latest_user_text(messages)
    if not user_text.strip():
        return {"messages": [AIMessage(content="")]}

    tool_out = retrieve_tb_docs.invoke({"query": user_text})
    sources = tool_out.get("sources", []) if isinstance(tool_out, dict) else []

    sources_block = _format_sources_block(sources)
    user_prompt = (
        TB_RAG_ANSWER_USER_TMPL.replace("{user_text}", user_text).replace("{sources_block}", sources_block)
    )

    # Add summary context (helps avoid repeating questions / keeps continuity)
    summary = (state.get("summary") or "").strip()
    summary_prefix = ""
    if summary:
        summary_prefix = (
            "Conversation memory (authoritative, concise). "
            "Use it to avoid repeating questions and keep continuity:\n"
            f"{summary}\n\n"
        )

    combined_system = f"{GLOBAL_RULES.strip()}\n\n---\n\n{TB_RAG_ANSWER_SYSTEM.strip()}".strip()

    llm = get_llm(temperature=0.2, max_tokens=700)
    result = llm.with_structured_output(TBAnswerJSON).invoke(
        [
            SystemMessage(content=combined_system),
            HumanMessage(content=summary_prefix + user_prompt),
        ]
    )

    final_text = _render_answer_with_references(
        answer_text=result.answer, sources=sources, citations_used=result.citations_used
    )
    return {"messages": [AIMessage(content=final_text)]}


# =========================================================
# DBT FULL subgraph (router -> module agent)
# Uses shared parent state keys (messages + dbt_* keys)
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
    """Calls the chosen DBT module prompt (user-facing coaching)."""
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
def branch_after_classify(state: State) -> Literal["crisis", "tb_info", "dbt"]:
    if state.get("safety_route") == "crisis":
        return "crisis"

    route = state.get("route", "faq")
    if route in ("faq", "psychoed"):
        return "tb_info"
    return "dbt"


def build_graph(*, dbt_node: Any):
    g = StateGraph(State)
    g.add_node("classify", classify_node)
    g.add_node("memory", memory_manager_node)

    g.add_node("crisis", crisis_node)
    g.add_node("tb_info", tb_info_node)
    g.add_node("dbt", dbt_node)

    g.add_edge(START, "classify")
    g.add_edge("classify", "memory")

    # Route AFTER memory management (so the next LLM call sees trimmed/summarized state)
    g.add_conditional_edges(
        "memory",
        branch_after_classify,
        {"crisis": "crisis", "tb_info": "tb_info", "dbt": "dbt"},
    )

    g.add_edge("crisis", END)
    g.add_edge("tb_info", END)
    g.add_edge("dbt", END)

    return g.compile(checkpointer=MemorySaver())


# Public compiled graphs
GRAPH_DBT_MINI = build_graph(dbt_node=dbt_mini_node)
GRAPH_DBT_FULL = build_graph(dbt_node=DBT_FULL_SUBGRAPH)

