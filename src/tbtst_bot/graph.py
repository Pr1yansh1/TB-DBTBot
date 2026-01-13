# graph.py
from __future__ import annotations

import json
import os
from uuid import uuid4
from typing import Any, Dict, List, Literal, Set

from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from .config import get_llm
from .prompts import load_prompt
from .rag_utils import retrieve_tb_docs


# -------------------------
# Prompts (ALL loaded; nothing hard-coded)
# -------------------------
CLASSIFY_SYSTEM = load_prompt("classify_route.txt")
CLASSIFY_USER_TMPL = load_prompt("classify_route_user.txt")
CRISIS_TEXT = load_prompt("crisis.txt")

DBT_SYSTEM = load_prompt("dbt_system.txt")

# New: citation-safe TB RAG answer prompts
# (You must add these two files to src/tbtst_bot/prompts/)
TB_RAG_ANSWER_SYSTEM = load_prompt("tb_rag_answer_system.txt")
TB_RAG_ANSWER_USER_TMPL = load_prompt("tb_rag_answer_user.txt")


# -------------------------
# State
# -------------------------
RiskLevel = Literal["none", "passive", "active_no_plan", "active_with_plan", "uncertain"]
Route = Literal["faq", "dbt", "psychoed"]


class State(MessagesState, total=False):
    safety_risk_level: RiskLevel
    safety_triggers: List[str]
    has_protective: bool
    safety_route: Literal["ok", "crisis"]
    route: Route
    route_source: str
    classifier_raw: str


# -------------------------
# Helpers
# -------------------------
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
    # This is only shown to the model; user never sees it directly.
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

    # Only allow citations that exist in the retrieved sources
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


# -------------------------
# Nodes: classify + safety
# -------------------------
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
    raw_msg = llm.invoke(
        [
            SystemMessage(content=CLASSIFY_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )
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
    return {"messages": [AIMessage(content=CRISIS_TEXT)]}


# -------------------------
# DBT specialist
# -------------------------
def dbt_node(state: State) -> Dict[str, Any]:
    messages: List[BaseMessage] = state.get("messages", [])
    llm = get_llm(temperature=0.25, max_tokens=420)
    reply = llm.invoke([SystemMessage(content=DBT_SYSTEM), *messages])
    return {"messages": [AIMessage(content=reply.content or "")]}


# =========================================================
# TB-info specialist: citation-safe RAG (Chroma retriever tool)
# - Always retrieve
# - LLM returns STRICT JSON: {"answer": "... [1] ...", "citations_used":[1]}
# - We deterministically append "Referencias:" from the retrieved sources
# =========================================================
class TBAnswerJSON(BaseModel):
    answer: str = Field(...)
    citations_used: List[int] = Field(default_factory=list)


def tb_info_node(state: State) -> Dict[str, Any]:
    messages: List[BaseMessage] = state.get("messages", [])
    user_text = _latest_user_text(messages)
    if not user_text.strip():
        return {"messages": [AIMessage(content="")]}

    # 1) Retrieve structured sources (tool returns {"sources":[{id,title,filename,excerpt}, ...]})
    tool_out = retrieve_tb_docs.invoke({"query": user_text})
    sources = tool_out.get("sources", []) if isinstance(tool_out, dict) else []

    sources_block = _format_sources_block(sources)
    user_prompt = (
        TB_RAG_ANSWER_USER_TMPL
        .replace("{user_text}", user_text)
        .replace("{sources_block}", sources_block)
    )

    # 2) Ask model for strict JSON with inline citations like [1]
    llm = get_llm(temperature=0.2, max_tokens=520)
    result = llm.with_structured_output(TBAnswerJSON).invoke(
        [
            SystemMessage(content=TB_RAG_ANSWER_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )

    # 3) Render references deterministically (no “the context says…”)
    final_text = _render_answer_with_references(
        answer_text=result.answer,
        sources=sources,
        citations_used=result.citations_used,
    )
    return {"messages": [AIMessage(content=final_text)]}


# -------------------------
# Routing
# -------------------------
def branch_after_classify(state: State) -> Literal["crisis", "tb_info", "dbt"]:
    if state.get("safety_route") == "crisis":
        return "crisis"

    route = state.get("route", "faq")
    # faq + psychoed both go to TB-info RAG specialist for now
    if route in ("faq", "psychoed"):
        return "tb_info"
    return "dbt"


# -------------------------
# Build main graph
# -------------------------
def build_graph():
    g = StateGraph(State)

    g.add_node("classify", classify_node)
    g.add_node("crisis", crisis_node)
    g.add_node("tb_info", tb_info_node)
    g.add_node("dbt", dbt_node)

    g.add_edge(START, "classify")

    g.add_conditional_edges(
        "classify",
        branch_after_classify,
        {"crisis": "crisis", "tb_info": "tb_info", "dbt": "dbt"},
    )

    g.add_edge("crisis", END)
    g.add_edge("tb_info", END)
    g.add_edge("dbt", END)

    return g.compile(checkpointer=MemorySaver())


GRAPH = build_graph()
THREAD_ID = os.getenv("THREAD_ID", f"cli-{uuid4().hex[:8]}")

