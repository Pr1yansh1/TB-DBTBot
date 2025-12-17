from __future__ import annotations

import json
import os
from uuid import uuid4
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from .config import bedrock_chat
from .prompts import load_prompt


# -------------------------
# Prompts
# -------------------------
CLASSIFY_SYSTEM = load_prompt("classify_route.txt")
CLASSIFY_USER_TMPL = load_prompt("classify_route_user.txt")
CRISIS_TEXT = load_prompt("crisis.txt")

FAQ_SYSTEM = load_prompt("faq_system.txt")
DBT_SYSTEM = load_prompt("dbt_system.txt")


# -------------------------
# State
# -------------------------
RiskLevel = Literal["none", "passive", "active_no_plan", "active_with_plan", "uncertain"]
Route = Literal["faq", "dbt"]

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
    # Avoid Python .format() because template contains JSON braces
    return template.replace("{user_text}", user_text)


def _parse_classifier_json(raw: str) -> Dict[str, Any]:
    """
    Expected keys (from classify prompt):
      {"safety_risk_level": "...", "safety_triggers": [...], "has_protective": true/false, "route": "faq|dbt|psychoed"}
    We map psychoed -> faq (since only faq+dbt specialists exist right now).
    """
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
    if route == "psychoed":
        route = "faq"

    return {
        "safety_risk_level": risk,
        "safety_triggers": triggers,
        "has_protective": has_protective,
        "route": route,
        "raw": raw,
    }


# -------------------------
# Nodes
# -------------------------
def classify_node(state: State) -> Dict[str, Any]:
    """
    Single Bedrock call: safety + routing.
    Returns metadata only (no messages added).
    """
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

    raw = bedrock_chat(
        [
            SystemMessage(content=CLASSIFY_SYSTEM),
            HumanMessage(content=user_prompt),
        ],
        max_tokens=220,
        temperature=0.0,
    )

    parsed = _parse_classifier_json(raw)
    risk: RiskLevel = parsed["safety_risk_level"]
    route: Route = parsed["route"]

    safety_route: Literal["ok", "crisis"]
    safety_route = "crisis" if risk in ("passive", "active_no_plan", "active_with_plan", "uncertain") else "ok"

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
    # Static crisis message; no LLM call.
    return {"messages": [AIMessage(content=CRISIS_TEXT)]}


def faq_node(state: State) -> Dict[str, Any]:
    """
    FAQ specialist.
    Uses full conversation history (messages state) but injects system prompt only for this call.
    """
    messages: List[BaseMessage] = state.get("messages", [])
    # Keep only human+ai from history, prepend system
    llm_msgs: List[BaseMessage] = [SystemMessage(content=FAQ_SYSTEM), *messages]

    reply = bedrock_chat(
        llm_msgs,
        max_tokens=320,
        temperature=0.2,
    )
    return {"messages": [AIMessage(content=reply)]}


def dbt_node(state: State) -> Dict[str, Any]:
    """
    DBT specialist.
    Uses full conversation history (messages state) but injects system prompt only for this call.
    """
    messages: List[BaseMessage] = state.get("messages", [])
    llm_msgs: List[BaseMessage] = [SystemMessage(content=DBT_SYSTEM), *messages]

    reply = bedrock_chat(
        llm_msgs,
        max_tokens=420,
        temperature=0.25,
    )
    return {"messages": [AIMessage(content=reply)]}


def branch_after_classify(state: State) -> Literal["crisis", "faq", "dbt"]:
    if state.get("safety_route") == "crisis":
        return "crisis"
    return state.get("route", "faq")


# -------------------------
# Build graph
# -------------------------
def build_graph():
    g = StateGraph(State)
    g.add_node("classify", classify_node)
    g.add_node("crisis", crisis_node)
    g.add_node("faq", faq_node)
    g.add_node("dbt", dbt_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        branch_after_classify,
        {"crisis": "crisis", "faq": "faq", "dbt": "dbt"},
    )

    g.add_edge("crisis", END)
    g.add_edge("faq", END)
    g.add_edge("dbt", END)

    return g.compile(checkpointer=MemorySaver())


GRAPH = build_graph()
THREAD_ID = os.getenv("THREAD_ID", f"cli-{uuid4().hex[:8]}")

