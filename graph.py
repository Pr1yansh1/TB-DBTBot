# graph.py
import os
from typing import Dict, List, TypedDict, Literal
from uuid import uuid4

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from safety import safety_gate_node
from routing import domain_router_node
from agents.faq_agent import faq_agent_node
from agents.dbt_agent import dbt_agent_node
from agents.psycho_agent import psycho_agent_node
from resources_loader import load_system_prompts

PROMPTS = load_system_prompts()


class State(TypedDict, total=False):
    # conversation
    messages: List[Dict[str, str]]

    # safety
    risk_level: str
    risk_triggers: List[str]
    has_protective: bool
    safety_route: Literal["ok", "crisis"]
    safety_llm_raw: str

    # routing
    route: Literal["faq", "dbt", "psychoed", "crisis"]
    route_source: str

    # metadata
    kb_id: str
    kb_question: str
    dbt_mode: str


def ingress_node(state: State) -> State:
    """Placeholder for any preprocessing / normalization."""
    return state


def crisis_node(state: State) -> State:
    """
    Crisis node: deterministic crisis message from system_prompts.json.
    No LLM call here to avoid hallucinated safety advice.
    """
    messages = state["messages"]
    template = PROMPTS["crisis_message_template"]
    # We allow templates to interpolate risk level if desired
    msg_text = template.format(risk_level=state.get("risk_level", "uncertain"))
    messages.append({"role": "assistant", "content": msg_text})
    state["messages"] = messages
    state["route"] = "crisis"
    return state


def safety_branch(state: State) -> Literal["crisis", "router"]:
    return "crisis" if state.get("safety_route") == "crisis" else "router"


def domain_branch(state: State) -> Literal["faq_agent", "dbt_agent", "psycho_agent"]:
    route = state.get("route", "psychoed")
    if route == "faq":
        return "faq_agent"
    if route == "dbt":
        return "dbt_agent"
    return "psycho_agent"


def build_graph():
    builder = StateGraph(State)

    # Nodes
    builder.add_node("ingress", ingress_node)
    builder.add_node("safety_gate", safety_gate_node)
    builder.add_node("crisis", crisis_node)
    builder.add_node("router", domain_router_node)
    builder.add_node("faq_agent", faq_agent_node)
    builder.add_node("dbt_agent", dbt_agent_node)
    builder.add_node("psycho_agent", psycho_agent_node)

    # Edges: START -> ingress -> safety
    builder.add_edge(START, "ingress")
    builder.add_edge("ingress", "safety_gate")

    # Safety branch
    builder.add_conditional_edges(
        "safety_gate",
        safety_branch,
        {
            "crisis": "crisis",
            "router": "router",
        },
    )

    # Domain router -> specific agents
    builder.add_conditional_edges(
        "router",
        domain_branch,
        {
            "faq_agent": "faq_agent",
            "dbt_agent": "dbt_agent",
            "psycho_agent": "psycho_agent",
        },
    )

    # Exit edges
    builder.add_edge("faq_agent", END)
    builder.add_edge("dbt_agent", END)
    builder.add_edge("psycho_agent", END)
    builder.add_edge("crisis", END)

    return builder.compile(checkpointer=MemorySaver())


GRAPH = build_graph()
THREAD_ID = os.getenv("THREAD_ID", f"cli-{uuid4().hex[:8]}")

