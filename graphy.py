# graph.py
import os
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import AnyMessage, SystemMessage

from safety import safety_gate_node
from supervisor import supervisor_router_node
from agents.faq_agent import faq_agent_node
from agents.dbt_agent import dbt_agent_node
from agents.psycho_agent import psycho_agent_node
from resources_loader import load_system_prompts

PROMPTS = load_system_prompts()


class State(TypedDict, total=False):
    # ✅ message channel w/ reducer for long-turn memory
    messages: Annotated[list[AnyMessage], add_messages]

    # safety
    risk_level: str
    risk_triggers: list[str]
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


def ingress_node(state: State) -> dict:
    """Inject global system prompt exactly once at conversation start."""
    messages = state.get("messages", [])
    global_prompt = PROMPTS.get("global_system_prompt")
    if global_prompt:
        if not messages or not isinstance(messages[0], SystemMessage):
            return {"messages": [SystemMessage(content=global_prompt)]}
    return {}


def crisis_node(state: State) -> dict:
    """
    Deterministic crisis message (no LLM).
    Delegates safety protocol content to the team-owned template.
    """
    template = PROMPTS["crisis_message_template"]
    msg_text = template.format(risk_level=state.get("risk_level", "uncertain"))
    return {
        "messages": [{"type": "ai", "content": msg_text}],
        "route": "crisis",
    }


def safety_branch(state: State) -> Literal["crisis", "supervisor"]:
    return "crisis" if state.get("safety_route") == "crisis" else "supervisor"


def build_graph():
    builder = StateGraph(State)

    builder.add_node("ingress", ingress_node)
    builder.add_node("safety_gate", safety_gate_node)
    builder.add_node("crisis", crisis_node)

    # ✅ supervisor tool-calling router (replaces routing.py as primary router)
    builder.add_node("supervisor", supervisor_router_node)

    builder.add_node("faq_agent", faq_agent_node)
    builder.add_node("dbt_agent", dbt_agent_node)
    builder.add_node("psycho_agent", psycho_agent_node)

    builder.add_edge(START, "ingress")
    builder.add_edge("ingress", "safety_gate")

    builder.add_conditional_edges(
        "safety_gate",
        safety_branch,
        {"crisis": "crisis", "supervisor": "supervisor"},
    )

    # supervisor returns Command(goto=...) so we just connect it to END as a fallback
    # (it will navigate to the right agent via Command)
    builder.add_edge("faq_agent", END)
    builder.add_edge("dbt_agent", END)
    builder.add_edge("psycho_agent", END)
    builder.add_edge("crisis", END)

    return builder.compile(checkpointer=MemorySaver())


GRAPH = build_graph()
THREAD_ID = os.getenv("THREAD_ID", f"cli-{uuid4().hex[:8]}")

