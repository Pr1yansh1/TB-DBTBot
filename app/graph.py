# app/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import ConversationState
from .nodes import (
    ingress_node,
    safety_node,
    safety_route,
    domain_router,
    crisis_node,
    faq_node,
    dbt_node,
    psycho_node,
)


def build_graph():
    """
    Build and compile the LangGraph for the TB DBT bot.

    Flow:
        START → ingress → safety
        safety --crisis--> crisis → END
        safety --non_crisis--> router → (faq|dbt|psycho) → END
    """
    g = StateGraph(ConversationState)

    # Core nodes
    g.add_node("ingress", ingress_node)
    g.add_node("safety", safety_node)

    # Router node for non-crisis domain routing (acts as a small orchestrator)
    # It doesn't modify state except setting route; we call domain_router via conditional edges.
    g.add_node("router", lambda s: s)

    # Leaf bots
    g.add_node("crisis", crisis_node)
    g.add_node("faq", faq_node)
    g.add_node("dbt", dbt_node)
    g.add_node("psycho", psycho_node)

    # Top-level flow
    g.add_edge(START, "ingress")
    g.add_edge("ingress", "safety")

    # Safety routing
    g.add_conditional_edges(
        "safety",
        safety_route,
        {
            "crisis": "crisis",
            "non_crisis": "router",
        },
    )

    # Domain routing from router
    g.add_conditional_edges(
        "router",
        domain_router,
        {
            "faq": "faq",
            "dbt": "dbt",
            "psycho": "psycho",
        },
    )

    # All leaf bots terminate
    for leaf in ("crisis", "faq", "dbt", "psycho"):
        g.add_edge(leaf, END)

    # Compile with in-memory checkpointing for persistence
    memory = MemorySaver()
    return g.compile(checkpointer=memory)

