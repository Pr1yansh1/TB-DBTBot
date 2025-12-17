# supervisor.py
from __future__ import annotations

from typing import Dict, Literal, Any

from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.types import Command

from config import bedrock_llm
from resources_loader import load_system_prompts, load_routing_keywords

PROMPTS = load_system_prompts()
KW = load_routing_keywords()

DomainRoute = Literal["faq", "dbt", "psychoed"]


def _last_user_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages or []):
        if getattr(m, "type", None) == "human":
            return m.content
    return ""


def _heuristic_route(text: str) -> DomainRoute | None:
    t = text.lower()

    def count_hits(words: list[str]) -> int:
        return sum(1 for kw in words if kw in t)

    faq_hits = count_hits(KW.get("faq", []))
    dbt_hits = count_hits(KW.get("dbt", []))
    psycho_hits = count_hits(KW.get("psychoed", []))

    scores = {"faq": faq_hits, "dbt": dbt_hits, "psychoed": psycho_hits}
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    if best_score == 0 or list(scores.values()).count(best_score) > 1:
        return None
    return best_label  # type: ignore[return-value]


@tool
def route_to_faq() -> str:
    """Route to the TB FAQ agent when the user asks logistics/meds/appointments/side-effects etc."""
    return "faq"


@tool
def route_to_dbt() -> str:
    """Route to the DBT coping-skills agent when the user expresses distress, strong emotions, overwhelm, or relationship conflict."""
    return "dbt"


@tool
def route_to_psychoed() -> str:
    """Route to psychoeducation when the user asks explanatory questions (what is TB, how spread, why pills, etc.)."""
    return "psychoed"


TOOLS = [route_to_faq, route_to_dbt, route_to_psychoed]


def supervisor_router_node(state: Dict) -> Command[Literal["faq_agent", "dbt_agent", "psycho_agent"]]:
    """
    Supervisor router (tool-calling):
    - Prefer heuristic for obvious cases (cheap, predictable).
    - Otherwise, ask Sonnet to call ONE routing tool.
    - Return Command(goto=...) for the chosen agent.
    """
    messages: list[AnyMessage] = state.get("messages", [])
    user_text = _last_user_text(messages)

    # 1) cheap heuristic first
    heuristic = _heuristic_route(user_text)
    if heuristic:
        route = heuristic
        source = "heuristic"
    else:
        # 2) tool-calling decision
        system_prompt = PROMPTS["domain_router_system"]
        user_template = PROMPTS["domain_router_user_template"]
        user_msg = user_template.format(user_text=user_text)

        llm = bedrock_llm.bind_tools(TOOLS)

        resp = llm.invoke(
            [
                {"type": "system", "content": system_prompt},
                {"type": "human", "content": user_msg},
            ],
            model_kwargs={"max_tokens": 120, "temperature": 0.0},
        )

        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            # safest default
            route = "psychoed"
            source = "supervisor_no_toolcall"
        else:
            name = tool_calls[0].get("name")
            if name == "route_to_faq":
                route = "faq"
            elif name == "route_to_dbt":
                route = "dbt"
            else:
                route = "psychoed"
            source = "supervisor_toolcall"

    if route == "faq":
        return Command(update={"route": "faq", "route_source": source}, goto="faq_agent")
    if route == "dbt":
        return Command(update={"route": "dbt", "route_source": source}, goto="dbt_agent")
    return Command(update={"route": "psychoed", "route_source": source}, goto="psycho_agent")

