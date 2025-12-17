# routing.py
import json
from typing import Any, Dict, Literal

from langchain_core.messages import BaseMessage, HumanMessage

from config import bedrock_chat
from resources_loader import load_routing_keywords, load_system_prompts

DomainRoute = Literal["faq", "dbt", "psychoed"]

KW = load_routing_keywords()
PROMPTS = load_system_prompts()


def _latest_user_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _heuristic_route(text: str) -> DomainRoute | None:
    t = (text or "").lower()

    def count_hits(words: list[str]) -> int:
        return sum(1 for kw in words if kw in t)

    faq_hits = count_hits(KW.get("faq", []))
    dbt_hits = count_hits(KW.get("dbt", []))
    psycho_hits = count_hits(KW.get("psychoed", []))

    scores = {"faq": faq_hits, "dbt": dbt_hits, "psychoed": psycho_hits}
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    # If everything is 0 or ties, we treat as ambiguous
    if best_score == 0 or list(scores.values()).count(best_score) > 1:
        return None
    return best_label  # type: ignore[return-value]


def _llm_route(text: str) -> DomainRoute:
    """
    Backup LLM router used for ambiguous inputs.
    Expected LLM output: JSON with `route` ∈ {"faq","dbt","psychoed"}.
    """
    system_prompt = PROMPTS["domain_router_system"]
    user_template = PROMPTS["domain_router_user_template"]
    user_msg = user_template.format(user_text=text)

    raw = bedrock_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=120,
        temperature=0.0,
    )

    try:
        parsed: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        return "psychoed"  # safest default

    route = parsed.get("route", "psychoed")
    if route not in ("faq", "dbt", "psychoed"):
        route = "psychoed"
    return route  # type: ignore[return-value]


def domain_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Domain router node.
    - First attempt: keyword-based routing (cheap).
    - On tie/low-signal: LLM-based router.
    """
    messages: list[BaseMessage] = state.get("messages", [])
    text = _latest_user_text(messages)

    if not text.strip():
        return {
            **state,
            "route": "psychoed",
            "route_source": "empty_input_default",
        }

    heuristic = _heuristic_route(text)
    if heuristic is not None:
        route: DomainRoute = heuristic
        source = "heuristic"
    else:
        route = _llm_route(text)
        source = "llm"

    return {
        **state,
        "route": route,
        "route_source": source,
    }

