# app/state.py
from typing import TypedDict, List, Dict, Any, Literal


class ConversationState(TypedDict, total=False):
    """
    Shared state that flows through all LangGraph nodes.
    """
    messages: List[Dict[str, str]]     # [{"role": "user"|"assistant", "content": "..."}]
    safety: Dict[str, Any]             # {"level": ..., "is_crisis": ..., "reason": ..., "flags": [...]}
    route: Literal["crisis", "faq", "dbt", "psycho"]  # high-level route for this turn
    meta: Dict[str, Any]               # debug / tracing info

