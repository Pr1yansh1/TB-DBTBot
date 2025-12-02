# app/state.py
from typing import TypedDict, List, Dict, Any, Literal


class ConversationState(TypedDict, total=False):
    """
    Shared state that flows through all LangGraph nodes.

    Fields:
      - messages: chat history as Anthropic-style turns
      - safety: rich safety metadata (see safety_node)
      - route: high-level domain route for this turn
      - meta: debug / tracing info
    """
    messages: List[Dict[str, str]]     # [{"role": "user"|"assistant", "content": "..."}]
    safety: Dict[str, Any]             # {"risk_level": ..., "is_crisis": ..., "keyword_assessment": {...}, ...}
    route: Literal["crisis", "faq", "dbt", "psycho"]  # high-level route for this turn
    meta: Dict[str, Any]               # debug / tracing info

