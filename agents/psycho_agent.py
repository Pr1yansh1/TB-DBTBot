# agents/psycho_agent.py
from typing import Dict, List

from langchain_core.messages import AnyMessage

from config import bedrock_text
from resources_loader import load_system_prompts

PROMPTS = load_system_prompts()


def _last_user_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages or []):
        if getattr(m, "type", None) == "human":
            return m.content
    return ""


def psycho_agent_node(state: Dict) -> Dict:
    messages: list[AnyMessage] = state.get("messages", [])
    user_text = _last_user_text(messages)

    system_prompt = PROMPTS["psychoed_system"]
    user_template = PROMPTS["psychoed_user_template"]
    user_msg = user_template.format(user_text=user_text)

    reply = bedrock_text(
        [
            {"type": "system", "content": system_prompt},
            {"type": "human", "content": user_msg},
        ],
        max_tokens=260,
        temperature=0.25,
    )

    return {
        "messages": [{"type": "ai", "content": reply}],
        "route": "psychoed",
    }

