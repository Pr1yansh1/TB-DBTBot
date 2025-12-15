# agents/psycho_agent.py
from typing import Dict, List

from config import bedrock_chat
from resources_loader import load_system_prompts

PROMPTS = load_system_prompts()


def psycho_agent_node(state: Dict) -> Dict:
    """
    Misc / psychoeducation agent.

    IMPORTANT:
    - Do not mutate state["messages"].
    - Return only the delta assistant message.
    """
    messages: List[Dict[str, str]] = state.get("messages", [])

    user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_text = m.get("content", "")
            break

    system_prompt = PROMPTS["psychoed_system"]
    user_template = PROMPTS["psychoed_user_template"]
    user_msg = user_template.format(user_text=user_text)

    reply = bedrock_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=260,
        temperature=0.25,
    )

    return {
        **state,
        "messages": [{"role": "assistant", "content": reply}],
        "route": "psychoed",
    }

