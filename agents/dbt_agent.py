# agents/dbt_agent.py
import json
from typing import Dict, List, Literal, Any

from langchain_core.messages import AnyMessage

from config import bedrock_text
from resources_loader import load_system_prompts, load_dbt_skills_kb

PROMPTS = load_system_prompts()
DBT_SKILLS_KB = load_dbt_skills_kb()

DBTMode = Literal["mindfulness", "distress", "emotion", "interpersonal"]


def _last_user_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages or []):
        if getattr(m, "type", None) == "human":
            return m.content
    return ""


def _heuristic_dbt_mode(text: str) -> DBTMode | None:
    t = text.lower()
    if any(k in t for k in PROMPTS.get("dbt_distress_keywords", [])):
        return "distress"
    if any(k in t for k in PROMPTS.get("dbt_interpersonal_keywords", [])):
        return "interpersonal"
    if any(k in t for k in PROMPTS.get("dbt_emotion_keywords", [])):
        return "emotion"
    if any(k in t for k in PROMPTS.get("dbt_mindfulness_keywords", [])):
        return "mindfulness"
    return None


def _llm_dbt_mode(text: str) -> DBTMode:
    system_prompt = PROMPTS["dbt_mode_router_system"]
    user_template = PROMPTS["dbt_mode_router_user_template"]
    user_msg = user_template.format(user_text=text)

    raw = bedrock_text(
        [
            {"type": "system", "content": system_prompt},
            {"type": "human", "content": user_msg},
        ],
        max_tokens=120,
        temperature=0.0,
    )

    try:
        parsed: Dict[str, Any] = json.loads(raw)
        mode = parsed.get("mode", "mindfulness")
    except json.JSONDecodeError:
        mode = "mindfulness"

    if mode not in ("mindfulness", "distress", "emotion", "interpersonal"):
        mode = "mindfulness"
    return mode  # type: ignore[return-value]


def dbt_agent_node(state: Dict) -> Dict:
    messages: list[AnyMessage] = state.get("messages", [])
    user_text = _last_user_text(messages)

    mode: DBTMode = _heuristic_dbt_mode(user_text) or _llm_dbt_mode(user_text)

    dbt_prompts = PROMPTS["dbt_modes"][mode]
    system_prompt = dbt_prompts["system_prompt"]
    user_template = dbt_prompts["user_template"]

    skills_for_mode: Any = DBT_SKILLS_KB.get(mode, [])
    user_msg = user_template.format(
        user_text=user_text,
        skills_json=json.dumps(skills_for_mode, ensure_ascii=False),
    )

    reply = bedrock_text(
        [
            {"type": "system", "content": system_prompt},
            {"type": "human", "content": user_msg},
        ],
        max_tokens=300,
        temperature=0.2,
    )

    return {
        "messages": [{"type": "ai", "content": reply}],
        "route": "dbt",
        "dbt_mode": mode,
    }

