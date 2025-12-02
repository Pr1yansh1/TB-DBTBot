# agents/dbt_agent.py
import json
from typing import Dict, List, Literal, Any

from config import bedrock_chat
from resources_loader import load_system_prompts, load_dbt_skills_kb

PROMPTS = load_system_prompts()
DBT_SKILLS_KB = load_dbt_skills_kb()

DBTMode = Literal["mindfulness", "distress", "emotion", "interpersonal"]


def _heuristic_dbt_mode(text: str) -> DBTMode | None:
    t = text.lower()
    # These keyword lists should be provided via system_prompts.json if you
    # want to move them fully out of code; here we just use logical buckets
    # based on the router keywords you already defined.
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
    """
    Backup LLM classifier for DBT mode.
    Expected JSON output: {"mode": "mindfulness" | "distress" | ...}
    """
    system_prompt = PROMPTS["dbt_mode_router_system"]
    user_template = PROMPTS["dbt_mode_router_user_template"]
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
        mode = parsed.get("mode", "mindfulness")
    except json.JSONDecodeError:
        mode = "mindfulness"

    if mode not in ("mindfulness", "distress", "emotion", "interpersonal"):
        mode = "mindfulness"
    return mode  # type: ignore[return-value]


def dbt_agent_node(state: Dict) -> Dict:
    """
    DBT agent:

    - Selects a DBT mode (mindfulness / distress / emotion / interpersonal)
      via heuristic + LLM fallback.
    - Feeds DBT skills mini-KB into a mode-specific system prompt.
    - Uses a single Bedrock call to generate the micro-coach response.
    """
    messages: List[Dict[str, str]] = state["messages"]
    user_text = messages[-1]["content"]

    mode = _heuristic_dbt_mode(user_text) or _llm_dbt_mode(user_text)

    dbt_prompts = PROMPTS["dbt_modes"][mode]
    system_prompt = dbt_prompts["system_prompt"]
    user_template = dbt_prompts["user_template"]

    # Optional: provide skill snippets for this mode
    skills_for_mode: Any = DBT_SKILLS_KB.get(mode, {})

    user_msg = user_template.format(
        user_text=user_text,
        skills_json=json.dumps(skills_for_mode, ensure_ascii=False),
    )

    reply = bedrock_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=260,
        temperature=0.2,
    )

    messages.append({"role": "assistant", "content": reply})
    return {
        **state,
        "messages": messages,
        "route": "dbt",
        "dbt_mode": mode,
    }

