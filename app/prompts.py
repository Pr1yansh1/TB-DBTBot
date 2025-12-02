# app/prompts.py
from pathlib import Path
from typing import List

from .config import PROMPTS_DIR


def _load_text(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8")


def load_prompt(name: str) -> str:
    """
    Load a single prompt file from prompts/ by stem name.
    E.g., load_prompt("global_system") -> prompts/global_system.txt
    """
    path = PROMPTS_DIR / f"{name}.txt"
    return _load_text(path)


def compose_system_prompt(parts: List[str]) -> str:
    """
    Combine several system prompt parts into a single string, separated
    by blank lines. Filters out empty strings.
    """
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return "\n\n".join(cleaned)


# -------- Global system prompt (tone, language, constraints) --------
GLOBAL_SYSTEM_PROMPT = load_prompt("global_system")

# -------- Safety classifier & crisis prompts --------
SAFETY_SYSTEM_PROMPT = load_prompt("safety_classifier")
CRISIS_SYSTEM_PROMPT = load_prompt("crisis_system")

# -------- Domain-specific system prompts --------
FAQ_SYSTEM_PROMPT = load_prompt("faq_system")
DBT_SYSTEM_PROMPT = load_prompt("dbt_system")
PSYCHO_SYSTEM_PROMPT = load_prompt("psycho_system")

# -------- Mini KBs (plain text, referenced in system prompts) --------
TB_FAQ_KB = load_prompt("tb_faq_kb")
DBT_SKILLS_KB = load_prompt("dbt_skills_kb")

