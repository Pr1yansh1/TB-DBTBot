# resources_loader.py
import json
import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List


RESOURCES_DIR = Path(
    os.getenv("RESOURCES_DIR", Path(__file__).parent / "resources")
)


def _load_json(filename: str) -> Any:
    path = RESOURCES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected resource file not found: {path}. "
            "Create it under `resources/` (or set RESOURCES_DIR)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_txt_lines(filename: str) -> List[str]:
    path = RESOURCES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected resource file not found: {path}. "
            "Create it under `resources/` (or set RESOURCES_DIR)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@lru_cache
def load_safety_keywords() -> Dict[str, List[str]]:
    """
    Expects JSON mapping string → list[string], e.g.:

    {
      "crisis_hard": [...],
      "passive_ideation": [...],
      "self_harm_non_suicidal": [...],
      "intent_markers": [...],
      "protective_factors": [...]
    }
    """
    return _load_json("safety_keywords.json")


@lru_cache
def load_routing_keywords() -> Dict[str, List[str]]:
    """
    Expects JSON like:

    {
      "faq": [...],       # TB / medication / logistics keywords
      "dbt": [...],       # emotion / skills / urges / conflict
      "psychoed": [...]   # "what is", "why", "explain", etc.
    }
    """
    return _load_json("routing_keywords.json")


@lru_cache
def load_system_prompts() -> Dict[str, Any]:
    """
    Expects JSON containing system prompts and templates, e.g. keys:

    - "safety_classifier_system"
    - "safety_classifier_user_template"
    - "domain_router_system"
    - "domain_router_user_template"
    - "faq_system"
    - "psychoed_system"
    - "dbt_modes" (mapping of DBT modes to system prompts)

    Exact wording is defined externally.
    """
    return _load_json("system_prompts.json")


@lru_cache
def load_tb_faq_kb() -> List[Dict[str, Any]]:
    """
    Expects a list of FAQ entries, each with at least:

    {
      "id": "string",
      "question": "string",
      "answer": "string",
      "keywords": ["..."]
    }
    """
    return _load_json("tb_faq_kb.json")


@lru_cache
def load_dbt_skills_kb() -> Dict[str, Any]:
    """
    Optional: DBT skills mini-KB, e.g. a mapping from skill name to
    description / step list. Exact schema is up to you.
    """
    return _load_json("dbt_skills_kb.json")

