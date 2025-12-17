# prompts.py
from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    # Always end with newline to reduce formatting edge cases
    return path.read_text(encoding="utf-8").strip() + "\n"

