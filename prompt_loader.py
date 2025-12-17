# prompt_loader.py
from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(name: str) -> str:
    """
    Loads a prompt from prompts/<name>.txt
    Example: load_prompt("supervisor_system") reads prompts/supervisor_system.txt
    """
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8").strip() + "\n"

