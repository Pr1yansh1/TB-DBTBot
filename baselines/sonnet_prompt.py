# baselines/sonnet_promptstuff_cli.py
"""
Baseline CLI: Sonnet + ONE giant system prompt (prompt stuffing).

Prompt file location (fixed):
  baselines/giant_prompt.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# Ensure repo root is on sys.path so `import app...` works
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import AWS_REGION, MODEL_ID, AWS_PROFILE  # noqa: E402
from app.llm import bedrock_chat  # noqa: E402


BANNER = """Baseline: Sonnet + Giant System Prompt (prompt stuffing)
Type 'quit' to exit.
"""

GIANT_PROMPT_PATH = REPO_ROOT / "baselines" / "giant_prompt.txt"


def _load_giant_prompt(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Giant prompt file missing: {path}")
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    giant_prompt = _load_giant_prompt(GIANT_PROMPT_PATH)

    print(
        f"[config] region={AWS_REGION}, model={MODEL_ID}, "
        f"profile={AWS_PROFILE or 'default'}"
    )
    print(f"[giant prompt] {GIANT_PROMPT_PATH} ({len(giant_prompt)} chars)")
    print(BANNER)

    # Conversation starts with a single giant system prompt
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": giant_prompt}
    ]

    while True:
        try:
            user = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return

        if user.lower() in {"q", "quit", "exit"}:
            print("bye.")
            return

        if not user:
            continue

        messages.append({"role": "user", "content": user})

        reply = bedrock_chat(messages, max_tokens=400, temperature=0.2)

        messages.append({"role": "assistant", "content": reply})

        print("\nbot:", reply, "\n")


if __name__ == "__main__":
    main()

