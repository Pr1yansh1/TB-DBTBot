# baselines/sonnet_plain_cli.py
"""
Baseline CLI: plain Sonnet chat (NO system prompt).
Uses your existing Bedrock wrapper: app.llm.bedrock_chat
and config: app.config (AWS_PROFILE/AWS_REGION/MODEL_ID).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure repo root is on sys.path so `import app...` works
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import AWS_REGION, MODEL_ID, AWS_PROFILE  # noqa: E402
from app.llm import bedrock_chat  # noqa: E402


BANNER = """Baseline: Plain Sonnet (no system prompt)
Type 'quit' to exit.
"""


def main() -> None:
    print(
        f"[config] region={AWS_REGION}, model={MODEL_ID}, "
        f"profile={AWS_PROFILE or 'default'}"
    )
    print(BANNER)

    messages: List[Dict[str, str]] = []

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

        # Plain call: no system message
        reply = bedrock_chat(messages, max_tokens=400, temperature=0.2)

        messages.append({"role": "assistant", "content": reply})

        print("\nbot:", reply, "\n")


if __name__ == "__main__":
    main()

