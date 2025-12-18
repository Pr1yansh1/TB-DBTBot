"""
Baseline: Sonnet + ONE giant system prompt (prompt stuffing).

- CLI mode:
    uv run python baselines/sonnet_prompt.py
- Library mode (Chainlit / other):
    from baselines.sonnet_prompt import run_baseline

Prompt file location (fixed):
  baselines/giant_prompt.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.tbtst_bot.config import bedrock_chat


REPO_ROOT = Path(__file__).resolve().parent.parent
GIANT_PROMPT_PATH = REPO_ROOT / "baselines" / "giant_prompt.txt"


BANNER = """Baseline: Sonnet + Giant System Prompt (prompt stuffing)
Type 'quit' to exit.
"""


def _load_giant_prompt() -> str:
    if not GIANT_PROMPT_PATH.exists():
        raise RuntimeError(f"Giant prompt file missing: {GIANT_PROMPT_PATH}")
    return GIANT_PROMPT_PATH.read_text(encoding="utf-8").strip()


def run_baseline(
    history: List[BaseMessage],
    *,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> str:
    """
    Run the baseline model for one turn.

    Input: prior conversation as LangChain message objects (HumanMessage/AIMessage).
    Output: assistant text.

    Implementation: prepend a single giant SystemMessage (prompt stuffing),
    then pass the full message list to `bedrock_chat`.
    """
    system = _load_giant_prompt()
    msgs: List[BaseMessage] = [SystemMessage(content=system), *history]
    return bedrock_chat(msgs, max_tokens=max_tokens, temperature=temperature)


def main() -> None:
    giant_prompt = _load_giant_prompt()
    print(f"[giant prompt] {GIANT_PROMPT_PATH} ({len(giant_prompt)} chars)")
    print(BANNER)

    history: List[BaseMessage] = []

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

        history.append(HumanMessage(content=user))
        reply = run_baseline(history, max_tokens=400, temperature=0.2)
        history.append(AIMessage(content=reply))

        print("\nbot:", reply, "\n")


if __name__ == "__main__":
    main()

