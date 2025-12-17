from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage

from .graph import GRAPH, THREAD_ID

BANNER = """TB-TST Helper (LangGraph + AWS Bedrock)

Flow:
  classify (safety + route) → (crisis | FAQ | DBT) → reply
"""


def main():
    print(BANNER)

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

        final: Dict[str, Any] = GRAPH.invoke(
            {"messages": [HumanMessage(content=user)]},
            config={"configurable": {"thread_id": THREAD_ID}},
        )

        reply = final["messages"][-1].content
        print("\nbot:", reply, "\n")


if __name__ == "__main__":
    main()

