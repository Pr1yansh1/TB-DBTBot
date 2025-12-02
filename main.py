# main.py
import os
from typing import Dict, Any

from app.config import AWS_REGION, MODEL_ID, AWS_PROFILE, DEFAULT_THREAD_ID
from app.graph import build_graph
from app.state import ConversationState

graph = build_graph()

BANNER = """TB DBT Helper Bot (LangGraph + AWS Bedrock)
Paths: crisis | faq | dbt | psycho
Type 'quit' to exit.
"""


def main() -> None:
    thread_id = os.getenv("THREAD_ID", DEFAULT_THREAD_ID)
    print(
        f"[config] region={AWS_REGION}, model={MODEL_ID}, "
        f"profile={AWS_PROFILE or 'default'}, thread_id={thread_id}"
    )
    print(BANNER)

    state: ConversationState = {"messages": []}

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

        # Append user message
        msgs = state.get("messages") or []
        msgs.append({"role": "user", "content": user})
        state["messages"] = msgs

        # Invoke graph with persistent thread_id
        final: Dict[str, Any] = graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # Update local state for next turn
        state = final  # type: ignore[assignment]

        # Debug route
        route = final.get("route", "unknown")
        print(f"[route: {route}]")

        # Print last assistant message
        messages = final.get("messages") or []
        last_assistant = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant = msg["content"]
                break

        if last_assistant:
            print("\nbot:", last_assistant, "\n")
        else:
            print("\nbot: (no assistant reply)\n")


if __name__ == "__main__":
    main()

