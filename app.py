# app.py
from typing import Dict

from graph import GRAPH, THREAD_ID, State

BANNER = """TB-TST Helper (LangGraph + AWS Bedrock)

Flow:
  ingress → safety_gate → (crisis | router) → (FAQ / DBT / Psycho-ed) → reply

Safety and routing are rule + LLM; FAQ / DBT / Psycho-ed are LLM agents.
Type 'quit' to exit.
"""


def main():
    print(BANNER)
    state: State = {"messages": []}

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

        state["messages"].append({"role": "user", "content": user})

        final: Dict = GRAPH.invoke(
            state,
            config={"configurable": {"thread_id": THREAD_ID}},
        )
        reply = final["messages"][-1]["content"]
        print("\nbot:", reply, "\n")

        # carry state forward so short-term context is preserved
        state = {"messages": final["messages"]}


if __name__ == "__main__":
    main()

