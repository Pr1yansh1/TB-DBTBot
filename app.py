# app.py
from typing import Dict, cast

import chainlit as cl

from graph import GRAPH, THREAD_ID, State

BANNER = """TB-TST Helper (LangGraph + AWS Bedrock)

Flow:
  ingress → safety_gate → (crisis | router) → (FAQ / DBT / Psycho-ed) → reply

Safety and routing are rule + LLM; FAQ / DBT / Psycho-ed are LLM agents.
Type 'quit' to exit (CLI mode).
"""


# ----------------- Chainlit UI -----------------


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize a fresh conversation state for this Chainlit session.
    We keep State in Chainlit's user_session so GRAPH.invoke always
    sees the accumulated message history, just like the CLI REPL.
    """
    initial_state: State = {"messages": []}
    cl.user_session.set("state", initial_state)

    # Send banner / intro to the UI
    await cl.Message(content=BANNER).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle a user message from the Chainlit UI.

    - Pull State from user_session
    - Append the new user message
    - Invoke the LangGraph graph with a per-session thread_id
    - Send back the last assistant message
    - Store updated State back into user_session
    """
    # Recover previous state (or start fresh if missing)
    state = cast(State, cl.user_session.get("state")) or {"messages": []}

    # Append user message
    state["messages"].append({"role": "user", "content": message.content})

    # Use Chainlit session id as thread_id so MemorySaver can
    # keep per-user traces separate.
    config = {"configurable": {"thread_id": cl.context.session.id}}

    final: Dict = GRAPH.invoke(state, config=config)
    reply = final["messages"][-1]["content"]

    # Send reply to UI
    await cl.Message(content=reply).send()

    # Carry state forward so short-term context is preserved
    cl.user_session.set("state", {"messages": final["messages"]})


# ----------------- CLI REPL (existing) -----------------


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
            # In CLI mode we keep using the static THREAD_ID
            config={"configurable": {"thread_id": THREAD_ID}},
        )
        reply = final["messages"][-1]["content"]
        print("\nbot:", reply, "\n")

        # carry state forward so short-term context is preserved
        state = {"messages": final["messages"]}


if __name__ == "__main__":
    main()

