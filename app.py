# app.py
from typing import Dict,cast,List
from langchain_core.messages import HumanMessage, BaseMessage

import chainlit as cl

from graph import GRAPH, THREAD_ID

BANNER = """TB-TST Helper (LangGraph + AWS Bedrock)

Flow:
  ingress → safety_gate → (crisis | router) → (FAQ / DBT / Psycho-ed) → reply

Safety and routing are rule + LLM; FAQ / DBT / Psycho-ed are LLM agents.
Type 'quit' to exit (CLI mode).
"""


# ----------------- Chainlit UI -----------------
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("thread_id", cl.context.session.id)
    await cl.Message(content=BANNER).send()

@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id") or cl.context.session.id
    config = {"configurable": {"thread_id": thread_id}}

    final: Dict = GRAPH.invoke(
        {"messages": [{"role": "user", "content": message.content}]},
        config=config,
    )

    reply = final["messages"][-1]["content"]
    await cl.Message(content=reply).send()
# ----------------- CLI REPL (existing) -----------------

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

        final: Dict = GRAPH.invoke(
            {"messages": [{"role": "user", "content": user}]},
            config={"configurable": {"thread_id": THREAD_ID}},
        )
        reply = final["messages"][-1]["content"]
        print("\nbot:", reply, "\n")

if __name__ == "__main__":
    main()

