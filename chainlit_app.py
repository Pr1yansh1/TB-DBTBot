from __future__ import annotations

from typing import List, Optional

import chainlit as cl
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# Your agentic workflow
from src.tbtst_bot.graph import GRAPH

# Your baseline wrapper (we'll define it below)
from baselines.sonnet_prompt import run_baseline


BANNER = """TB-TST Helper (Chainlit UI)

Modes:
- Agentic (LangGraph + tools)
- Baseline (single-prompt Sonnet)

Notes:
- With data persistence enabled, Chainlit will store threads and show 👍/👎 feedback automatically.
"""


def _last_ai_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m.content
    return ""


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content=BANNER).send()

    actions = [
        cl.Action(name="mode_agentic", value="agentic", label="Agentic (LangGraph)"),
        cl.Action(name="mode_baseline", value="baseline", label="Baseline (Sonnet prompt)"),
    ]
    await cl.Message(content="Choose a mode:", actions=actions).send()

    # default state
    cl.user_session.set("mode", None)
    cl.user_session.set("baseline_messages", [])  # type: ignore[arg-type]


@cl.action_callback("mode_agentic")
async def on_mode_agentic(action: cl.Action):
    cl.user_session.set("mode", "agentic")
    await cl.Message(content="✅ Mode set to: Agentic (LangGraph).").send()


@cl.action_callback("mode_baseline")
async def on_mode_baseline(action: cl.Action):
    cl.user_session.set("mode", "baseline")
    cl.user_session.set("baseline_messages", [])
    await cl.Message(content="✅ Mode set to: Baseline (Sonnet prompt).").send()


@cl.on_message
async def on_message(message: cl.Message):
    mode: Optional[str] = cl.user_session.get("mode")
    if mode not in {"agentic", "baseline"}:
        await cl.Message(content="Pick a mode first (Agentic or Baseline).").send()
        return

    user_text = message.content.strip()
    if not user_text:
        return

    # Use Chainlit session id as LangGraph thread id so memory is per chat session
    thread_id = cl.context.session.id

    if mode == "agentic":
        final = GRAPH.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        # Your graph returns LangChain message objects (MessagesState)
        reply = final["messages"][-1].content
        await cl.Message(content=reply).send()
        return

    # baseline mode
    baseline_messages: List[BaseMessage] = cl.user_session.get("baseline_messages") or []
    baseline_messages = baseline_messages + [HumanMessage(content=user_text)]

    reply_text = run_baseline(baseline_messages)
    baseline_messages = baseline_messages + [AIMessage(content=reply_text)]
    cl.user_session.set("baseline_messages", baseline_messages)

    await cl.Message(content=reply_text).send()

