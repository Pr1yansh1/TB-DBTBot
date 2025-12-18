from __future__ import annotations

from typing import List, Optional
import asyncio

import chainlit as cl
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.tbtst_bot.graph import GRAPH
from baselines.sonnet_prompt import run_baseline
from src.tbtst_bot.db import init_db, upsert_thread_meta


BANNER = """TB-TST Helper (Chainlit UI)

Modes:
- Agentic (LangGraph + tools)
- Baseline (single-prompt Sonnet)

Everything is stored (threads, messages, feedback).
"""


@cl.on_chat_start
async def on_chat_start():
    init_db()

    await cl.Message(content=BANNER).send()

    # Ask for name once per thread
    name = await cl.AskUserMessage(content="Hi — what name should I call you?", timeout=60).send()
    user_name = (name["output"] or "").strip() if name else ""
    if not user_name:
        user_name = "friend"

    cl.user_session.set("user_name", user_name)

    thread_id = cl.context.session.id
    upsert_thread_meta(thread_id, user_name=user_name)

    actions = [
        cl.Action(name="mode_agentic", value="agentic", label="Agentic (LangGraph)"),
        cl.Action(name="mode_baseline", value="baseline", label="Baseline (Sonnet prompt)"),
    ]
    await cl.Message(content=f"Nice to meet you, {user_name}. Choose a mode:", actions=actions).send()

    cl.user_session.set("mode", None)
    cl.user_session.set("baseline_messages", [])


@cl.action_callback("mode_agentic")
async def on_mode_agentic(action: cl.Action):
    cl.user_session.set("mode", "agentic")
    thread_id = cl.context.session.id
    upsert_thread_meta(thread_id, mode="agentic")
    await cl.Message(content="✅ Mode set to: Agentic (LangGraph).").send()


@cl.action_callback("mode_baseline")
async def on_mode_baseline(action: cl.Action):
    cl.user_session.set("mode", "baseline")
    cl.user_session.set("baseline_messages", [])
    thread_id = cl.context.session.id
    upsert_thread_meta(thread_id, mode="baseline")
    await cl.Message(content="✅ Mode set to: Baseline (Sonnet prompt).").send()


@cl.on_message
async def on_message(message: cl.Message):
    mode: Optional[str] = cl.user_session.get("mode")
    if mode not in {"agentic", "baseline"}:
        await cl.Message(content="Pick a mode first (Agentic or Baseline).").send()
        return

    user_text = (message.content or "").strip()
    if not user_text:
        return

    thread_id = cl.context.session.id

    msg = cl.Message(content="…")
    await msg.send()

    try:
        if mode == "agentic":
            def _run_graph():
                return GRAPH.invoke(
                    {"messages": [HumanMessage(content=user_text)]},
                    config={"configurable": {"thread_id": thread_id}},
                )

            final = await asyncio.to_thread(_run_graph)
            reply = final["messages"][-1].content
            msg.content = reply
            await msg.update()
            return

        # baseline
        baseline_messages: List[BaseMessage] = cl.user_session.get("baseline_messages") or []
        baseline_messages = baseline_messages + [HumanMessage(content=user_text)]

        reply_text = await asyncio.to_thread(run_baseline, baseline_messages)
        baseline_messages = baseline_messages + [AIMessage(content=reply_text)]
        cl.user_session.set("baseline_messages", baseline_messages)

        msg.content = reply_text
        await msg.update()

    except Exception as e:
        msg.content = f"Sorry — something went wrong.\n\nError: {type(e).__name__}: {e}"
        await msg.update()

