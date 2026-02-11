# chainlit_app.py
from __future__ import annotations

from typing import Optional

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage

from tbtst_bot.graph import GRAPH_DBT_FULL, GRAPH_DBT_MINI


MINI_PROFILE = "DBT Mini"
FULL_PROFILE = "DBT Full"


@cl.set_chat_profiles
async def chat_profile(current_user: Optional[cl.User] = None):
    return [
        cl.ChatProfile(
            name=MINI_PROFILE,
            markdown_description=(
                "Uses the **mini** graph: classify → (crisis | TB FAQ RAG | DBT mini). "
                "DBT is one baseline node."
            ),
        ),
        cl.ChatProfile(
            name=FULL_PROFILE,
            markdown_description=(
                "Uses the **full** graph: classify → (crisis | TB FAQ RAG | DBT full). "
                "DBT uses brain-router → (DT/MIND/ER/IE) module coaching."
            ),
        ),
    ]


def _get_selected_graph() -> tuple[str, object]:
    """Return (label, compiled_graph) based on the active chat profile."""
    profile = cl.user_session.get("chat_profile") or MINI_PROFILE
    if profile == FULL_PROFILE:
        return "full", GRAPH_DBT_FULL
    return "mini", GRAPH_DBT_MINI


@cl.on_chat_start
async def on_chat_start():
    # Chainlit docs recommend this as a stable per-chat identifier.
    thread_id = cl.context.session.id  # type: ignore[attr-defined]
    cl.user_session.set("thread_id", thread_id)

    graph_label, _ = _get_selected_graph()
    await cl.Message(
        content=(
            f"✅ TB-TST Helper started.\n"
            f"- Chat profile: **{cl.user_session.get('chat_profile')}**\n"
            f"- Graph: **{graph_label}**\n"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    graph_label, graph = _get_selected_graph()
    thread_id = cl.user_session.get("thread_id") or cl.context.session.id  # type: ignore[attr-defined]

    user_text = (message.content or "").strip()
    if not user_text:
        return

    try:
        # Your graphs use MemorySaver() checkpointer, so pass a thread_id.
        result_state = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_text)]},
            config={"configurable": {"thread_id": thread_id}},
        )

        # Extract the latest assistant message from the returned state.
        assistant_text = ""
        msgs = result_state.get("messages") if isinstance(result_state, dict) else None
        if isinstance(msgs, list):
            for m in reversed(msgs):
                if isinstance(m, AIMessage):
                    assistant_text = (m.content or "").strip()
                    break

        if not assistant_text:
            assistant_text = "(No response produced.)"

        await cl.Message(content=assistant_text).send()

    except Exception as e:
        # Keep UI clean; log the details server-side.
        print(f"[chainlit_app] ERROR ({graph_label=}, {thread_id=}): {e}")
        await cl.Message(
            content=(
                "⚠️ Something went wrong while running the graph.\n"
                "Check the server logs for the full error."
            )
        ).send()

