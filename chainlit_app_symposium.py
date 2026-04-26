"""
Symposium-facing Chainlit app for the UW TB Symposium (April 29, 2026).

Identical to chainlit_app.py except:
  - No chat profile / persona selector
  - English welcome message with overview, example questions, and a warning
  - Uses a fixed generic onboarding profile (no persona context)

Run with:
    uv run chainlit run chainlit_app_symposium.py --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import logging
from uuid import uuid4

import chainlit as cl

# Re-use all shared machinery from the main app
from chainlit_app import (
    _build_async_conninfo,
    _get_session_lock,
    _GRAPH_INVOKE_SEMAPHORE,
    _thread_already_initialized,
    build_onboarding_profile,
    invoke_graph_with_retry,
    is_retryable_graph_error,
    log_session_event,
    seed_onboarding_profile,
    select_graph,
    warmup_once,
)
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict

logger = logging.getLogger("tbtst.chainlit_symposium")

# -------------------------
# Symposium welcome copy
# -------------------------

SYMPOSIUM_WELCOME = """\
## Welcome to the TB Treatment Support Bot

This chatbot was developed at the University of Washington to support people navigating TB treatment. \
It can help with questions about medications and side effects, and it also offers emotional support \
for the day-to-day challenges that come with a long treatment journey.

You can ask in **English or Spanish** — the bot will respond in **Spanish**.

**Some examples of what you can ask:**
- *"My urine turned orange after starting my medication — is that normal?"*
- *"I've been feeling really isolated at work since my diagnosis. I don't know how to talk about it."*
- *"I'm exhausted all the time and it's hard to keep taking my pills. What can I do?"*
- *"What are the signs that my liver might be reacting badly to the treatment?"*

> **Please note:** This is a research prototype. It is not a substitute for advice from your doctor or TB care team. \
For any urgent medical concerns, please contact your clinic directly.

---
Questions, feedback, or just want to chat about the build? Reach out: **pgarg2@uw.edu**
"""

SYMPOSIUM_OPENER = "What questions do you have about TB treatment today?"

# Fixed generic persona used when no profile is selected
SYMPOSIUM_PERSONA_TITLE = "TB Symposium Guest"
SYMPOSIUM_PERSONA_MD = (
    "The user is attending a TB symposium and may be a healthcare professional, "
    "researcher, or community member with general interest in TB treatment and support. "
    "Respond in the user's language. Be clear, concise, and avoid unnecessary jargon."
)


# -------------------------
# Data layer
# -------------------------

@cl.data_layer
def get_data_layer() -> SQLAlchemyDataLayer:
    return SQLAlchemyDataLayer(conninfo=_build_async_conninfo(), ssl_require=False)


@cl.header_auth_callback
def header_auth_callback(headers: dict) -> cl.User:
    return cl.User(identifier="symposium-guest", metadata={"role": "guest"})


# -------------------------
# Chat lifecycle
# -------------------------

@cl.on_chat_start
async def on_chat_start():
    graph, variant = select_graph()

    session = getattr(cl.context, "session", None)
    base_thread = (
        getattr(session, "thread_id", None)
        or getattr(session, "id", None)
        or f"cl-{uuid4().hex[:10]}"
    ) if session else f"cl-{uuid4().hex[:10]}"
    thread_id = f"{base_thread}:{variant}"

    cl.user_session.set("graph", graph)
    cl.user_session.set("variant", variant)
    cl.user_session.set("thread_id", thread_id)

    await warmup_once()

    async with _get_session_lock(thread_id):
        if _thread_already_initialized(graph, thread_id):
            logger.info(
                "on_chat_start skipped re-init for existing thread thread_id=%s", thread_id
            )
            return

        onboarding_profile = build_onboarding_profile(
            SYMPOSIUM_PERSONA_TITLE, SYMPOSIUM_PERSONA_MD
        )
        cl.user_session.set("onboarding_profile", onboarding_profile)

        await seed_onboarding_profile(graph, thread_id, onboarding_profile)

        log_session_event(thread_id, {
            "event": "session_start",
            "variant": variant,
            "persona": SYMPOSIUM_PERSONA_TITLE,
            "onboarding_profile": onboarding_profile,
        })
        log_session_event(thread_id, {
            "event": "message",
            "role": "assistant",
            "content": SYMPOSIUM_WELCOME,
        })
        log_session_event(thread_id, {
            "event": "message",
            "role": "assistant",
            "content": SYMPOSIUM_OPENER,
        })

        await cl.Message(content=SYMPOSIUM_WELCOME).send()
        await cl.Message(content=SYMPOSIUM_OPENER).send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    graph, variant = select_graph()
    thread_id = f"{thread['id']}:{variant}"

    cl.user_session.set("graph", graph)
    cl.user_session.set("variant", variant)
    cl.user_session.set("thread_id", thread_id)

    log_session_event(thread_id, {
        "event": "session_resume",
        "variant": variant,
        "thread_chainlit_id": thread["id"],
    })


@cl.on_message
async def on_message(msg: cl.Message):
    graph = cl.user_session.get("graph")
    thread_id = cl.user_session.get("thread_id")
    variant = cl.user_session.get("variant") or ""

    if graph is None or thread_id is None:
        await cl.Message(content="Error: session not started. Please reload the page.").send()
        return

    log_session_event(thread_id, {
        "event": "message",
        "role": "user",
        "variant": variant,
        "content": msg.content,
    })

    thinking = cl.Message(content="")
    await thinking.send()

    session_lock = _get_session_lock(thread_id)

    try:
        async with session_lock:
            async with _GRAPH_INVOKE_SEMAPHORE:
                import anyio
                final = await anyio.to_thread.run_sync(
                    invoke_graph_with_retry,
                    graph,
                    msg.content,
                    thread_id,
                )

        reply = final["messages"][-1].content

        log_session_event(thread_id, {
            "event": "message",
            "role": "assistant",
            "variant": variant,
            "content": reply,
        })

        thinking.content = reply
        await thinking.update()

    except Exception as e:
        logger.exception("Final failure in symposium on_message thread_id=%s", thread_id)

        if is_retryable_graph_error(e):
            user_facing_error = (
                "I'm having a temporary issue responding right now. "
                "Please try sending your message again in a few seconds."
            )
            error_kind = "retryable_backend_error"
        else:
            user_facing_error = (
                "An unexpected error occurred. Please try again."
            )
            error_kind = "unexpected_backend_error"

        log_session_event(thread_id, {
            "event": "error",
            "variant": variant,
            "error_kind": error_kind,
            "error_type": type(e).__name__,
            "error": str(e),
        })

        thinking.content = user_facing_error
        await thinking.update()
