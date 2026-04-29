"""
Symposium-facing Chainlit app for the UW TB Symposium (April 29, 2026).

Identical to chainlit_app.py except:
  - No chat profile / persona selector
  - English welcome message with overview, example questions, and a warning
  - Uses a fixed generic onboarding profile (no persona context)

Run with:
    uv run chainlit run chainlit_app_symposium.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from uuid import uuid4

import anyio
import boto3
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
# Translation layer (Amazon Translate — no LLM, ~100ms)
# -------------------------

# Lazy singleton — created once, reused across requests.
_translate_client = None


def _get_translate_client():
    global _translate_client
    if _translate_client is None:
        import os
        _translate_client = boto3.client(
            "translate",
            region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2",
        )
    return _translate_client


def _translate_to_english_sync(text: str) -> str:
    client = _get_translate_client()
    response = client.translate_text(
        Text=text,
        SourceLanguageCode="es",
        TargetLanguageCode="en",
    )
    return response["TranslatedText"]


async def translate_to_english(text: str) -> str:
    return await anyio.to_thread.run_sync(_translate_to_english_sync, text)

# -------------------------
# Symposium welcome copy
# -------------------------

SYMPOSIUM_WELCOME = """\
## Welcome to the TB Treatment Support Bot

**💬 Share your feedback:** After each bot response, you'll see a 👍 / 👎 button. Click it to rate the response and optionally leave a comment — your input helps us improve the bot.

---

This chatbot was developed at the University of Washington to support people navigating TB treatment. \
It can help with questions about medications and side effects, and it also offers emotional support \
for the day-to-day challenges that come with a long treatment journey.

You can ask in **English or Spanish** — the bot will respond in **English only**.

**Some examples of what you can ask:**
- *"What warning signs do I watch for that mean my symptoms are getting worse or that I need to seek immediate medical attention?"*
- *"I get so angry when the doctors don't listen. How can I manage these feelings?"*
- *"I feel very fatigued. Is this a medication side effect? How long will it last?"*
- *"My chest feels tight before I grab a drink. How do I know if it's just stress or if I really want to drink?"*

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
    "Always respond in Spanish. Be clear, concise, and avoid unnecessary jargon."
)


# Override the chat profiles registered by chainlit_app's module-level import.
# Without this, importing chainlit_app causes @cl.set_chat_profiles to fire and
# the 4 persona profiles from the main app appear in the symposium UI.
@cl.set_chat_profiles
async def chat_profiles(current_user=None):
    return None


# -------------------------
# Data layer
# -------------------------

@cl.data_layer
def get_data_layer() -> SQLAlchemyDataLayer:
    return SQLAlchemyDataLayer(conninfo=_build_async_conninfo(), ssl_require=False)


@cl.header_auth_callback
def header_auth_callback(headers: dict) -> cl.User:
    return cl.User(identifier=f"guest-{uuid4().hex[:8]}", metadata={"role": "guest"})


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
                final = await anyio.to_thread.run_sync(
                    invoke_graph_with_retry,
                    graph,
                    msg.content,
                    thread_id,
                )

        reply_es = final["messages"][-1].content
        reply_en = await translate_to_english(reply_es)

        log_session_event(thread_id, {
            "event": "message",
            "role": "assistant",
            "variant": variant,
            "content_es": reply_es,
            "content_en": reply_en,
        })

        thinking.content = reply_en
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
