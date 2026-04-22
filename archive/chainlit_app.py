"""
Archive Chainlit app — runs older bot versions for reference/testing.

Launch with:
    ARCHIVE_VERSION=dbt-mini chainlit run archive/chainlit_app.py

Supported values for ARCHIVE_VERSION:
    dbt-mini   — DBT Mini graph (single DBT node, no module routing)

Notes:
  - This app imports from the main tbtst_bot package (same virtual env).
  - Personas, resources, and TB-RAG-documents are shared with the active bot.
  - Run from the repo root so relative imports and prompt paths resolve.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is on the path when running from archive/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import chainlit as cl
from langchain_core.messages import HumanMessage

# ── version selection ──────────────────────────────────────────────────────────
VERSION = (os.getenv("ARCHIVE_VERSION") or "").strip().lower()

SUPPORTED = {"dbt-mini"}
if VERSION not in SUPPORTED:
    raise SystemExit(
        f"ARCHIVE_VERSION='{VERSION}' is not supported.\n"
        f"Supported values: {', '.join(sorted(SUPPORTED))}\n\n"
        f"Example:  ARCHIVE_VERSION=dbt-mini chainlit run archive/chainlit_app.py"
    )

# ── graph import ───────────────────────────────────────────────────────────────
if VERSION == "dbt-mini":
    from tbtst_bot.graph import GRAPH_DBT_MINI as GRAPH  # type: ignore
    DISPLAY_VERSION = "DBT Mini"

# ── minimal Chainlit handlers ──────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start() -> None:
    thread_id = cl.context.session.id
    cl.user_session.set("thread_id", thread_id)
    await cl.Message(
        content=f"**[Archive: {DISPLAY_VERSION}]** Chat started. Thread: `{thread_id}`"
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}

    result = await cl.make_async(GRAPH.invoke)(
        {"messages": [HumanMessage(content=message.content)]},
        config=config,
    )

    messages = result.get("messages", [])
    reply = messages[-1].content if messages else "(no response)"
    await cl.Message(content=reply).send()
