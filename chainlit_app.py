from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from uuid import uuid4

import anyio
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from langchain_core.messages import HumanMessage

# Warmup should use your cached RAG objects (rag_utils.py) instead of loading SentenceTransformer here.
from tbtst_bot.rag_utils import warmup_rag

try:
    from tbtst_bot.graph import GRAPH_DBT_MINI, GRAPH_DBT_FULL
except Exception:  # pragma: no cover
    from src.tbtst_bot.graph import GRAPH_DBT_MINI, GRAPH_DBT_FULL  # type: ignore


logger = logging.getLogger("tbtst.chainlit_app")

# -------------------------
# UI / Banner copy
# -------------------------
BANNER_INTRO_MD = (
    "**Hi! I’m a TB treatment support bot.**\n\n"
    "For this chat, I already have the background profile below from prior onboarding, "
    "and I can use it as ongoing context throughout the conversation.\n\n"
    "You can ask questions about TB (treatment, side effects, routines) **or** tell me what’s bothering you today."
)

SEND_CHAT_WELCOME_MESSAGE = False
WELCOME_MESSAGE = "Hi! I’m a TB treatment support bot. Ask TB questions or tell me what’s bothering you."

# Support either ./personas or ./persona
PERSONAS_DIR_CANDIDATES = [
    Path(__file__).parent / "personas",
    Path(__file__).parent / "persona",
]

# Per-session transcript storage
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"

# Production resiliency knobs
CHAINLIT_GRAPH_MAX_ATTEMPTS = int(os.getenv("TBTST_CHAINLIT_GRAPH_MAX_ATTEMPTS", "1"))
CHAINLIT_GRAPH_BASE_SLEEP_SECONDS = float(os.getenv("TBTST_CHAINLIT_GRAPH_BASE_SLEEP_SECONDS", "1.25"))
MAX_CONCURRENT_GRAPH_INVOCATIONS = int(os.getenv("TBTST_MAX_CONCURRENT_GRAPH_INVOCATIONS", "4"))

# Global concurrency control and per-session serialization
_GRAPH_INVOKE_SEMAPHORE = anyio.Semaphore(MAX_CONCURRENT_GRAPH_INVOCATIONS)
# Per-thread locks used by on_chat_start (race guard) and on_message (serialize
# concurrent sends within a session). Grows by one entry per unique thread_id and
# is never evicted — intentionally acceptable at research prototype scale
# (~10–20 evaluator sessions total).
_SESSION_LOCKS: Dict[str, anyio.Lock] = {}


# -------------------------
# Warmup (ONCE per process)
# -------------------------
_WARMED_UP = False
_WARMUP_LOCK = anyio.Lock()


async def warmup_once() -> None:
    global _WARMED_UP
    if _WARMED_UP:
        return
    async with _WARMUP_LOCK:
        if _WARMED_UP:
            return
        # warmup_rag() initializes cached embeddings + chroma vectorstore
        await anyio.to_thread.run_sync(warmup_rag)
        _WARMED_UP = True


class PersonaUI:
    __slots__ = ("profile_name", "banner_title", "persona_md", "icon")

    def __init__(self, profile_name: str, banner_title: str, persona_md: str, icon: Optional[str] = None):
        self.profile_name = profile_name
        self.banner_title = banner_title
        self.persona_md = persona_md
        self.icon = icon


def _resolve_personas_dir() -> Path:
    for p in PERSONAS_DIR_CANDIDATES:
        if p.exists():
            return p
    return PERSONAS_DIR_CANDIDATES[0]


PERSONAS_DIR = _resolve_personas_dir()


def _safe_session_filename(thread_id: str) -> str:
    return thread_id.replace("/", "_").replace("\\", "_").replace(":", "__")


def _transcript_path(thread_id: str) -> Path:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    return TRANSCRIPTS_DIR / f"{_safe_session_filename(thread_id)}.jsonl"


def log_session_event(thread_id: str, payload: Dict[str, Any]) -> None:
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        **payload,
    }
    path = _transcript_path(thread_id)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_persona_txt(text: str, fallback_name: str) -> Tuple[str, str]:
    """
    Persona file format:

      Daniel

      Description:
      ...

      Mental Health Concerns:
      ...

      Suggestions (example things you could ask):
      - ...

    Rules:
      - First non-empty line => title
      - Rest => body
      - Lines ending with ":" become headings
    """
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    title = None
    start_idx = 0
    for i, ln in enumerate(lines):
        if ln.strip():
            title = ln.strip()
            start_idx = i + 1
            break
    if not title:
        title = fallback_name

    body_lines = lines[start_idx:]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)

    body_raw = "\n".join(body_lines).strip() or "Persona details."

    out: List[str] = []
    for ln in body_raw.split("\n"):
        s = ln.strip()
        if s.endswith(":") and len(s) <= 80:
            out.append(f"### {s[:-1]}")
        else:
            out.append(ln)

    return title, "\n".join(out).strip()


def build_banner_message(persona_title: str, persona_md: str) -> str:
    """
    Visible "banner" in Chainlit: first message sent on chat start.
    Makes it explicit that the bot already has this background info.
    """
    return (
        f"{BANNER_INTRO_MD}\n\n"
        f"---\n\n"
        f"**Background profile already available for this conversation: {persona_title}**\n\n"
        f"{persona_md}"
    )


def build_onboarding_profile(persona_title: str, persona_md: str) -> str:
    """
    Hidden background context stored in graph state.
    """
    return (
        "User background profile for this conversation.\n"
        "This profile comes from prior onboarding and should be treated as stable ongoing context.\n"
        "The details below are known background facts about the user for this chat and may be used throughout the conversation.\n\n"
        f"Persona: {persona_title}\n\n"
        f"{persona_md}"
    )


def load_personas() -> Dict[str, PersonaUI]:
    personas: Dict[str, PersonaUI] = {}

    if not PERSONAS_DIR.exists():
        personas["Default"] = PersonaUI(
            profile_name="Default",
            banner_title="Default",
            persona_md="### Description\nNo persona files found in `./personas/` or `./persona/`.",
        )
        return personas

    files = sorted(PERSONAS_DIR.glob("*.txt"))
    if not files:
        personas["Default"] = PersonaUI(
            profile_name="Default",
            banner_title="Default",
            persona_md="### Description\nNo persona files found in `./personas/` or `./persona/`.",
        )
        return personas

    for f in files:
        raw = f.read_text(encoding="utf-8").strip()
        fallback_name = f.stem.replace("_", " ").title()
        title, persona_md = parse_persona_txt(raw, fallback_name)

        personas[title] = PersonaUI(
            profile_name=title,
            banner_title=title,
            persona_md=persona_md,
            icon=None,
        )

    return personas


PERSONAS = load_personas()


# -------------------------
# Chainlit data layer + auth
#
# The data layer makes session.thread_id stable across reconnects so that
# on_chat_resume fires (instead of on_chat_start) when a user returns to an
# existing conversation. This is the fix for Type B long-gap re-init crashes.
# -------------------------

# Default Chainlit session DB. Independent of the LangGraph checkpointer path.
# Override with TBTST_CHAINLIT_DB (any async-driver URL) or DATABASE_URL (Postgres).
_DEFAULT_CHAINLIT_DB = "./data/chainlit.db"


def _build_async_conninfo() -> str:
    """
    Return an async-driver connection string for SQLAlchemyDataLayer.

    Resolution order:
      1. TBTST_CHAINLIT_DB  — explicit override; must already use an async driver
      2. DATABASE_URL       — swap in asyncpg driver for postgres:// / postgresql://
      3. Default            — SQLite at ./data/chainlit.db

    The LangGraph checkpointer (graph.py) uses its own separate SQLite file and
    its own env var (TBTST_CHECKPOINT_DB). The two stores are intentionally
    independent.
    """
    explicit = os.getenv("TBTST_CHAINLIT_DB", "").strip()
    if explicit:
        logger.info("Chainlit data layer: using TBTST_CHAINLIT_DB override")
        return explicit

    db_url = os.getenv("DATABASE_URL", "").strip()
    if db_url:
        for sync_prefix, async_prefix in (
            ("postgresql://", "postgresql+asyncpg://"),
            ("postgres://", "postgresql+asyncpg://"),
        ):
            if db_url.startswith(sync_prefix):
                conninfo = async_prefix + db_url[len(sync_prefix):]
                logger.info("Chainlit data layer: using DATABASE_URL (asyncpg)")
                return conninfo
        logger.info("Chainlit data layer: using DATABASE_URL as-is")
        return db_url

    sqlite_path = os.path.abspath(_DEFAULT_CHAINLIT_DB)
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    logger.info("Chainlit data layer: using default SQLite at %s", sqlite_path)
    return f"sqlite+aiosqlite:///{sqlite_path}"


@cl.data_layer
def get_data_layer() -> SQLAlchemyDataLayer:
    return SQLAlchemyDataLayer(conninfo=_build_async_conninfo(), ssl_require=False)


@cl.header_auth_callback
def header_auth_callback(headers: Dict[str, str]) -> Optional[cl.User]:
    """
    Return an anonymous user so that session.user is populated, which is
    required for Chainlit's resume_thread logic to fire on_chat_resume.
    Real auth can be added here later without changing on_chat_resume.
    """
    return cl.User(identifier="anonymous", metadata={})


def select_graph() -> Tuple[Any, str]:
    v = (os.getenv("TBTST_VARIANT") or "full").strip().lower()
    if v == "mini":
        return GRAPH_DBT_MINI, "mini"
    return GRAPH_DBT_FULL, "full"


def seed_onboarding_profile_sync(graph: Any, thread_id: str, onboarding_profile: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    graph.update_state(config, {"onboarding_profile": onboarding_profile})


async def seed_onboarding_profile(graph: Any, thread_id: str, onboarding_profile: str) -> None:
    await anyio.to_thread.run_sync(
        seed_onboarding_profile_sync,
        graph,
        thread_id,
        onboarding_profile,
    )


def is_retryable_graph_error(e: Exception) -> bool:
    msg = str(e)
    retry_markers = [
        "ServiceUnavailableException",
        "Service Unavailable",
        "Too many connections",
        "ThrottlingException",
        "Too many requests",
        "Rate exceeded",
        "InternalServerException",
        "ModelTimeoutException",
        "RequestTimeout",
        "TimeoutError",
        "timed out",
        "ReadTimeout",
        "Connection reset",
        "Connection aborted",
    ]
    return any(marker in msg for marker in retry_markers)


def _sleep_seconds_for_attempt(e: Exception, attempt_index: int) -> float:
    msg = str(e)
    if "Too many connections" in msg or "ServiceUnavailableException" in msg:
        base = CHAINLIT_GRAPH_BASE_SLEEP_SECONDS
        cap = 4.0
    else:
        base = 0.75
        cap = 3.0
    jitter = random.uniform(0.0, 0.35)
    return min(cap, base * (2 ** attempt_index)) + jitter


def invoke_graph_with_retry(graph: Any, user_text: str, thread_id: str) -> Dict[str, Any]:
    payload = {"messages": [HumanMessage(content=user_text)]}
    config = {"configurable": {"thread_id": thread_id}}

    last_exc: Optional[Exception] = None

    for attempt in range(CHAINLIT_GRAPH_MAX_ATTEMPTS):
        try:
            return graph.invoke(payload, config=config)
        except Exception as e:
            last_exc = e
            retryable = is_retryable_graph_error(e)
            is_last_attempt = attempt >= (CHAINLIT_GRAPH_MAX_ATTEMPTS - 1)

            logger.exception(
                "graph.invoke failed thread_id=%s attempt=%d/%d retryable=%s",
                thread_id,
                attempt + 1,
                CHAINLIT_GRAPH_MAX_ATTEMPTS,
                retryable,
            )

            if (not retryable) or is_last_attempt:
                raise

            sleep_s = _sleep_seconds_for_attempt(e, attempt)
            logger.warning(
                "Retrying graph.invoke thread_id=%s in %.2fs due to transient error: %s",
                thread_id,
                sleep_s,
                type(e).__name__,
            )
            time.sleep(sleep_s)

    raise last_exc or RuntimeError("graph.invoke failed without a captured exception")


def _get_session_lock(thread_id: str) -> anyio.Lock:
    lock = _SESSION_LOCKS.get(thread_id)
    if lock is None:
        lock = anyio.Lock()
        _SESSION_LOCKS[thread_id] = lock
    return lock


@cl.set_chat_profiles
async def chat_profiles(current_user: Optional[cl.User] = None):
    return [
        cl.ChatProfile(
            name=p.profile_name,
            markdown_description="Select a persona profile for the conversation.",
            icon=p.icon,
        )
        for p in PERSONAS.values()
    ]


def _thread_already_initialized(graph: Any, thread_id: str) -> bool:
    """
    Return True if this thread already has an onboarding profile in the graph
    checkpointer. Used as an idempotency guard in on_chat_start so that
    reconnects and concurrent double-fires don't re-send the intro messages.
    Falls back to False on any error so a genuinely broken state still inits.
    """
    try:
        snap = graph.get_state({"configurable": {"thread_id": thread_id}})
        return bool(snap and snap.values.get("onboarding_profile"))
    except Exception:
        return False


@cl.on_chat_start
async def on_chat_start():
    graph, variant = select_graph()

    session = getattr(cl.context, "session", None)
    # Use session.thread_id (stable conversation ID managed by Chainlit's data
    # layer) so that the same thread_id is used across WebSocket reconnects.
    # Falls back to session.id (WebSocket session ID) and finally a random UUID
    # when no data layer is configured.
    base_thread = (
        getattr(session, "thread_id", None)
        or getattr(session, "id", None)
        or f"cl-{uuid4().hex[:10]}"
    ) if session else f"cl-{uuid4().hex[:10]}"
    thread_id = f"{base_thread}:{variant}"

    cl.user_session.set("graph", graph)
    cl.user_session.set("variant", variant)
    cl.user_session.set("thread_id", thread_id)

    # Warm up RAG ONCE per process (not per chat)
    await warmup_once()

    # Guard: acquire the per-thread lock before checking state so that two
    # concurrent on_chat_start calls (race condition at connect) don't both
    # pass the check before either has finished seeding.
    async with _get_session_lock(thread_id):
        if _thread_already_initialized(graph, thread_id):
            logger.info(
                "on_chat_start skipped re-init for existing thread thread_id=%s", thread_id
            )
            return

        chosen = cl.user_session.get("chat_profile")
        if not chosen or chosen not in PERSONAS:
            chosen = next(iter(PERSONAS.keys()))

        persona = PERSONAS[chosen]
        onboarding_profile = build_onboarding_profile(persona.banner_title, persona.persona_md)

        cl.user_session.set("onboarding_profile", onboarding_profile)

        await seed_onboarding_profile(graph, thread_id, onboarding_profile)

        banner = build_banner_message(persona.banner_title, persona.persona_md)
        opener = "¿Qué te preocupa hoy?"

        log_session_event(
            thread_id,
            {
                "event": "session_start",
                "variant": variant,
                "persona": persona.banner_title,
                "onboarding_profile": onboarding_profile,
            },
        )
        log_session_event(
            thread_id,
            {
                "event": "message",
                "role": "assistant",
                "content": banner,
            },
        )
        log_session_event(
            thread_id,
            {
                "event": "message",
                "role": "assistant",
                "content": opener,
            },
        )

        await cl.Message(content=banner).send()
        await cl.Message(content=opener).send()

        if SEND_CHAT_WELCOME_MESSAGE:
            log_session_event(
                thread_id,
                {
                    "event": "message",
                    "role": "assistant",
                    "content": WELCOME_MESSAGE,
                },
            )
            await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """
    Called by Chainlit when a user returns to an existing conversation (the
    data layer + header_auth_callback make this possible). Restores the graph
    and thread_id into user_session — no greeting is sent.

    This is the complement to the idempotency guard in on_chat_start: together
    they cover every reconnect scenario:
      - on_chat_resume:  data layer available, returning user (Type B long-gap)
      - on_chat_start idempotency guard: same process, state already in graph
    """
    graph, variant = select_graph()
    thread_id = f"{thread['id']}:{variant}"

    cl.user_session.set("graph", graph)
    cl.user_session.set("variant", variant)
    cl.user_session.set("thread_id", thread_id)

    logger.info(
        "on_chat_resume: restored session thread_id=%s", thread_id
    )

    log_session_event(
        thread_id,
        {
            "event": "session_resume",
            "variant": variant,
            "thread_chainlit_id": thread["id"],
        },
    )


@cl.on_message
async def on_message(msg: cl.Message):
    graph = cl.user_session.get("graph")
    thread_id = cl.user_session.get("thread_id")
    variant = cl.user_session.get("variant") or ""

    if graph is None or thread_id is None:
        await cl.Message(content="Error: sesión no iniciada. Por favor recarga la página.").send()
        return

    log_session_event(
        thread_id,
        {
            "event": "message",
            "role": "user",
            "variant": variant,
            "content": msg.content,
        },
    )

    thinking = cl.Message(content="")
    await thinking.send()

    session_lock = _get_session_lock(thread_id)

    try:
        async with session_lock:
            async with _GRAPH_INVOKE_SEMAPHORE:
                final: Dict[str, Any] = await anyio.to_thread.run_sync(
                    invoke_graph_with_retry,
                    graph,
                    msg.content,
                    thread_id,
                )

        reply = final["messages"][-1].content

        log_session_event(
            thread_id,
            {
                "event": "message",
                "role": "assistant",
                "variant": variant,
                "content": reply,
            },
        )

        thinking.content = reply
        await thinking.update()

    except Exception as e:
        logger.exception("Final failure in Chainlit on_message thread_id=%s", thread_id)

        if is_retryable_graph_error(e):
            user_facing_error = (
                "Lo siento, estoy teniendo un problema temporal para responder en este momento. "
                "Por favor intenta enviar tu mensaje otra vez en unos segundos."
            )
            error_kind = "retryable_backend_error"
        else:
            user_facing_error = (
                "Lo siento, ocurrió un error inesperado. "
                "Por favor intenta de nuevo."
            )
            error_kind = "unexpected_backend_error"

        log_session_event(
            thread_id,
            {
                "event": "error",
                "variant": variant,
                "error_kind": error_kind,
                "error_type": type(e).__name__,
                "error": str(e),
            },
        )

        thinking.content = user_facing_error
        await thinking.update()
