from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from uuid import uuid4

import anyio
import chainlit as cl
from langchain_core.messages import HumanMessage

# Warmup should use your cached RAG objects (rag_utils.py) instead of loading SentenceTransformer here.
from tbtst_bot.rag_utils import warmup_rag

try:
    from tbtst_bot.graph import GRAPH_DBT_MINI, GRAPH_DBT_FULL
except Exception:  # pragma: no cover
    from src.tbtst_bot.graph import GRAPH_DBT_MINI, GRAPH_DBT_FULL  # type: ignore


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


def select_graph() -> Tuple[Any, str]:
    v = (os.getenv("TBTST_VARIANT") or "mini").strip().lower()
    if v == "full":
        return GRAPH_DBT_FULL, "full"
    return GRAPH_DBT_MINI, "mini"


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


@cl.on_chat_start
async def on_chat_start():
    graph, variant = select_graph()

    session = getattr(cl.context, "session", None)
    base_thread = session.id if session and getattr(session, "id", None) else f"cl-{uuid4().hex[:10]}"
    thread_id = f"{base_thread}:{variant}"

    cl.user_session.set("graph", graph)
    cl.user_session.set("variant", variant)
    cl.user_session.set("thread_id", thread_id)

    # Warm up RAG ONCE per process (not per chat)
    await warmup_once()

    chosen = cl.user_session.get("chat_profile")
    if not chosen or chosen not in PERSONAS:
        chosen = next(iter(PERSONAS.keys()))

    persona = PERSONAS[chosen]
    onboarding_profile = build_onboarding_profile(persona.banner_title, persona.persona_md)

    cl.user_session.set("onboarding_profile", onboarding_profile)

    await seed_onboarding_profile(graph, thread_id, onboarding_profile)
    await cl.Message(content=build_banner_message(persona.banner_title, persona.persona_md)).send()
    await cl.Message(content="¿Qué te preocupa hoy?").send()

    if SEND_CHAT_WELCOME_MESSAGE:
        await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def on_message(msg: cl.Message):
    graph = cl.user_session.get("graph")
    thread_id = cl.user_session.get("thread_id")

    if graph is None or thread_id is None:
        await cl.Message(content="Error: session not initialized. Please refresh.").send()
        return

    thinking = cl.Message(content="")
    await thinking.send()

    try:
        final: Dict[str, Any] = await anyio.to_thread.run_sync(
            lambda: graph.invoke(
                {"messages": [HumanMessage(content=msg.content)]},
                config={"configurable": {"thread_id": thread_id}},
            )
        )
        reply = final["messages"][-1].content
        thinking.content = reply
        await thinking.update()
    except Exception as e:
        thinking.content = f"❌ Error: {type(e).__name__}: {e}"
        await thinking.update()
