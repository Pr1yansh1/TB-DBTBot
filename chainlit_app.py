# chainlit_app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from uuid import uuid4

import anyio
import chainlit as cl
from langchain_core.messages import HumanMessage

try:
    from tbtst_bot.graph import GRAPH_DBT_MINI, GRAPH_DBT_FULL
except Exception:  # pragma: no cover
    from .src.tbtst_bot.graph import GRAPH_DBT_MINI, GRAPH_DBT_FULL  # adjust only if needed


# -------------------------
# UI / Banner copy (NOT "TB TST")
# -------------------------
BANNER_INTRO_MD = (
    "**Hi! I’m a TB treatment support bot.**\n\n"
    "You can ask questions about TB (treatment, side effects, routines) **or** just tell me what’s bothering you.\n\n"
    "*Persona hint:* The persona below describes a sample person. "
    "You can use **any** details from it (or **none**) when asking questions."
)

SEND_CHAT_WELCOME_MESSAGE = False
WELCOME_MESSAGE = "Hi! I’m a TB treatment support bot. Ask TB questions or tell me what’s bothering you."

PERSONAS_DIR = Path(__file__).parent / "personas"

# -------------------------
# RAG warmup settings
# -------------------------
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(Path(__file__).parent / "cache" / "chroma")))
EMBED_MODEL_NAME = os.getenv("TBTST_EMBED_MODEL", "hiiamsid/sentence_similarity_spanish_es")


class PersonaUI:
    __slots__ = ("profile_name", "banner_title", "persona_md", "icon")

    def __init__(self, profile_name: str, banner_title: str, persona_md: str, icon: Optional[str] = None):
        self.profile_name = profile_name
        self.banner_title = banner_title
        self.persona_md = persona_md
        self.icon = icon


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
    This is the visible "banner" in Chainlit: a first message sent on chat start.
    (Chainlit does not render ui.description as an in-chat banner.)
    """
    return f"{BANNER_INTRO_MD}\n\n---\n\n**Persona: {persona_title}**\n\n{persona_md}"


def load_personas() -> Dict[str, PersonaUI]:
    personas: Dict[str, PersonaUI] = {}

    if not PERSONAS_DIR.exists():
        personas["Default"] = PersonaUI(
            profile_name="Default",
            banner_title="Default",
            persona_md="### Description\nNo persona files found in `./personas/`.",
        )
        return personas

    files = sorted(PERSONAS_DIR.glob("*.txt"))
    if not files:
        personas["Default"] = PersonaUI(
            profile_name="Default",
            banner_title="Default",
            persona_md="### Description\nNo persona files found in `./personas/`.",
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


def _warmup_rag_sync() -> None:
    """
    Best-effort warmup:
      - Ensure Chroma persist dir exists
      - Load embedding model (SentenceTransformer) once
      - Initialize a Chroma client/collection to force DB open

    This does NOT change your RAG logic; it just preloads expensive bits.
    """
    try:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If the dir can't be created, we'll still try to proceed.
        pass

    # 1) Load embedding model
    try:
        from sentence_transformers import SentenceTransformer

        _ = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    except Exception as e:
        # Don't crash the app on warmup failures—log and keep going.
        # The actual RAG path may still work if it uses a different embedder.
        print(f"[warmup] Embedding model warmup failed: {type(e).__name__}: {e}")

    # 2) Initialize Chroma persistence
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        # touching list_collections forces open/reads
        _ = client.list_collections()
    except Exception as e:
        print(f"[warmup] Chroma warmup failed: {type(e).__name__}: {e}")


@cl.set_chat_profiles
async def chat_profiles(current_user: Optional[cl.User] = None):
    # Profiles appear in UI; persona content is UI-only.
    return [
        cl.ChatProfile(
            name=p.profile_name,
            markdown_description="Select a persona (shown in the banner message).",
            icon=p.icon,
        )
        for p in PERSONAS.values()
    ]


@cl.on_chat_start
async def on_chat_start():
    # 1) Store graph + thread id (separate memory per session/variant)
    graph, variant = select_graph()

    session = getattr(cl.context, "session", None)
    base_thread = session.id if session and getattr(session, "id", None) else f"cl-{uuid4().hex[:10]}"
    thread_id = f"{base_thread}:{variant}"

    cl.user_session.set("graph", graph)
    cl.user_session.set("variant", variant)
    cl.user_session.set("thread_id", thread_id)

    # 2) Warm up RAG in a background thread (so we don't block the event loop)
    await anyio.to_thread.run_sync(_warmup_rag_sync)

    # 3) Show persona banner as first message (Chainlit-native "banner" approach)
    chosen = cl.user_session.get("chat_profile")
    if not chosen or chosen not in PERSONAS:
        chosen = next(iter(PERSONAS.keys()))

    persona = PERSONAS[chosen]
    banner_text = build_banner_message(persona.banner_title, persona.persona_md)
    await cl.Message(content=banner_text).send()

    # Optional extra welcome message (usually unnecessary if banner exists)
    if SEND_CHAT_WELCOME_MESSAGE:
        await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def on_message(msg: cl.Message):
    graph = cl.user_session.get("graph")
    thread_id = cl.user_session.get("thread_id")

    if graph is None or thread_id is None:
        await cl.Message(content="Error: session not initialized. Please refresh.").send()
        return

    # Run sync graph.invoke in a worker thread to avoid UI disconnects/timeouts.
    thinking = cl.Message(content="")  # empty bubble we can update
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
