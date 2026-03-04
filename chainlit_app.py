# chainlit_app.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import chainlit as cl
from chainlit.config import ChainlitConfigOverrides, UISettings


# UI banner copy (concise, readable; does NOT say "TB-TST")
BANNER_INTRO_MD = (
    "**Hi! I’m a TB treatment support bot.**\n\n"
    "You can ask questions about TB (treatment, side effects, routines) **or** just tell me what’s bothering you.\n\n"
    "*Persona hint:* The banner below describes a sample person. "
    "You can use **any** details from it (or **none**) when asking questions."
)

# If you still want a chat message, set this True and keep it short.
SEND_CHAT_WELCOME_MESSAGE = False
WELCOME_MESSAGE = "Hi! I’m a TB treatment support bot. Ask TB questions or tell me what’s bothering you."


PERSONAS_DIR = Path(__file__).parent / "personas"


@dataclass(frozen=True)
class PersonaUI:
    profile_name: str      # shows in profile picker
    banner_title: str      # UI banner title
    persona_md: str        # persona content (markdown)
    icon: Optional[str] = None


def parse_persona_txt(text: str, fallback_name: str) -> Tuple[str, str]:
    """
    Expected persona file style:

      Daniel

      Description:
      ...

      Mental Health Issues:
      ...

      Questions to ask the conversational AI:
        ...

    Parsing rules:
      - First non-empty line => title
      - Rest => body, with "Section:" lines turned into markdown headings
    """
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    # Title = first non-empty line
    title = None
    start_idx = 0
    for i, ln in enumerate(lines):
        if ln.strip():
            title = ln.strip()
            start_idx = i + 1
            break
    if not title:
        title = fallback_name

    # Body = rest (trim leading blank lines)
    body_lines = lines[start_idx:]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)

    body_raw = "\n".join(body_lines).strip()
    if not body_raw:
        body_raw = "Persona details."

    # Light formatting: "Something:" -> heading
    out: List[str] = []
    for ln in body_raw.split("\n"):
        s = ln.strip()
        if s.endswith(":") and len(s) <= 60:
            out.append(f"### {s[:-1]}")
        else:
            out.append(ln)

    body_md = "\n".join(out).strip()
    return title, body_md


def build_banner_description(persona_md: str) -> str:
    """
    Final banner description shown in the UI.
    Includes intro + persona content (but still concise and readable).
    """
    return (
        f"{BANNER_INTRO_MD}\n\n"
        f"---\n\n"
        f"{persona_md}"
    )


def load_personas() -> Dict[str, PersonaUI]:
    personas: Dict[str, PersonaUI] = {}

    if not PERSONAS_DIR.exists():
        personas["Default"] = PersonaUI(
            profile_name="Default",
            banner_title="Persona",
            persona_md="### Description\nNo persona files found in `./personas/`.",
        )
        return personas

    files = sorted(PERSONAS_DIR.glob("*.txt"))
    if not files:
        personas["Default"] = PersonaUI(
            profile_name="Default",
            banner_title="Persona",
            persona_md="### Description\nNo persona files found in `./personas/`.",
        )
        return personas

    for f in files:
        raw = f.read_text(encoding="utf-8").strip()
        fallback_name = f.stem.replace("_", " ").title()
        title, persona_md = parse_persona_txt(raw, fallback_name)

        personas[title] = PersonaUI(
            profile_name=title,
            banner_title=title,     # banner title = persona name (e.g., Daniel)
            persona_md=persona_md,  # persona body
            icon=None,
        )

    return personas


PERSONAS = load_personas()


@cl.set_chat_profiles
async def chat_profiles(current_user: Optional[cl.User] = None):
    profiles: List[cl.ChatProfile] = []
    for persona in PERSONAS.values():
        banner_desc = build_banner_description(persona.persona_md)

        profiles.append(
            cl.ChatProfile(
                name=persona.profile_name,
                # Profile picker description (keep short)
                markdown_description="Shows this persona in the banner.",
                icon=persona.icon,
                config_overrides=ChainlitConfigOverrides(
                    ui=UISettings(
                        name=persona.banner_title,
                        description=banner_desc,  # banner includes intro + persona
                    )
                ),
            )
        )
    return profiles


@cl.on_chat_start
async def on_chat_start():
    # Persona is shown in banner; we don't force it into the chat.
    if SEND_CHAT_WELCOME_MESSAGE:
        await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def on_message(msg: cl.Message):
    # Hook your actual bot/graph response here.
    await cl.Message(content="(hook your existing bot response here)").send()
