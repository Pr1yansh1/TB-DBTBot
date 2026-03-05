# ui_types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class PersonaUI:
    profile_name: str
    banner_title: str
    persona_md: str
    icon: Optional[str] = None
