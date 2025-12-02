# app/keywords.py
from pathlib import Path
from typing import List

from .config import KEYWORDS_DIR


def _load_phrases(name: str) -> List[str]:
    """
    Load one keyword file from keywords/ as a list of lowercase phrases.

    - Skips blank lines and lines starting with '#'
    - Strips whitespace
    """
    path = KEYWORDS_DIR / f"{name}.txt"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    phrases: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        phrases.append(s.lower())
    return phrases


# ---------- Safety / crisis detection ----------
# These lists are meant to reflect signal from the systematic review
# + clinical scales like C-SSRS and PHQ-9 item 9.

# Explicit “this is an emergency / I’m about to act” phrases
SAFETY_CRISIS_PHRASES = _load_phrases("safety_crisis")

# Intensifiers & superlatives that amplify risk terms (e.g. “really”, “so”, “now”)
SAFETY_INTENSIFIERS = _load_phrases("safety_intensifiers")

# Passive ideation: “wish I were dead”, “better off dead”, “don’t want to be here”
SAFETY_PASSIVE_PHRASES = _load_phrases("safety_passive_ideation")

# Active ideation: explicit thoughts of killing oneself / ending life
SAFETY_ACTIVE_PHRASES = _load_phrases("safety_active_ideation")

# Suicidal / self-harm behaviors or attempts (“cut last night”, “overdosed”, etc.)
SAFETY_BEHAVIOR_PHRASES = _load_phrases("safety_behavior")

# Mentions of specific means or preparation (“pills”, “rope”, “bridge”, “gun”, etc.)
SAFETY_MEANS_PHRASES = _load_phrases("safety_means")

# Time-urgency phrases (“right now”, “tonight”, “can’t go on anymore”, etc.)
SAFETY_TIME_URGENCY_PHRASES = _load_phrases("safety_time_urgency")

# ---------- Domain routing ----------
FAQ_KEYWORDS = _load_phrases("domain_faq")
DBT_KEYWORDS = _load_phrases("domain_dbt")
# Psychoeducation is the default; you can add domain_psycho.txt if needed later.

