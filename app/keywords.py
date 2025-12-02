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
        # You could log a warning here if you want
        return []
    text = path.read_text(encoding="utf-8")
    phrases: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        phrases.append(s.lower())
    return phrases


# Safety / crisis detection
SAFETY_CRISIS_PHRASES = _load_phrases("safety_crisis")
SAFETY_INTENSIFIERS = _load_phrases("safety_intensifiers")

# Domain routing
FAQ_KEYWORDS = _load_phrases("domain_faq")
DBT_KEYWORDS = _load_phrases("domain_dbt")
# Psychoeducation is the default; you can add domain_psycho.txt if needed later.

