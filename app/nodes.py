# app/nodes.py
import json
from typing import Dict, Any, List

from .state import ConversationState
from .llm import bedrock_chat
from .prompts import (
    GLOBAL_SYSTEM_PROMPT,
    SAFETY_SYSTEM_PROMPT,
    CRISIS_SYSTEM_PROMPT,
    FAQ_SYSTEM_PROMPT,
    DBT_SYSTEM_PROMPT,
    PSYCHO_SYSTEM_PROMPT,
    TB_FAQ_KB,
    DBT_SKILLS_KB,
    compose_system_prompt,
)
from .keywords import (
    SAFETY_CRISIS_PHRASES,
    SAFETY_INTENSIFIERS,
    FAQ_KEYWORDS,
    DBT_KEYWORDS,
)


# ---------- Helper: extract latest user text ----------

def _latest_user_text(state: ConversationState) -> str:
    msgs = state.get("messages") or []
    for msg in reversed(msgs):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ---------- Ingress node ----------

def ingress_node(state: ConversationState) -> ConversationState:
    """
    Minimal ingress node.
    Hook for global normalization / logging later.
    """
    meta = state.get("meta") or {}
    meta["ingress_seen"] = True
    state["meta"] = meta
    return state


# ---------- Safety gate ----------

def _check_crisis_keywords(text: str) -> bool:
    """
    Simple keyword-based heuristic for acute risk.
    - Direct crisis phrases
    - Combination of high-risk terms + intensifiers
    """
    t = text.lower()
    if any(phrase in t for phrase in SAFETY_CRISIS_PHRASES):
        return True

    if SAFETY_INTENSIFIERS:
        # If we have intensifiers configured, check for co-occurrence
        has_intensifier = any(word in t for word in SAFETY_INTENSIFIERS)
        has_risky_term = any(word in t for word in ["die", "kill myself", "hurt myself", "self-harm"])
        if has_intensifier and has_risky_term:
            return True

    return False


def safety_node(state: ConversationState) -> ConversationState:
    """
    Classify safety level for the latest user message using:
      1) Keyword heuristic
      2) LLM classifier (JSON output)
    """
    text = _latest_user_text(state)
    keyword_hit = _check_crisis_keywords(text)

    # LLM classification
    messages = [
        {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    raw = bedrock_chat(messages, max_tokens=220, temperature=0.0)

    level = "low"
    flags: List[str] = []
    reason = "no_reason"
    parse_error: str | None = None

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            level = str(data.get("level", "low")).lower()
            reason = str(data.get("reason", "") or "no_reason")
            if isinstance(data.get("flags"), list):
                flags = [str(f) for f in data["flags"]]
    except Exception as e:  # noqa: BLE001
        parse_error = f"{type(e).__name__}: {e}"
        # Fall back to keyword + conservative default

    if parse_error:
        if keyword_hit:
            level = "high"
            reason = f"fallback_parse_error_and_keyword_hit: {parse_error}"
        else:
            level = "low"
            reason = f"fallback_parse_error: {parse_error}"

    is_crisis = keyword_hit or (level == "high")

    safety_info: Dict[str, Any] = {
        "level": level,
        "is_crisis": is_crisis,
        "reason": reason,
        "flags": flags,
        "keyword_hit": keyword_hit,
        "raw_classifier_output": raw,
    }

    state["safety"] = safety_info
    # For debugging
    meta = state.get("meta") or {}
    meta["safety_seen"] = True
    state["meta"] = meta

    return state


def safety_route(state: ConversationState) -> str:
    """
    Used by LangGraph as conditional router after safety_node.
    Returns:
      - "crisis"     -> crisis path
      - "non_crisis" -> continue to domain routing
    """
    safety = state.get("safety") or {}
    if safety.get("is_crisis"):
        state["route"] = "crisis"
        return "crisis"
    return "non_crisis"


# ---------- Domain router (FAQ / DBT / Psycho) ----------

def domain_router(state: ConversationState) -> str:
    """
    Decide whether the non-crisis turn is best handled by:
      - FAQ (TB logistics / side effects / treatment)
      - DBT (emotions, urges, conflict, coping)
      - Psychoeducation (default)
    """
    text = _latest_user_text(state).lower()

    # FAQ-ish if clear TB logistics / side-effects / treatment questions
    if FAQ_KEYWORDS and any(kw in text for kw in FAQ_KEYWORDS):
        state["route"] = "faq"
        return "faq"

    # DBT-ish if clear emotional coping / urges / interpersonal content
    if DBT_KEYWORDS and any(kw in text for kw in DBT_KEYWORDS):
        state["route"] = "dbt"
        return "dbt"

    # Fallback: psychoeducation
    state["route"] = "psycho"
    return "psycho"


# ---------- Shared helper for leaf bots ----------

def _run_leaf_bot(state: ConversationState, system_prompt: str) -> ConversationState:
    """
    Append an assistant turn using the given system prompt and full history.

    - Uses only user/assistant messages from state["messages"] (no nested systems).
    - System prompt should already include global + domain + optional KB text.
    """
    history = state.get("messages") or []
    convo = [m for m in history if m.get("role") in ("user", "assistant")]

    messages = [
        {"role": "system", "content": system_prompt},
        *convo,
    ]
    reply = bedrock_chat(messages, max_tokens=220, temperature=0.2)
    history.append({"role": "assistant", "content": reply})
    state["messages"] = history
    return state


# ---------- Crisis bot ----------

def crisis_node(state: ConversationState) -> ConversationState:
    """
    Crisis path: immediate safety-focused response.
    Does NOT try to do DBT coaching.
    """
    system_prompt = CRISIS_SYSTEM_PROMPT
    return _run_leaf_bot(state, system_prompt)


# ---------- FAQ bot ----------

def faq_node(state: ConversationState) -> ConversationState:
    """
    TB FAQ bot: practical TB questions (meds, side effects, contagion, etc.)
    Uses:
      - GLOBAL_SYSTEM_PROMPT
      - FAQ_SYSTEM_PROMPT
      - TB_FAQ_KB
    """
    system_prompt = compose_system_prompt(
        [
            GLOBAL_SYSTEM_PROMPT,
            FAQ_SYSTEM_PROMPT,
            f"TB FAQ reference (mini-KB):\n{TB_FAQ_KB}",
        ]
    )
    return _run_leaf_bot(state, system_prompt)


# ---------- DBT bot ----------

def dbt_node(state: ConversationState) -> ConversationState:
    """
    DBT coaching bot: brief skills support.
    Uses:
      - GLOBAL_SYSTEM_PROMPT
      - DBT_SYSTEM_PROMPT (encodes the formulate->type->module->skill sequence)
      - DBT_SKILLS_KB
    """
    system_prompt = compose_system_prompt(
        [
            GLOBAL_SYSTEM_PROMPT,
            DBT_SYSTEM_PROMPT,
            f"DBT skills mini-KB:\n{DBT_SKILLS_KB}",
        ]
    )
    return _run_leaf_bot(state, system_prompt)


# ---------- Psychoeducation bot ----------

def psycho_node(state: ConversationState) -> ConversationState:
    """
    Psychoeducation bot: explanation, normalization, simple guidance
    about TB and mental health interactions, without going deep into
    DBT coaching.
    Uses:
      - GLOBAL_SYSTEM_PROMPT
      - PSYCHO_SYSTEM_PROMPT
    """
    system_prompt = compose_system_prompt(
        [
            GLOBAL_SYSTEM_PROMPT,
            PSYCHO_SYSTEM_PROMPT,
        ]
    )
    return _run_leaf_bot(state, system_prompt)

