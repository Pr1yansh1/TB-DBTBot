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
    SAFETY_PASSIVE_PHRASES,
    SAFETY_ACTIVE_PHRASES,
    SAFETY_BEHAVIOR_PHRASES,
    SAFETY_MEANS_PHRASES,
    SAFETY_TIME_URGENCY_PHRASES,
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

_RISK_ORDER = ["none", "low", "moderate", "high", "imminent"]
_RISK_RANK = {lvl: i for i, lvl in enumerate(_RISK_ORDER)}


def _max_risk_level(a: str, b: str) -> str:
    """Return the more severe of two textual risk levels."""
    a_norm = (a or "none").lower()
    b_norm = (b or "none").lower()
    return a_norm if _RISK_RANK.get(a_norm, 0) >= _RISK_RANK.get(b_norm, 0) else b_norm


def _keyword_suicide_screen(text: str) -> Dict[str, Any]:
    """
    Keyword-based screen, inspired by:
      - Linguistic features review (intensifiers, pronouns, death refs, etc.)
      - C-SSRS distinctions between passive vs active ideation + behavior
      - PHQ-9 item 9 (thoughts of being better off dead or self-harm)

    It distinguishes:
      ideation: "none" | "passive" | "active"
      behavior: "none" | "self_harm_or_attempt"
      timeframe: "unspecified" | "imminent"
      risk_level: "none" | "low" | "moderate" | "high" | "imminent"
    """
    t = (text or "").lower()
    flags: List[str] = []

    ideation = "none"
    behavior = "none"
    timeframe = "unspecified"
    risk_level = "low"  # default for “no obvious risk language”
    keyword_hit = False

    if not t.strip():
        return {
            "keyword_hit": False,
            "risk_level": "none",
            "ideation": "none",
            "behavior": "none",
            "timeframe": "unspecified",
            "flags": [],
        }

    # Passive ideation (“better off dead”, “don’t want to be here”)
    if SAFETY_PASSIVE_PHRASES and any(p in t for p in SAFETY_PASSIVE_PHRASES):
        ideation = "passive"
        flags.append("kw_passive_ideation")
        keyword_hit = True

    # Active ideation (“kill myself”, “end my life”, etc.)
    if SAFETY_ACTIVE_PHRASES and any(p in t for p in SAFETY_ACTIVE_PHRASES):
        ideation = "active"
        flags.append("kw_active_ideation")
        keyword_hit = True

    # Explicit behavior / attempts (“cut last night”, “overdosed”, etc.)
    if SAFETY_BEHAVIOR_PHRASES and any(p in t for p in SAFETY_BEHAVIOR_PHRASES):
        behavior = "self_harm_or_attempt"
        flags.append("kw_behavior")
        keyword_hit = True

    # Mentions of means / preparation
    has_means = bool(
        SAFETY_MEANS_PHRASES and any(p in t for p in SAFETY_MEANS_PHRASES)
    )
    if has_means:
        flags.append("kw_means_access")
        keyword_hit = True

    # Time-urgency / imminence (“right now”, “tonight”, etc.)
    has_urgency = bool(
        SAFETY_TIME_URGENCY_PHRASES and any(p in t for p in SAFETY_TIME_URGENCY_PHRASES)
    )
    if has_urgency:
        timeframe = "imminent"
        flags.append("kw_time_urgency")
        keyword_hit = True

    # Explicit crisis phrases (“I am going to kill myself now”, etc.)
    if SAFETY_CRISIS_PHRASES and any(p in t for p in SAFETY_CRISIS_PHRASES):
        flags.append("kw_explicit_crisis_phrase")
        keyword_hit = True
        if ideation == "none":
            ideation = "active"

    # Intensifiers & superlatives around risk language
    has_intensifier = bool(
        SAFETY_INTENSIFIERS and any(p in t for p in SAFETY_INTENSIFIERS)
    )
    if has_intensifier:
        flags.append("kw_intensifier")
        keyword_hit = True

    # Derive risk level:
    # - Any behavior ⇒ high
    # - Active ideation ⇒ high
    # - Passive ideation ⇒ moderate
    # - Active + (intensifier or urgency or means or explicit crisis) ⇒ imminent
    if behavior != "none":
        risk_level = "high"
    elif ideation == "active":
        risk_level = "high"
    elif ideation == "passive":
        risk_level = "moderate"
    else:
        risk_level = "low" if keyword_hit else "none"

    if ideation == "active" and (has_intensifier or has_urgency or has_means or "kw_explicit_crisis_phrase" in flags):
        risk_level = "imminent"

    return {
        "keyword_hit": keyword_hit,
        "risk_level": risk_level,
        "ideation": ideation,
        "behavior": behavior,
        "timeframe": timeframe,
        "flags": flags,
    }


def safety_node(state: ConversationState) -> ConversationState:
    """
    Classify safety level for the latest user message using:
      1) Keyword-based heuristics (review + C-SSRS-inspired categories)
      2) LLM classifier (JSON output) using SAFETY_SYSTEM_PROMPT

    Expected (ideal) LLM JSON format (you can enforce this in the prompt):
    {
      "risk_level": "none|low|moderate|high|imminent",
      "ideation": "none|passive|active_no_plan|active_with_plan|active_with_intent",
      "behavior": "none|self_harm_no_suicidal_intent|self_harm_with_intent|suicide_attempt",
      "timeframe": "none|lifetime|past_year|past_month|past_48h|imminent",
      "reason": "...",
      "flags": ["phq9_item9_positive", "cssrs_passive", ...]
    }

    But we also remain backward compatible with a simpler:
    {"level": "high|low", "reason": "...", "flags": [...]}
    """
    text = _latest_user_text(state)

    # --- 1) Keyword-based heuristics ---
    kw = _keyword_suicide_screen(text)
    kw_level = kw.get("risk_level", "none")

    # --- 2) LLM classification ---
    messages = [
        {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    raw = bedrock_chat(messages, max_tokens=220, temperature=0.0)

    llm_level = "none"
    llm_ideation = "none"
    llm_behavior = "none"
    llm_timeframe = "unspecified"
    llm_reason = "no_reason"
    llm_flags: List[str] = []
    parse_error: str | None = None

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            # Prefer 'risk_level', fall back to 'level'
            llm_level = str(
                data.get("risk_level", data.get("level", "none"))
            ).lower()

            llm_ideation = str(
                data.get("ideation", data.get("ideation_type", "none"))
            ).lower()

            llm_behavior = str(
                data.get("behavior", data.get("behavior_type", "none"))
            ).lower()

            llm_timeframe = str(
                data.get("timeframe", data.get("time_window", "unspecified"))
            ).lower()

            llm_reason = str(data.get("reason", "") or "no_reason")

            flags_val = data.get("flags")
            if isinstance(flags_val, list):
                llm_flags = [str(f) for f in flags_val]
    except Exception as e:  # noqa: BLE001
        parse_error = f"{type(e).__name__}: {e}"

    # --- Combine keyword and LLM signals conservatively ---
    # If parsing failed, fall back to keywords + “low” (or higher) risk.
    if parse_error:
        combined_level = _max_risk_level(kw_level, "low")
        combined_reason = f"fallback_parse_error: {parse_error}"
    else:
        combined_level = _max_risk_level(kw_level, llm_level)
        combined_reason = llm_reason

    # For backwards compatibility, map "none" → "low" in the simple 'level' field
    level_for_state = combined_level if combined_level != "none" else "low"

    # Crisis routing threshold:
    # - high or imminent combined level
    # - OR keyword behavior mentions (attempt / self-harm)
    # - OR explicit crisis / imminent timeframe
    is_crisis = False
    if _RISK_RANK.get(combined_level, 0) >= _RISK_RANK["high"]:
        is_crisis = True
    if kw.get("behavior") != "none":
        is_crisis = True
    if kw.get("timeframe") == "imminent":
        is_crisis = True
    if "kw_explicit_crisis_phrase" in kw.get("flags", []):
        is_crisis = True

    # Merge flags
    all_flags = list({*kw.get("flags", []), *llm_flags})

    safety_info: Dict[str, Any] = {
        # Backwards-compatible fields:
        "level": level_for_state,
        "is_crisis": is_crisis,
        "reason": combined_reason,
        "flags": all_flags,
        "keyword_hit": bool(kw.get("keyword_hit")),
        "raw_classifier_output": raw,
        # Richer, research-aligned structure:
        "risk_level": combined_level,
        "keyword_assessment": kw,
        "classifier": {
            "risk_level": llm_level,
            "ideation": llm_ideation,
            "behavior": llm_behavior,
            "timeframe": llm_timeframe,
            "reason": llm_reason,
            "flags": llm_flags,
            "parse_error": parse_error,
        },
    }

    state["safety"] = safety_info

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
    state["route"] = "non_crisis"
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

