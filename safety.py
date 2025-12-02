# safety.py
import json
from typing import Dict, List, Literal, Any

from config import bedrock_chat
from resources_loader import load_safety_keywords, load_system_prompts

RiskLevel = Literal["none", "passive", "active_no_plan", "active_with_plan", "uncertain"]


KW = load_safety_keywords()
PROMPTS = load_system_prompts()


def _heuristic_assess(text: str) -> Dict[str, Any]:
    """
    Rule-based pass inspired by clinical screeners; final labels are *not*
    diagnoses, just routing hints for the chatbot.
    """
    t = text.lower()

    crisis_hard = [kw for kw in KW.get("crisis_hard", []) if kw in t]
    passive = [kw for kw in KW.get("passive_ideation", []) if kw in t]
    self_harm = [kw for kw in KW.get("self_harm_non_suicidal", []) if kw in t]
    intent = [kw for kw in KW.get("intent_markers", []) if kw in t]
    protective = [kw for kw in KW.get("protective_factors", []) if kw in t]

    level: RiskLevel = "none"
    triggers: List[str] = []

    if crisis_hard:
        level = "active_no_plan"
        triggers.extend(crisis_hard)

    if passive and level == "none":
        level = "passive"
        triggers.extend(passive)

    if self_harm and level == "none":
        level = "passive"
        triggers.extend(self_harm)

    if intent and level.startswith("active"):
        level = "active_with_plan"
        triggers.extend(intent)

    if (crisis_hard or passive or self_harm) and not intent and not protective:
        # ambiguous, treat conservatively
        if level == "none":
            level = "uncertain"

    return {
        "risk_level": level,
        "triggers": list(set(triggers)),
        "has_protective": bool(protective),
    }


def _llm_assess(text: str) -> Dict[str, Any]:
    """
    Backup LLM classifier, used when heuristics are ambiguous.
    Prompts come from `system_prompts.json`.
    Expected LLM output: a JSON object with at least `risk_level`.
    """
    system_prompt = PROMPTS["safety_classifier_system"]
    user_template = PROMPTS["safety_classifier_user_template"]
    user_msg = user_template.format(user_text=text)

    raw = bedrock_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=200,
        temperature=0.0,
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # fall back conservatively if LLM doesn't respond with JSON
        return {
            "risk_level": "uncertain",
            "triggers": [],
            "has_protective": False,
            "llm_raw": raw,
        }

    # normalize result
    level = parsed.get("risk_level", "uncertain")
    if level not in ["none", "passive", "active_no_plan", "active_with_plan", "uncertain"]:
        level = "uncertain"

    return {
        "risk_level": level,
        "triggers": parsed.get("triggers", []),
        "has_protective": bool(parsed.get("has_protective", False)),
        "llm_raw": raw,
    }


def safety_gate_node(state: Dict) -> Dict:
    """
    Safety node for LangGraph:

    - Always run heuristic pass.
    - If heuristic returns 'uncertain', back off to an LLM classifier.
    - Route:
        * 'crisis' if LLM/heuristics say passive/active/uncertain
        * 'ok' otherwise.
    """
    messages: List[Dict[str, str]] = state["messages"]
    user_text = messages[-1]["content"]

    heuristic = _heuristic_assess(user_text)
    level: RiskLevel = heuristic["risk_level"]
    triggers = heuristic["triggers"]
    has_protective = heuristic["has_protective"]
    llm_info: Dict[str, Any] = {}

    if level == "uncertain":
        llm_result = _llm_assess(user_text)
        level = llm_result["risk_level"]  # may still be 'uncertain'
        # merge triggers conservatively
        if llm_result.get("triggers"):
            triggers = list(set(triggers + llm_result["triggers"]))
        has_protective = has_protective or llm_result.get("has_protective", False)
        llm_info = {"safety_llm_raw": llm_result.get("llm_raw")}

    safety_route: Literal["ok", "crisis"]
    if level in ("active_no_plan", "active_with_plan", "passive", "uncertain"):
        safety_route = "crisis"
    else:
        safety_route = "ok"

    return {
        **state,
        "risk_level": level,
        "risk_triggers": triggers,
        "has_protective": has_protective,
        "safety_route": safety_route,
        **llm_info,
    }

