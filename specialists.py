# specialists.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional, List
from langchain_core.tools import tool

from config import model
from prompt_loader import load_prompt
from resources_loader import load_tb_faq_kb, load_dbt_skills_kb

FAQ_KB = load_tb_faq_kb()
DBT_KB = load_dbt_skills_kb()

def _retrieve_best_faq(user_text: str) -> Optional[Dict[str, Any]]:
    t = (user_text or "").lower()
    best = None
    best_score = 0
    for item in FAQ_KB:
        score = 0
        for kw in item.get("keywords", []):
            if kw.lower() in t:
                score += 2
        score += sum(1 for tok in item.get("question", "").lower().split() if tok in t)
        if score > best_score:
            best_score = score
            best = item
    return best if best_score > 0 else None

@tool
def faq_tool(request: str) -> str:
    """
    TB factual / medication / side effects / routine questions.
    Must not say "FAQ says..." or reveal internal KB usage.
    """
    system = load_prompt("faq_system")
    kb_item = _retrieve_best_faq(request)

    if kb_item:
        user_prompt = load_prompt("faq_with_kb_user").format(
            user_text=request,
            kb_question=kb_item.get("question", ""),
            kb_answer=kb_item.get("answer", ""),
        )
    else:
        user_prompt = load_prompt("faq_no_kb_user").format(user_text=request)

    resp = model.invoke([("system", system), ("user", user_prompt)])
    return resp.content or ""

@tool
def dbt_tool(request: str) -> str:
    """
    Emotional support / coping strategies (DBT-ish).
    """
    system = load_prompt("dbt_system")

    # You can make this more sophisticated later; keep simple now.
    skills_json = json.dumps(DBT_KB, ensure_ascii=False)

    user_prompt = load_prompt("dbt_user").format(
        user_text=request,
        skills_json=skills_json,
    )
    resp = model.invoke([("system", system), ("user", user_prompt)])
    return resp.content or ""

@tool
def psychoed_tool(request: str) -> str:
    """
    Psychoeducation: explain TB concepts, what to expect, support resources.
    """
    system = load_prompt("psychoed_system")
    user_prompt = load_prompt("psychoed_user").format(user_text=request)
    resp = model.invoke([("system", system), ("user", user_prompt)])
    return resp.content or ""

