# agents/faq_agent.py
from typing import Dict, List, TypedDict, Any

from langchain_core.messages import AnyMessage

from config import bedrock_text
from resources_loader import load_tb_faq_kb, load_system_prompts

PROMPTS = load_system_prompts()
FAQ_KB = load_tb_faq_kb()


class FAQItem(TypedDict):
    id: str
    question: str
    answer: str
    keywords: List[str]


def _last_user_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages or []):
        if getattr(m, "type", None) == "human":
            return m.content
    return ""


def _lexical_score(user_text: str, item: FAQItem) -> int:
    t = user_text.lower()
    score = 0
    for kw in item.get("keywords", []):
        if kw.lower() in t:
            score += 2
    question_tokens = item["question"].lower().split()
    score += sum(1 for tok in question_tokens if tok in t)
    return score


def _retrieve_best_faq(user_text: str) -> FAQItem | None:
    best_item: FAQItem | None = None
    best_score = 0
    for raw in FAQ_KB:
        item: FAQItem = raw  # type: ignore[assignment]
        score = _lexical_score(user_text, item)
        if score > best_score:
            best_score = score
            best_item = item
    return best_item if best_score > 0 else None


def faq_agent_node(state: Dict) -> Dict:
    messages: list[AnyMessage] = state.get("messages", [])
    user_text = _last_user_text(messages)

    system_prompt = PROMPTS["faq_system"]
    kb_item = _retrieve_best_faq(user_text)

    if kb_item:
        user_prompt = PROMPTS["faq_with_kb_user_template"].format(
            user_text=user_text,
            kb_question=kb_item["question"],
            kb_answer=kb_item["answer"],
        )
        kb_meta: Dict[str, Any] = {"kb_id": kb_item["id"], "kb_question": kb_item["question"]}
    else:
        user_prompt = PROMPTS["faq_no_kb_user_template"].format(user_text=user_text)
        kb_meta = {}

    reply = bedrock_text(
        [
            {"type": "system", "content": system_prompt},
            {"type": "human", "content": user_prompt},
        ],
        max_tokens=260,
        temperature=0.2,
    )

    return {
        "messages": [{"type": "ai", "content": reply}],
        "route": "faq",
        **kb_meta,
    }

