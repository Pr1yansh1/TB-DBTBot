# agents/faq_agent.py
from typing import Dict, List, TypedDict, Any

from config import bedrock_chat
from resources_loader import load_tb_faq_kb, load_system_prompts


PROMPTS = load_system_prompts()
FAQ_KB = load_tb_faq_kb()


class FAQItem(TypedDict):
    id: str
    question: str
    answer: str
    keywords: List[str]


def _lexical_score(user_text: str, item: FAQItem) -> int:
    t = user_text.lower()
    score = 0
    for kw in item.get("keywords", []):
        if kw.lower() in t:
            score += 2
    # mild bonus if question tokens appear
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
    # If nothing scores, we treat as no-match and let LLM handle more freely
    return best_item if best_score > 0 else None

def faq_agent_node(state: Dict) -> Dict:
    """
    FAQ agent:

    - Uses lexical retrieval over TB FAQ mini-KB.
    - Always uses an LLM to generate the final answer:
        * If KB match exists: LLM rewrites / adapts that answer.
        * Otherwise: LLM answers directly in a cautious FAQ style.

    IMPORTANT (LangGraph best practice):
    - Do not mutate state["messages"].
    - Return only the delta message; add_messages will append it.
    """
    messages: List[Dict[str, str]] = state.get("messages", [])

    # Latest user text robustly
    user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_text = m.get("content", "")
            break

    system_prompt = PROMPTS["faq_system"]
    kb_item = _retrieve_best_faq(user_text)

    if kb_item:
        user_prompt = PROMPTS["faq_with_kb_user_template"].format(
            user_text=user_text,
            kb_question=kb_item["question"],
            kb_answer=kb_item["answer"],
        )
        kb_meta: Dict[str, Any] = {
            "kb_id": kb_item["id"],
            "kb_question": kb_item["question"],
        }
    else:
        user_prompt = PROMPTS["faq_no_kb_user_template"].format(user_text=user_text)
        kb_meta = {}

    reply = bedrock_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=260,
        temperature=0.2,
    )

    return {
        **state,
        "messages": [{"role": "assistant", "content": reply}],
        "route": "faq",
        **kb_meta,
    }

