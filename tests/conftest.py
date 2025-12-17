import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


@dataclass
class BedrockCall:
    messages: List[BaseMessage]
    max_tokens: int
    temperature: float


class FakeBedrock:
    """
    Deterministic stub for tbtst_bot.config.bedrock_chat.

    We detect which "kind" of call it is based on the first SystemMessage content:
      - CLASSIFY: returns JSON routing/safety
      - FAQ: returns a fixed FAQ answer
      - DBT: returns a fixed DBT answer
    """
    def __init__(self) -> None:
        self.calls: List[BedrockCall] = []
        self.next_classifier_json: Optional[Dict[str, Any]] = None
        self.next_classifier_raw: Optional[str] = None
        self.faq_answer: str = "FAQ_ANSWER"
        self.dbt_answer: str = "DBT_ANSWER"

    def set_classifier(self, payload: Dict[str, Any]) -> None:
        self.next_classifier_json = payload
        self.next_classifier_raw = None

    def set_classifier_raw(self, raw: str) -> None:
        self.next_classifier_raw = raw
        self.next_classifier_json = None

    def __call__(self, messages: List[BaseMessage], max_tokens: int = 220, temperature: float = 0.0) -> str:
        self.calls.append(BedrockCall(messages=messages, max_tokens=max_tokens, temperature=temperature))

        sys = messages[0].content if messages and isinstance(messages[0], SystemMessage) else ""

        if sys == "CLASSIFY":
            if self.next_classifier_raw is not None:
                return self.next_classifier_raw
            if self.next_classifier_json is None:
                # default classifier response if test didn't set one
                return json.dumps(
                    {
                        "safety_risk_level": "none",
                        "safety_triggers": [],
                        "has_protective": False,
                        "route": "faq",
                    }
                )
            return json.dumps(self.next_classifier_json)

        if sys == "FAQ":
            return self.faq_answer

        if sys == "DBT":
            return self.dbt_answer

        # If your code ever calls without a recognized system prompt,
        # make it obvious in tests.
        raise AssertionError(f"FakeBedrock got unexpected system prompt: {sys!r}")


@pytest.fixture()
def fake_bedrock(monkeypatch) -> FakeBedrock:
    """
    Patch the graph module's globals so:
      - No real Bedrock calls happen
      - Prompts are simple stable sentinels ("CLASSIFY", "FAQ", "DBT")
      - Crisis message is known constant
    """
    from tbtst_bot import graph as g

    fb = FakeBedrock()

    # Patch the callable used by nodes
    monkeypatch.setattr(g, "bedrock_chat", fb)

    # Patch prompts to simple sentinels so FakeBedrock can route on them
    monkeypatch.setattr(g, "CLASSIFY_SYSTEM", "CLASSIFY")
    monkeypatch.setattr(g, "CLASSIFY_USER_TMPL", 'Latest patient message:\n"""{user_text}"""\nReturn STRICT JSON only:\n{"safety_risk_level":"none|passive|active_no_plan|active_with_plan|uncertain","safety_triggers":["..."],"has_protective":true|false,"route":"faq|dbt|psychoed"}')
    monkeypatch.setattr(g, "FAQ_SYSTEM", "FAQ")
    monkeypatch.setattr(g, "DBT_SYSTEM", "DBT")
    monkeypatch.setattr(g, "CRISIS_TEXT", "CRISIS_STATIC_MESSAGE")

    return fb


def invoke(graph, text: str, thread_id: str) -> Dict[str, Any]:
    """Helper to invoke your graph with a single HumanMessage."""
    return graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread_id}},
    )

