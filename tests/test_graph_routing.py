from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def test_routes_to_faq_when_classifier_says_faq(fake_bedrock):
    from tbtst_bot.graph import GRAPH

    fake_bedrock.set_classifier(
        {
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "route": "faq",
        }
    )

    out: Dict[str, Any] = GRAPH.invoke(
        {"messages": [HumanMessage(content="What are common TB side effects?")]},
        config={"configurable": {"thread_id": "t_faq_1"}},
    )

    assert out["route"] == "faq"
    assert out["safety_risk_level"] == "none"
    assert out["safety_route"] == "ok"
    assert out["route_source"] == "llm"
    assert isinstance(out["messages"][-1], AIMessage)
    assert out["messages"][-1].content == "FAQ_ANSWER"

    # classify + faq = 2 calls
    assert len(fake_bedrock.calls) == 2
    assert isinstance(fake_bedrock.calls[0].messages[0], SystemMessage)
    assert fake_bedrock.calls[0].messages[0].content == "CLASSIFY"
    assert fake_bedrock.calls[1].messages[0].content == "FAQ"


def test_routes_to_dbt_when_classifier_says_dbt(fake_bedrock):
    from tbtst_bot.graph import GRAPH

    fake_bedrock.set_classifier(
        {
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "route": "dbt",
        }
    )

    out = GRAPH.invoke(
        {"messages": [HumanMessage(content="I feel overwhelmed and ashamed.")]},
        config={"configurable": {"thread_id": "t_dbt_1"}},
    )

    assert out["route"] == "dbt"
    assert out["safety_route"] == "ok"
    assert out["messages"][-1].content == "DBT_ANSWER"
    assert len(fake_bedrock.calls) == 2
    assert fake_bedrock.calls[1].messages[0].content == "DBT"


def test_crisis_overrides_route_and_uses_static_message(fake_bedrock):
    from tbtst_bot.graph import GRAPH

    fake_bedrock.set_classifier(
        {
            "safety_risk_level": "passive",
            "safety_triggers": ["don't want to live"],
            "has_protective": False,
            "route": "faq",
        }
    )

    out = GRAPH.invoke(
        {"messages": [HumanMessage(content="I don't want to live anymore.")]},
        config={"configurable": {"thread_id": "t_crisis_1"}},
    )

    assert out["safety_route"] == "crisis"
    assert out["safety_risk_level"] == "passive"
    assert out["messages"][-1].content == "CRISIS_STATIC_MESSAGE"

    # Only the classifier should have called the LLM; crisis node is static
    assert len(fake_bedrock.calls) == 1
    assert fake_bedrock.calls[0].messages[0].content == "CLASSIFY"


def test_psychoed_is_mapped_to_faq(fake_bedrock):
    from tbtst_bot.graph import GRAPH

    fake_bedrock.set_classifier(
        {
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "route": "psychoed",
        }
    )

    out = GRAPH.invoke(
        {"messages": [HumanMessage(content="How does TB spread?")]},
        config={"configurable": {"thread_id": "t_psy_1"}},
    )

    # graph.py maps psychoed -> faq in _parse_classifier_json
    assert out["route"] == "faq"
    assert out["messages"][-1].content == "FAQ_ANSWER"


def test_malformed_classifier_json_falls_back_to_uncertain_and_crisis(fake_bedrock):
    from tbtst_bot.graph import GRAPH

    fake_bedrock.set_classifier_raw("NOT JSON AT ALL")

    out = GRAPH.invoke(
        {"messages": [HumanMessage(content="hello")]},
        config={"configurable": {"thread_id": "t_badjson_1"}},
    )

    # _parse_classifier_json fallback sets risk="uncertain" and route="dbt"
    # classify_node treats "uncertain" as crisis route
    assert out["safety_risk_level"] == "uncertain"
    assert out["safety_route"] == "crisis"
    assert out["messages"][-1].content == "CRISIS_STATIC_MESSAGE"

    # Again: only classifier call should happen
    assert len(fake_bedrock.calls) == 1


def test_empty_input_defaults_to_faq_without_classifier_call(fake_bedrock):
    """
    This matches classify_node's early return for blank input.
    It should not call Bedrock for classification, but will still hit FAQ node.
    """
    from tbtst_bot.graph import GRAPH

    out = GRAPH.invoke(
        {"messages": [HumanMessage(content="   ")]},
        config={"configurable": {"thread_id": "t_empty_1"}},
    )

    assert out["route"] == "faq"
    assert out["route_source"] == "empty_input_default"
    assert out["safety_route"] == "ok"
    assert out["messages"][-1].content == "FAQ_ANSWER"

    # Only FAQ call, no classifier call
    assert len(fake_bedrock.calls) == 1
    assert fake_bedrock.calls[0].messages[0].content == "FAQ"


def test_memory_keeps_history_across_turns_and_specialist_sees_it(fake_bedrock):
    """
    Your graph is compiled with a MemorySaver checkpointer.
    With the same thread_id, the 2nd turn should include turn-1 messages in history,
    and the specialist node should receive that full history.
    """
    from tbtst_bot.graph import GRAPH

    thread_id = "t_mem_1"

    # Turn 1 -> DBT
    fake_bedrock.set_classifier(
        {"safety_risk_level": "none", "safety_triggers": [], "has_protective": False, "route": "dbt"}
    )
    out1 = GRAPH.invoke(
        {"messages": [HumanMessage(content="I'm feeling really overwhelmed today.")]},
        config={"configurable": {"thread_id": thread_id}},
    )
    assert out1["messages"][-1].content == "DBT_ANSWER"

    # Turn 2 -> FAQ
    fake_bedrock.set_classifier(
        {"safety_risk_level": "none", "safety_triggers": [], "has_protective": False, "route": "faq"}
    )
    out2 = GRAPH.invoke(
        {"messages": [HumanMessage(content="Also, is orange urine normal on rifampin?")]},
        config={"configurable": {"thread_id": thread_id}},
    )
    assert out2["messages"][-1].content == "FAQ_ANSWER"

    # Find the last FAQ call and verify it received BOTH turns of history.
    faq_call = None
    for call in reversed(fake_bedrock.calls):
        if call.messages and isinstance(call.messages[0], SystemMessage) and call.messages[0].content == "FAQ":
            faq_call = call
            break
    assert faq_call is not None

    # FAQ call messages = [System("FAQ"), ...history...]
    # Expect at least: Human(turn1), AI(turn1), Human(turn2)
    contents = [m.content for m in faq_call.messages if hasattr(m, "content")]
    assert any("I'm feeling really overwhelmed today." in c for c in contents)
    assert any("DBT_ANSWER" in c for c in contents)
    assert any("orange urine" in c for c in contents)

