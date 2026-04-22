"""
Graph routing and state tests.

Tests are scenario-driven: each test describes a real failure mode or
behavioral contract that could silently break if graph.py is refactored.

These are NOT "does graph.invoke() return something" tests — they verify
specific state fields, routing decisions, and cross-turn memory behavior
that the graph is contractually required to produce.

Fixture: fake_graph from conftest.py — no Bedrock or RAG calls.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from conftest import GraphFixture, invoke


# ---------------------------------------------------------------------------
# Route dispatch — does the correct node fire for each classifier output?
# ---------------------------------------------------------------------------

class TestRouteDispatch:

    def test_faq_route_reaches_tb_info_node(self, fake_graph: GraphFixture):
        """
        Classifier says route=faq → tb_info_node runs → answer in final messages.
        Also verifies tb_info structured calls fire (tb_gate + tb_info).
        """
        fake_graph.llm.set_classifier({
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "route": "faq",
            "tb_topic": "general",
        })
        out = invoke(fake_graph.graph, "What are TB side effects?", "t-faq-1")

        assert out["route"] == "faq"
        assert out["safety_route"] == "ok"
        assert out["route_source"] == "llm"
        assert isinstance(out["messages"][-1], AIMessage)
        assert fake_graph.llm.tb_answer in out["messages"][-1].content

        # tb_gate + tb_info must have been called (RAG path ran)
        structured_names = [c.name for c in fake_graph.llm.calls]
        assert "tb_gate" in structured_names
        assert "tb_info" in structured_names

    def test_dbt_route_reaches_dbt_mini_node(self, fake_graph: GraphFixture):
        """
        Classifier says route=dbt → dbt_mini_node runs → DBT answer in messages.
        """
        fake_graph.llm.set_classifier({
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "route": "dbt",
            "tb_topic": "general",
        })
        out = invoke(fake_graph.graph, "I feel overwhelmed and ashamed.", "t-dbt-1")

        assert out["route"] == "dbt"
        assert out["safety_route"] == "ok"
        assert isinstance(out["messages"][-1], AIMessage)
        assert fake_graph.llm.dbt_answer in out["messages"][-1].content

        # Classifier + dbt node = 2 calls, no tb_info calls
        assert not fake_graph.llm.calls_named("tb_gate")
        assert not fake_graph.llm.calls_named("tb_info")

    def test_misc_route_reaches_misc_node(self, fake_graph: GraphFixture):
        """
        Classifier says route=misc → misc_node runs.
        """
        fake_graph.llm.set_classifier({
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "route": "misc",
            "tb_topic": "general",
        })
        out = invoke(fake_graph.graph, "Hello!", "t-misc-1")

        assert out["route"] == "misc"
        assert isinstance(out["messages"][-1], AIMessage)
        assert fake_graph.llm.misc_answer in out["messages"][-1].content

    def test_crisis_overrides_route_uses_static_message(self, fake_graph: GraphFixture):
        """
        When risk is passive/active, safety_route=crisis and the CRISIS_TEXT static
        message fires — regardless of what `route` the classifier returned.

        Critically: only the classifier LLM call should happen. No specialist node
        should run. (crisis_node is a static string, not an LLM call.)
        """
        from tbtst_bot.graph import CRISIS_TEXT

        fake_graph.llm.set_classifier({
            "safety_risk_level": "passive",
            "safety_triggers": ["don't want to live"],
            "has_protective": False,
            "route": "faq",       # ← route doesn't matter, crisis should override
            "tb_topic": "general",
        })
        out = invoke(fake_graph.graph, "I don't want to live anymore.", "t-crisis-1")

        assert out["safety_route"] == "crisis"
        assert out["safety_risk_level"] == "passive"
        assert out["messages"][-1].content == CRISIS_TEXT

        # Only the classify call should have fired — no specialist LLM calls
        assert len(fake_graph.llm.non_classifier_calls) == 0


# ---------------------------------------------------------------------------
# Safety classification edge cases
# ---------------------------------------------------------------------------

class TestSafetyEdgeCases:

    def test_uncertain_risk_does_not_trigger_crisis(self, fake_graph: GraphFixture):
        """
        `uncertain` is NOT in the crisis-trigger set (passive/active_no_plan/active_with_plan).
        A common mistake would be to treat it like passive — don't.

        The old test suite had this wrong. Correct behavior: uncertain → safety_route=ok,
        conversation continues to the requested route node.
        """
        fake_graph.llm.set_classifier({
            "safety_risk_level": "uncertain",
            "safety_triggers": ["maybe"],
            "has_protective": False,
            "route": "misc",
            "tb_topic": "general",
        })
        out = invoke(fake_graph.graph, "I'm not sure how I feel.", "t-uncertain-1")

        assert out["safety_risk_level"] == "uncertain"
        assert out["safety_route"] == "ok", (
            "uncertain risk must NOT trigger crisis — only passive/active_no_plan/active_with_plan do."
        )
        # Misc node should have fired (not crisis)
        assert fake_graph.llm.misc_answer in out["messages"][-1].content

    def test_active_with_plan_triggers_crisis(self, fake_graph: GraphFixture):
        """
        active_with_plan is the most severe risk level — always crisis.
        """
        from tbtst_bot.graph import CRISIS_TEXT

        fake_graph.llm.set_classifier({
            "safety_risk_level": "active_with_plan",
            "safety_triggers": ["tonight", "plan"],
            "has_protective": False,
            "route": "dbt",
            "tb_topic": "general",
        })
        out = invoke(fake_graph.graph, "I have a plan for tonight.", "t-active-plan-1")

        assert out["safety_route"] == "crisis"
        assert out["messages"][-1].content == CRISIS_TEXT

    def test_malformed_classifier_json_falls_back_to_misc_not_crisis(self, fake_graph: GraphFixture):
        """
        If the classifier returns unparseable JSON (even after retry), the fallback is:
          safety_risk_level="none", route="misc", safety_route="ok", classifier_parse_ok=False

        The old test suite asserted this produced a CRISIS response — that was wrong.
        Malformed output is treated as low-risk and routed to misc, not crisis.
        """
        fake_graph.llm.set_classifier_raw("NOT JSON AT ALL :: gibberish")

        out = invoke(fake_graph.graph, "hello", "t-badjson-1")

        assert out["safety_risk_level"] == "none"
        assert out["safety_route"] == "ok"
        assert out["route"] == "misc"
        assert out["classifier_parse_ok"] is False, (
            "classifier_parse_ok should be False when JSON could not be parsed."
        )
        # Misc node must have fired (graceful degradation, not crisis)
        assert fake_graph.llm.misc_answer in out["messages"][-1].content

    def test_uncertain_with_no_triggers_retries_classifier(self, fake_graph: GraphFixture):
        """
        classify_node retries if safety_risk_level=uncertain AND safety_triggers=[].
        This means at least 2 classifier calls should have fired.
        """
        fake_graph.llm.set_classifier({
            "safety_risk_level": "uncertain",
            "safety_triggers": [],     # ← empty triggers + uncertain → forces retry
            "has_protective": False,
            "route": "misc",
            "tb_topic": "general",
        })
        invoke(fake_graph.graph, "I don't know.", "t-uncertain-retry-1")

        classifier_calls = fake_graph.llm.classifier_calls
        assert len(classifier_calls) >= 2, (
            f"uncertain + empty triggers must trigger a classifier retry. "
            f"Got {len(classifier_calls)} classify call(s)."
        )


# ---------------------------------------------------------------------------
# Empty input special case
# ---------------------------------------------------------------------------

class TestEmptyInput:

    def test_blank_input_skips_classifier_llm_call(self, fake_graph: GraphFixture):
        """
        classify_node has an early return for blank input that sets route=misc
        without calling Bedrock. This is important for telemetry (no spurious
        LLM spend on accidental empty sends).
        """
        out = invoke(fake_graph.graph, "   ", "t-empty-1")

        assert out["route"] == "misc"
        assert out["route_source"] == "empty_input_default"
        assert out["safety_route"] == "ok"

        # No classifier LLM call should have happened
        assert len(fake_graph.llm.classifier_calls) == 0, (
            "Blank input must not trigger a Bedrock classify call."
        )


# ---------------------------------------------------------------------------
# State flow — does the right data reach the right node?
# ---------------------------------------------------------------------------

class TestStateFlow:

    def test_tb_topic_latent_sets_allow_latent_on_retrieval(self, fake_graph: GraphFixture):
        """
        When the classifier signals tb_topic=latent, tb_info_node must call
        retrieve_tb_docs with allow_latent=True. This flag controls which
        document corpus is searched — getting it wrong silently returns wrong answers.
        """
        fake_graph.llm.set_classifier({
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "route": "faq",
            "tb_topic": "latent",
        })
        invoke(fake_graph.graph, "What is latent TB treatment?", "t-latent-1")

        assert fake_graph.retrieval.calls, "retrieve_tb_docs should have been called."
        args = fake_graph.retrieval.calls[-1]
        assert args.get("allow_latent") is True, (
            f"tb_topic=latent must set allow_latent=True on retrieval. Got: {args}"
        )

    def test_tb_topic_general_sets_allow_latent_false(self, fake_graph: GraphFixture):
        """
        Complement: tb_topic=general must NOT allow latent document retrieval.
        """
        fake_graph.llm.set_classifier({
            "safety_risk_level": "none",
            "safety_triggers": [],
            "has_protective": False,
            "route": "faq",
            "tb_topic": "general",
        })
        invoke(fake_graph.graph, "What are TB symptoms?", "t-general-1")

        assert fake_graph.retrieval.calls
        args = fake_graph.retrieval.calls[-1]
        assert args.get("allow_latent") is False, (
            f"tb_topic=general must set allow_latent=False. Got: {args}"
        )

    def test_onboarding_profile_persists_to_second_turn(self, fake_graph: GraphFixture):
        """
        onboarding_profile is seeded once and must survive into subsequent turns.
        If it gets wiped on turn 2, all personalization is lost and the LLM context
        becomes incoherent.

        This tests the SqliteSaver (here MemorySaver) checkpoint round-trip.
        """
        thread_id = "t-profile-persist"
        profile = "Carla — business owner, TB treatment month 3, speaks Spanish."

        # Seed the profile directly (mirrors what on_chat_start does)
        fake_graph.graph.update_state(
            {"configurable": {"thread_id": thread_id}},
            {"onboarding_profile": profile},
        )

        fake_graph.llm.set_classifier({"safety_risk_level": "none", "safety_triggers": [], "has_protective": False, "route": "misc", "tb_topic": "general"})
        invoke(fake_graph.graph, "Hello.", thread_id)

        # Turn 2: read state directly and verify profile is still there
        snap = fake_graph.graph.get_state({"configurable": {"thread_id": thread_id}})
        assert snap.values.get("onboarding_profile") == profile, (
            "onboarding_profile must persist across turns — it disappeared after turn 1."
        )

    def test_message_history_accumulates_across_turns(self, fake_graph: GraphFixture):
        """
        Each turn's messages must be appended to the thread, not replaced.
        If history doesn't accumulate, the LLM never has context from prior turns
        and cannot maintain continuity.

        Verifies the LangGraph checkpointer is wiring turns together correctly.
        """
        thread_id = "t-history"
        fake_graph.llm.set_classifier({"safety_risk_level": "none", "safety_triggers": [], "has_protective": False, "route": "misc", "tb_topic": "general"})

        invoke(fake_graph.graph, "My name is Maria.", thread_id)
        out2 = invoke(fake_graph.graph, "What did I just say?", thread_id)

        all_message_contents = [m.content for m in out2["messages"] if hasattr(m, "content")]
        assert any("Maria" in c for c in all_message_contents), (
            "Turn 1 message content must appear in turn 2 state — history is not accumulating."
        )

    def test_classifier_parse_ok_true_in_state_when_json_valid(self, fake_graph: GraphFixture):
        """
        classifier_parse_ok=True must be set when the classifier returns valid JSON.
        This field is used for monitoring/logging and its absence or wrong value
        would make parse failures invisible.
        """
        fake_graph.llm.set_classifier({"safety_risk_level": "none", "safety_triggers": [], "has_protective": False, "route": "misc", "tb_topic": "general"})
        out = invoke(fake_graph.graph, "Hello.", "t-parseok-1")

        assert out.get("classifier_parse_ok") is True

    def test_tb_topic_stored_in_state(self, fake_graph: GraphFixture):
        """
        tb_topic from the classifier must be stored in state so downstream nodes
        (and future turns) can read it.
        """
        fake_graph.llm.set_classifier({"safety_risk_level": "none", "safety_triggers": [], "has_protective": False, "route": "faq", "tb_topic": "latent"})
        out = invoke(fake_graph.graph, "Tell me about latent TB.", "t-tbtopic-state")

        assert out.get("tb_topic") == "latent"
