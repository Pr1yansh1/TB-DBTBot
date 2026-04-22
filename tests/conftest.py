"""
Shared test fixtures for graph.py unit and integration tests.

The `fake_graph` fixture intercepts the two LLM call points in graph.py
(_timed_invoke and _timed_invoke_structured) plus the RAG retrieval tool,
so no Bedrock or vector-store calls happen during tests.

Architecture notes:
  - All LLM calls in graph.py go through _timed_invoke or _timed_invoke_structured.
    Patching those two functions covers every node: classify, tb_info, dbt_mini,
    dbt_brain, dbt_module_agents, misc, memory_manager.
  - retrieve_tb_docs is a @tool with a .invoke() method; it's called via _timed_tool.
    Patching the reference in tbtst_bot.graph is sufficient.
  - The graph is built fresh with MemorySaver so tests don't share state or hit
    the real SQLite checkpointer file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage


# ---------------------------------------------------------------------------
# Call recorder
# ---------------------------------------------------------------------------

@dataclass
class LLMCall:
    name: str
    messages: List[BaseMessage]


# ---------------------------------------------------------------------------
# Fake LLM controller
# ---------------------------------------------------------------------------

class FakeLLM:
    """
    Intercepts _timed_invoke and _timed_invoke_structured.

    Routes responses by call `name` (the label each graph node passes as the
    first argument). Responses are deterministic and use answer strings that
    end with '.' so the auto-continue heuristic (_looks_truncated) never fires.
    """

    def __init__(self):
        self.calls: List[LLMCall] = []
        self._classifier_json: Optional[Dict[str, Any]] = None
        self._classifier_raw: Optional[str] = None

        # These are the strings nodes will produce — override in individual tests
        self.tb_answer = "La información sobre tuberculosis está aquí."
        self.dbt_answer = "Entiendo cómo te sientes."
        self.misc_answer = "Claro, ¿en qué puedo ayudarte?"

    # --- Controller API ---

    def set_classifier(self, payload: Dict[str, Any]) -> None:
        self._classifier_json = payload
        self._classifier_raw = None

    def set_classifier_raw(self, raw: str) -> None:
        self._classifier_raw = raw
        self._classifier_json = None

    @property
    def classifier_calls(self) -> List[LLMCall]:
        return [c for c in self.calls if c.name.startswith("classify")]

    @property
    def non_classifier_calls(self) -> List[LLMCall]:
        return [c for c in self.calls if not c.name.startswith("classify")]

    def calls_named(self, prefix: str) -> List[LLMCall]:
        return [c for c in self.calls if c.name.startswith(prefix)]

    # --- _timed_invoke replacement ---

    def timed_invoke(
        self,
        name: str,
        llm: Any,
        messages: List[BaseMessage],
        *,
        trace_id: str,
    ):
        from tbtst_bot.graph import CallMetrics

        self.calls.append(LLMCall(name=name, messages=list(messages)))

        if name.startswith("classify"):
            if self._classifier_raw is not None:
                content = self._classifier_raw
            else:
                payload = self._classifier_json or {
                    "safety_risk_level": "none",
                    "safety_triggers": [],
                    "has_protective": False,
                    "route": "misc",
                    "tb_topic": "general",
                }
                content = json.dumps(payload)
        elif name.startswith("dbt"):
            content = self.dbt_answer
        elif name == "misc":
            content = self.misc_answer
        elif name == "summarize":
            content = "Resumen de conversación."
        else:
            content = ""

        resp = AIMessage(content=content)
        metrics = CallMetrics(
            name=name,
            trace_id=trace_id,
            latency_ms=0.0,
            in_tokens=0,
            out_chars=len(content),
            stop_reason="end_turn",
        )
        return resp, metrics

    # --- _timed_invoke_structured replacement ---

    def timed_invoke_structured(
        self,
        name: str,
        llm_structured: Any,
        messages: List[BaseMessage],
        *,
        trace_id: str,
    ) -> Any:
        from tbtst_bot.graph import (
            DBTAgentOut,
            DBTBrainOut,
            DBTBrainSignals,
            TBAnswerJSON,
            TBGateOut,
        )

        self.calls.append(LLMCall(name=name, messages=list(messages)))

        if name == "tb_gate":
            # Default: don't ask for clarification — proceed to answer
            return TBGateOut(action="answer", clarifying_question="")

        if name == "tb_info":
            return TBAnswerJSON(answer=self.tb_answer, citations_used=[])

        if name.startswith("dbt_brain"):
            return DBTBrainOut(
                module="MIND",
                mode="connect",
                continuity="new",
                confidence=0.8,
                rationale_brief="test",
                signals=DBTBrainSignals(
                    emotion_intensity="low",
                    impulse_or_urge_present=False,
                    problem_solvable_now=False,
                    interpersonal_context=False,
                    attention_or_judgment_issue=False,
                ),
            )

        if name.startswith("dbt:"):
            # Full-graph module agents (DT/MIND/ER/IE)
            return DBTAgentOut(
                message=self.dbt_answer,
                skill="",
                skill_status="none",
            )

        return None


# ---------------------------------------------------------------------------
# Fake RAG retrieval
# ---------------------------------------------------------------------------

class FakeRetrieve:
    """
    Stand-in for the retrieve_tb_docs LangChain @tool.
    Records calls so tests can assert on allow_latent.
    """

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def invoke(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(args)
        return {"sources": []}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@dataclass
class GraphFixture:
    llm: FakeLLM
    retrieval: FakeRetrieve
    graph: Any  # compiled LangGraph graph


@pytest.fixture()
def fake_graph() -> GraphFixture:
    """
    Yields a GraphFixture with a fresh graph (MemorySaver checkpointer) and all
    Bedrock + RAG calls patched out.

    The graph is built inside the patch context so it compiles with the
    patched _CHECKPOINTER — tests run against an isolated in-memory store.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from tbtst_bot.graph import build_graph, dbt_mini_node

    llm = FakeLLM()
    retrieval = FakeRetrieve()

    with (
        patch("tbtst_bot.graph._timed_invoke", side_effect=llm.timed_invoke),
        patch("tbtst_bot.graph._timed_invoke_structured", side_effect=llm.timed_invoke_structured),
        patch("tbtst_bot.graph.retrieve_tb_docs", retrieval),
        patch("tbtst_bot.graph._CHECKPOINTER", MemorySaver()),
    ):
        graph = build_graph(dbt_node=dbt_mini_node)
        yield GraphFixture(llm=llm, retrieval=retrieval, graph=graph)


# ---------------------------------------------------------------------------
# Shared invoke helper
# ---------------------------------------------------------------------------

def invoke(graph: Any, text: str, thread_id: str) -> Dict[str, Any]:
    return graph.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread_id}},
    )


from langchain_core.messages import HumanMessage  # noqa: E402  (after dataclasses)
