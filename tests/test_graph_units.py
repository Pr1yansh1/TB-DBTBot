"""
Pure-function unit tests for graph.py and chainlit_app.py helpers.

No Bedrock, no RAG, no graph invocation — these functions are deterministic
given their inputs. Tests focus on edge cases that could silently produce
wrong behavior (wrong route, silent crash, bad state coercion).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# _looks_truncated — drives auto-continuation logic
# ---------------------------------------------------------------------------

class TestLooksTruncated:
    """
    _looks_truncated determines whether the model was cut off mid-reply and a
    continuation prompt should fire. Wrong results here mean either:
      - False negative: user sees a truncated response with no completion attempt
      - False positive: model gets a spurious "continue" prompt, adding noise
    """

    def _check(self, text: str) -> bool:
        from tbtst_bot.graph import _looks_truncated
        return _looks_truncated(text)

    def test_empty_string_is_not_truncated(self):
        assert self._check("") is False

    def test_whitespace_only_is_not_truncated(self):
        assert self._check("   \n  ") is False

    def test_sentence_ending_with_period_is_not_truncated(self):
        assert self._check("Esta es una respuesta completa.") is False

    def test_sentence_ending_with_question_mark_is_not_truncated(self):
        assert self._check("¿Cómo te sientes hoy?") is False

    def test_sentence_ending_with_exclamation_is_not_truncated(self):
        assert self._check("¡Muy bien!") is False

    def test_ends_with_alnum_looks_truncated(self):
        # Common Bedrock cutoff signature: response ends mid-word
        assert self._check("La tuberculosis se trata con") is True

    def test_ends_with_opener_looks_truncated(self):
        assert self._check("¿Có") is True  # exact pattern match
        assert self._check("Antes de continuar (") is True
        assert self._check("Hay varios síntomas [") is True

    def test_ends_with_ellipsis_looks_truncated(self):
        assert self._check("Tengo una pregunta…") is True


# ---------------------------------------------------------------------------
# _parse_classifier_json — the JSON→State translation layer
# ---------------------------------------------------------------------------

class TestParseClassifierJson:
    """
    _parse_classifier_json is the boundary between LLM output and typed state.
    Failures here cause routing errors that are hard to trace — a wrong route
    looks like a logic issue, not a parse issue.
    """

    def _parse(self, raw: str):
        from tbtst_bot.graph import _parse_classifier_json
        return _parse_classifier_json(raw)

    def test_valid_minimal_json_parses_correctly(self):
        raw = '{"safety_risk_level":"none","safety_triggers":[],"has_protective":false,"route":"faq"}'
        result, ok = self._parse(raw)
        assert ok is True
        assert result["route"] == "faq"
        assert result["safety_risk_level"] == "none"
        assert result["tb_topic"] == "general"  # default when absent

    def test_tb_topic_latent_is_preserved(self):
        raw = '{"safety_risk_level":"none","safety_triggers":[],"has_protective":false,"route":"faq","tb_topic":"latent"}'
        result, ok = self._parse(raw)
        assert ok is True
        assert result["tb_topic"] == "latent"

    def test_completely_invalid_json_returns_safe_fallback(self):
        result, ok = self._parse("NOT JSON")
        assert ok is False
        # Fallback must be safe defaults — never a crisis-triggering value
        assert result["safety_risk_level"] == "none"
        assert result["route"] == "misc"
        assert result["safety_triggers"] == []

    def test_invalid_route_enum_returns_fallback(self):
        # "psychoed" was a route in an older schema — it no longer exists
        raw = '{"safety_risk_level":"none","safety_triggers":[],"has_protective":false,"route":"psychoed"}'
        result, ok = self._parse(raw)
        assert ok is False
        assert result["route"] == "misc"

    def test_invalid_risk_enum_returns_fallback(self):
        raw = '{"safety_risk_level":"extremely_bad","safety_triggers":[],"has_protective":false,"route":"dbt"}'
        result, ok = self._parse(raw)
        assert ok is False
        # Must not propagate the invalid value
        assert result["safety_risk_level"] == "none"

    def test_missing_required_key_returns_fallback(self):
        # Missing "route" field
        raw = '{"safety_risk_level":"none","safety_triggers":[],"has_protective":false}'
        result, ok = self._parse(raw)
        assert ok is False

    def test_extra_keys_are_tolerated(self):
        # LLM sometimes adds an "explanation" field — should not break parsing
        raw = '{"safety_risk_level":"none","safety_triggers":[],"has_protective":false,"route":"dbt","explanation":"reasons"}'
        result, ok = self._parse(raw)
        assert ok is True
        assert result["route"] == "dbt"

    def test_safety_triggers_are_capped_at_five(self):
        import json as _json
        triggers = ["a", "b", "c", "d", "e", "f", "g"]
        raw = _json.dumps({"safety_risk_level": "passive", "safety_triggers": triggers, "has_protective": False, "route": "misc"})
        result, ok = self._parse(raw)
        assert ok is True
        assert len(result["safety_triggers"]) <= 5

    def test_empty_string_returns_fallback(self):
        result, ok = self._parse("")
        assert ok is False

    def test_json_array_instead_of_object_returns_fallback(self):
        result, ok = self._parse('["none", [], false, "faq"]')
        assert ok is False


# ---------------------------------------------------------------------------
# _apply_dbt_hard_overrides — prevents skill offers after a user decline
# ---------------------------------------------------------------------------

class TestApplyDBTHardOverrides:
    """
    When a user declines or pauses a DBT skill, the router must be prevented
    from immediately offering another skill. These overrides enforce that contract
    regardless of what the LLM router returns.

    Without these guards, a user saying "no, just listen" would get another
    skill offer on the very next turn.
    """

    def _override(self, state_dict: Dict, router_mode: str, router_continuity: str) -> Tuple[str, str]:
        from tbtst_bot.graph import _apply_dbt_hard_overrides, State
        state = State(**state_dict)
        return _apply_dbt_hard_overrides(state, router_mode, router_continuity)

    def test_declined_forces_connect_mode_regardless_of_router(self):
        state = {"dbt_skill_status": "declined", "dbt_active_skill": "TIP"}
        mode, cont = self._override(state, "offer", "new")
        assert mode == "connect", "declined status must force mode=connect"
        assert cont == "same"

    def test_paused_forces_connect_mode(self):
        state = {"dbt_skill_status": "paused", "dbt_active_skill": "STOP"}
        mode, cont = self._override(state, "coach", "same")
        assert mode == "connect"

    def test_accepted_does_not_override(self):
        state = {"dbt_skill_status": "accepted", "dbt_active_skill": "STOP"}
        mode, cont = self._override(state, "coach", "same")
        assert mode == "coach"
        assert cont == "same"

    def test_none_status_does_not_override(self):
        state = {"dbt_skill_status": "none"}
        mode, cont = self._override(state, "offer", "new")
        assert mode == "offer"
        assert cont == "new"


# ---------------------------------------------------------------------------
# _coerce_agent_skill_state — keeps skill tracking coherent across turns
# ---------------------------------------------------------------------------

class TestCoerceAgentSkillState:
    """
    The DBT agent returns a (skill, status) pair each turn. _coerce_agent_skill_state
    enforces invariants the LLM agent may violate:
      - connect mode must not update skill state
      - declined/paused can be exited only by explicit user acceptance
      - coach mode coerces ambiguous status to "accepted"
    """

    def _coerce(self, **kwargs) -> Tuple[str, str]:
        from tbtst_bot.graph import _coerce_agent_skill_state
        return _coerce_agent_skill_state(**kwargs)

    def test_connect_mode_never_updates_skill(self):
        """In connect mode the agent should not be updating skill state."""
        skill, status = self._coerce(
            prev_active_skill="STOP",
            prev_skill_status="offered",
            mode="connect",
            agent_skill="DEAR MAN",
            agent_status="offered",
        )
        assert skill == "STOP"
        assert status == "offered"

    def test_offer_mode_records_decline(self):
        """If user says no during an offer, the decline must be recorded."""
        skill, status = self._coerce(
            prev_active_skill="",
            prev_skill_status="none",
            mode="offer",
            agent_skill="TIP",
            agent_status="declined",
        )
        assert status == "declined"

    def test_offer_mode_sets_offered_status(self):
        skill, status = self._coerce(
            prev_active_skill="",
            prev_skill_status="none",
            mode="offer",
            agent_skill="TIP",
            agent_status="offered",
        )
        assert skill == "TIP"
        assert status == "offered"

    def test_coach_mode_with_ambiguous_status_coerces_to_accepted(self):
        """
        LLM agent sometimes returns skill_status="none" during coaching.
        The coercer must treat this as "accepted" — the skill is ongoing.
        """
        skill, status = self._coerce(
            prev_active_skill="STOP",
            prev_skill_status="accepted",
            mode="coach",
            agent_skill="STOP",
            agent_status="none",  # ambiguous — agent forgot to set it
        )
        assert status == "accepted"

    def test_declined_state_can_be_exited_by_explicit_acceptance(self):
        """
        After declining, if the user later explicitly accepts a skill, the
        declined state should clear. Otherwise the user is permanently locked
        out of skill coaching.
        """
        skill, status = self._coerce(
            prev_active_skill="TIP",
            prev_skill_status="declined",
            mode="coach",
            agent_skill="STOP",
            agent_status="accepted",
        )
        assert status == "accepted", (
            "A user who previously declined but now accepts must exit the declined state."
        )

    def test_declined_state_is_preserved_for_non_acceptance(self):
        """declined + agent_status=offered should NOT clear declined."""
        skill, status = self._coerce(
            prev_active_skill="TIP",
            prev_skill_status="declined",
            mode="offer",
            agent_skill="STOP",
            agent_status="offered",
        )
        assert status == "declined"


# ---------------------------------------------------------------------------
# _select_recent_messages — token-budget-aware context selection
# ---------------------------------------------------------------------------

class TestSelectRecentMessages:
    """
    _select_recent_messages selects the tail of the message list within a token
    budget. Subtle failures here cause the LLM to lose conversation context or
    (worse) exceed the context window and fail.
    """

    def _select(self, messages, *, max_tokens: int, min_messages: int = 2) -> List:
        from tbtst_bot.graph import _select_recent_messages
        return _select_recent_messages(messages, max_tokens=max_tokens, min_messages=min_messages)

    def test_empty_list_returns_empty(self):
        assert self._select([], max_tokens=1000) == []

    def test_all_messages_fit_in_budget_returns_all(self):
        msgs = [HumanMessage(content="Hi"), AIMessage(content="Hello.")]
        result = self._select(msgs, max_tokens=10000)
        assert len(result) == 2

    def test_min_messages_respected_even_over_budget(self):
        """
        If min_messages=4 but budget is tiny, we still return at least 4 messages.
        Coherence matters more than strict budget when the floor isn't met.
        """
        msgs = [HumanMessage(content="msg " * 50) for _ in range(6)]
        result = self._select(msgs, max_tokens=5, min_messages=4)
        assert len(result) >= 4

    def test_returns_tail_not_head(self):
        """Recent messages matter more than old ones — the tail must be preserved."""
        msgs = [
            HumanMessage(content="very old message"),
            AIMessage(content="old reply."),
            HumanMessage(content="recent message"),
            AIMessage(content="recent reply."),
        ]
        # Budget fits only 2 messages
        result = self._select(msgs, max_tokens=30, min_messages=2)
        contents = [m.content for m in result]
        assert any("recent" in c for c in contents), (
            "Recent tail messages must be preferred over old ones."
        )


# ---------------------------------------------------------------------------
# parse_persona_txt (chainlit_app.py) — persona file parsing
# ---------------------------------------------------------------------------

class TestParsePersonaTxt:
    """
    parse_persona_txt parses the persona .txt files that define the onboarding
    profile shown to users. Parsing bugs here produce garbled persona banners.
    """

    def _parse(self, text: str, fallback: str = "Default") -> Tuple[str, str]:
        from chainlit_app import parse_persona_txt
        return parse_persona_txt(text, fallback)

    def test_first_nonempty_line_becomes_title(self):
        title, _ = self._parse("\n\nCarla\n\nDescription:\nsome details")
        assert title == "Carla"

    def test_section_headers_become_markdown_h3(self):
        _, body = self._parse("Carla\n\nDescription:\nsome text")
        assert "### Description" in body

    def test_fallback_name_used_when_file_is_blank(self):
        title, body = self._parse("", fallback="TestPersona")
        assert title == "TestPersona"

    def test_body_does_not_include_title_line(self):
        title, body = self._parse("Daniel\n\nDescription:\nRural patient.")
        assert "Daniel" not in body.split("\n")[0]

    def test_windows_line_endings_normalised(self):
        title, body = self._parse("Isabel\r\n\r\nDescription:\r\nPatient details.")
        assert title == "Isabel"
        assert "### Description" in body


# ---------------------------------------------------------------------------
# _build_async_conninfo (chainlit_app.py) — DB connection string resolution
# ---------------------------------------------------------------------------

class TestBuildAsyncConninfo:
    """
    _build_async_conninfo determines which database the Chainlit data layer
    connects to. Errors here break session persistence silently — the app
    starts fine but on_chat_resume never fires.
    """

    def _build(self, env: Dict[str, str]) -> str:
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, env, clear=False):
            # Re-import to pick up fresh env
            import importlib
            import chainlit_app
            importlib.reload(chainlit_app)
            return chainlit_app._build_async_conninfo()

    def test_explicit_override_wins(self):
        from unittest.mock import patch
        import os
        with patch.dict(os.environ, {"TBTST_CHAINLIT_DB": "postgresql+asyncpg://user:pw@host/db"}, clear=False):
            from chainlit_app import _build_async_conninfo
            result = _build_async_conninfo()
        assert result == "postgresql+asyncpg://user:pw@host/db"

    def test_database_url_postgres_gets_asyncpg_driver(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"TBTST_CHAINLIT_DB": "", "DATABASE_URL": "postgresql://user:pw@host/db"}, clear=False):
            from chainlit_app import _build_async_conninfo
            result = _build_async_conninfo()
        assert result.startswith("postgresql+asyncpg://")

    def test_database_url_old_postgres_prefix_gets_asyncpg_driver(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"TBTST_CHAINLIT_DB": "", "DATABASE_URL": "postgres://user:pw@host/db"}, clear=False):
            from chainlit_app import _build_async_conninfo
            result = _build_async_conninfo()
        assert result.startswith("postgresql+asyncpg://")

    def test_no_env_vars_returns_sqlite_default(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"TBTST_CHAINLIT_DB": "", "DATABASE_URL": ""}, clear=False):
            from chainlit_app import _build_async_conninfo
            result = _build_async_conninfo()
        assert "sqlite+aiosqlite" in result
        assert "chainlit.db" in result

    def test_default_sqlite_path_does_not_couple_to_checkpoint_db_envvar(self):
        """
        The Chainlit DB path must be independent of TBTST_CHECKPOINT_DB.
        The old code derived the Chainlit DB location from the checkpointer
        path env var — coupling two unrelated systems.
        """
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {
            "TBTST_CHAINLIT_DB": "",
            "DATABASE_URL": "",
            "TBTST_CHECKPOINT_DB": "/some/other/path/langgraph.db",
        }, clear=False):
            from chainlit_app import _build_async_conninfo
            result = _build_async_conninfo()

        # Result must not be derived from TBTST_CHECKPOINT_DB
        assert "/some/other/path/" not in result, (
            "Chainlit DB path must not be derived from TBTST_CHECKPOINT_DB. "
            "These are independent systems."
        )
