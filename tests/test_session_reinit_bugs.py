"""
Session reinit crash regression tests.

Tests are organized by the user scenario that breaks, not by the suspected
implementation cause. Each test describes what a real user would observe.
They should fail today (bugs are real) and pass once correct behavior is
restored — regardless of which implementation approach is taken.

Real crash data that motivated these tests (from transcript analysis):
  Type A — threads 0d0e133e, 22b24c11, edb671a8: duplicate greeting on connect
  Type B — threads 1e3e514d, 417c9c81, 54aebaa7, 97ac78c7, c5463fc0, eca88bd5:
            re-init fires mid-conversation after idle (30s → 11h gaps)
  Pathological — thread 1e3e514d: 35 re-inits over 12 days
  Pathological — thread e97a9b78: 10 re-inits over 6 days, zero user messages
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

def _make_graph() -> Any:
    """
    Minimal LangGraph graph backed by MemorySaver — same checkpointer the
    production code uses. No Bedrock nodes needed; we only exercise
    update_state / get_state, which is what session seeding uses.
    """
    from langgraph.graph import StateGraph, MessagesState, END, START

    class MinimalState(MessagesState, total=False):
        onboarding_profile: str

    g = StateGraph(MinimalState)
    g.add_node("noop", lambda state: {})
    g.add_edge(START, "noop")
    g.add_edge("noop", END)
    return g.compile(checkpointer=MemorySaver())


def _config(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _read_profile(graph: Any, thread_id: str) -> str | None:
    snap = graph.get_state(_config(thread_id))
    return snap.values.get("onboarding_profile") if snap else None


class _FakeUserSession:
    """Thin dict stand-in for cl.user_session."""
    def __init__(self, initial: Dict[str, Any] | None = None) -> None:
        self._store: Dict[str, Any] = dict(initial or {})

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)


def _capture_message(content: str, sent: List[str]) -> AsyncMock:
    """
    Mock for cl.Message(...) that:
      .send()   → appends current content to `sent`
      .update() → overwrites last entry (Chainlit sends a "" placeholder,
                  then updates it once the reply is ready)
    """
    instance = AsyncMock()
    instance.content = content

    async def _send():
        sent.append(instance.content)

    async def _update():
        if sent:
            sent[-1] = instance.content

    instance.send = _send
    instance.update = _update
    return instance


def _make_cl_mock(session_id: str, sent: List[str]) -> MagicMock:
    """
    Build an explicit MagicMock for the chainlit module.

    Must be passed as `new=` to patch() — Chainlit's module-level __getattr__
    raises KeyError (not AttributeError) for unknown attrs, which breaks
    mock's auto-introspection when called without `new=`.
    """
    cl_mock = MagicMock()
    session = MagicMock()
    session.id = session_id
    cl_mock.context.session = session
    cl_mock.user_session = _FakeUserSession()
    cl_mock.Message.side_effect = lambda content="": _capture_message(content, sent)
    return cl_mock


def _on_chat_start_patches(graph: Any, cl_mock: MagicMock) -> tuple:
    """Return the standard patch stack for on_chat_start tests."""
    return (
        patch("chainlit_app.cl", new=cl_mock),
        patch("chainlit_app.warmup_once", new=AsyncMock()),
        patch("chainlit_app.select_graph", return_value=(graph, "mini")),
        patch("chainlit_app.log_session_event"),
    )


# ---------------------------------------------------------------------------
# Scenario 1: New user opens a chat for the first time
# ---------------------------------------------------------------------------

class TestNewUserOpensChat:
    """
    A user who has never used the app opens it.
    They should see exactly one greeting (banner + opener = 2 messages).
    """

    @pytest.mark.asyncio
    async def test_new_user_sees_exactly_one_greeting(self):
        """Baseline: fresh thread → exactly 2 messages, no more."""
        graph = _make_graph()
        sent: List[str] = []
        cl_mock = _make_cl_mock("new-user-session", sent)

        p1, p2, p3, p4 = _on_chat_start_patches(graph, cl_mock)
        with p1, p2, p3, p4:
            from chainlit_app import on_chat_start
            await on_chat_start()

        assert len(sent) == 2, (
            f"New user should see exactly 2 messages (banner + opener), got {len(sent)}."
        )

    @pytest.mark.asyncio
    async def test_two_rapid_connections_show_one_greeting_not_two(self):
        """
        Chainlit sometimes fires on_chat_start twice in rapid succession during
        WebSocket handshake — both fire before either has a chance to record
        that init already happened.

        The user should see ONE greeting. Two greetings is confusing and breaks
        the session state (the bot now has two 'opening' messages in history).

        Observed in: threads 0d0e133e, 22b24c11, edb671a8.
        Signature: duplicate banner+opener pair with <1ms between them.
        """
        graph = _make_graph()
        sent: List[str] = []
        cl_mock = _make_cl_mock("rapid-connect-session", sent)

        p1, p2, p3, p4 = _on_chat_start_patches(graph, cl_mock)
        with p1, p2, p3, p4:
            from chainlit_app import on_chat_start
            await asyncio.gather(on_chat_start(), on_chat_start())

        assert len(sent) == 2, (
            f"Two rapid connections should produce exactly one greeting (2 messages). "
            f"Got {len(sent)} — the session init ran twice."
        )


# ---------------------------------------------------------------------------
# Scenario 2: User returns to a chat they started earlier (idle reconnect)
# ---------------------------------------------------------------------------

class TestUserReturnsToExistingChat:
    """
    A user who already had a conversation comes back — by closing and reopening
    the tab, returning after an idle timeout, or navigating back to the thread.

    They should land back inside their existing conversation, not see a fresh
    greeting as if they never chatted before.

    Type B crash description from transcripts:
        "A normal conversation is underway. After a time gap with no user input,
        the system fires a full session re-init (intro message) mid-thread."
    """

    @pytest.mark.asyncio
    async def test_user_returning_after_idle_sees_no_new_greeting(self):
        """
        User had a conversation (e.g. sent several messages). They went idle for
        30 seconds to 11 hours. When they come back, Chainlit reconnects and fires
        on_chat_start on the same thread.

        Expected: they see nothing new — they're still in context.
        Today: they get a full re-init (banner + opener) as if starting fresh.

        Observed in: threads c5463fc0 (84s gap), 417c9c81 (30s), 1e3e514d (11h).
        """
        graph = _make_graph()
        thread_id = "user-returning-after-idle"

        # Simulate the prior conversation: user already has session state and messages
        graph.update_state(_config(f"{thread_id}:mini"), {
            "onboarding_profile": "Carla — business owner, TB treatment month 3",
            "messages": [HumanMessage(content="I'm having nausea from the medication")],
        })

        sent: List[str] = []
        cl_mock = _make_cl_mock(thread_id, sent)

        p1, p2, p3, p4 = _on_chat_start_patches(graph, cl_mock)
        with p1, p2, p3, p4:
            from chainlit_app import on_chat_start
            await on_chat_start()

        assert len(sent) == 0, (
            f"User returning to an existing session should not see a new greeting. "
            f"Got {len(sent)} message(s): {[m[:60] for m in sent]}"
        )

    @pytest.mark.asyncio
    async def test_user_who_was_mid_conversation_is_not_reset_by_reconnect(self):
        """
        Variant of the above: user was actively mid-conversation (recent messages
        on both sides). A reconnect should not wipe their context.

        This is the pathological case: thread 1e3e514d had 35 re-inits over 12
        days. The user sent messages like "quiero una herramienta para manejar mis
        problemas de atención" and got valid responses, but each exchange was
        followed by another re-init loop.
        """
        from langchain_core.messages import AIMessage

        graph = _make_graph()
        thread_id = "user-mid-conversation"

        graph.update_state(_config(f"{thread_id}:mini"), {
            "onboarding_profile": "Isabel — first-generation immigrant, anxiety",
            "messages": [
                HumanMessage(content="I want a tool for managing my attention problems"),
                AIMessage(content="Let's try a mindfulness exercise together..."),
                HumanMessage(content="I want to learn nonjudgment"),
            ],
        })

        sent: List[str] = []
        cl_mock = _make_cl_mock(thread_id, sent)

        p1, p2, p3, p4 = _on_chat_start_patches(graph, cl_mock)
        with p1, p2, p3, p4:
            from chainlit_app import on_chat_start
            await on_chat_start()

        assert len(sent) == 0, (
            f"Mid-conversation reconnect should not re-init the session. "
            f"Got {len(sent)} message(s)."
        )


# ---------------------------------------------------------------------------
# Scenario 3: Backend restarts while user has an active session
# ---------------------------------------------------------------------------

class TestSessionAfterBackendRestart:
    """
    When a worker process restarts (deploy, crash, memory limit, scheduled recycle),
    all in-process state is lost. The user's Chainlit thread still exists in the DB,
    but the backend has no memory of the conversation.

    Two failures compound here:
      (a) The graph's state store is in-memory, so the new process has no state.
      (b) Even if an idempotency check existed, it would find no state and re-init.

    Together these caused thread e97a9b78 to fire 10 re-inits over 6 days with
    zero user messages — the backend kept restarting and re-initializing the thread
    autonomously every 6–22 hours.
    """

    def test_user_context_is_accessible_after_backend_restarts(self):
        """
        A user's conversation context (persona profile, history) should survive
        a backend process restart. If it doesn't, the next on_chat_start call has
        no way to know the session was already initialized — and will re-init it.

        This test operates at the graph layer only: two separate graph instances
        with the same thread_id simulate two separate process lifetimes.

        Currently FAILS — MemorySaver is per-instance, not shared across processes.
        """
        thread_id = "user-persisted:mini"

        # First process lifetime: user had a session
        graph_process_1 = _make_graph()
        graph_process_1.update_state(_config(thread_id), {
            "onboarding_profile": "Sofia — Sofia's full profile content here",
        })

        # Backend restarts: new process, new graph instance, same thread_id
        graph_process_2 = _make_graph()

        recovered = _read_profile(graph_process_2, thread_id)
        assert recovered is not None, (
            "User's session profile was not accessible after a backend restart. "
            "This means any reconnect triggers a full re-init, as if the user is new. "
            "The fix requires a checkpointer that persists across process restarts."
        )

    @pytest.mark.asyncio
    async def test_user_sees_no_greeting_when_reconnecting_after_backend_restart(self):
        """
        End-to-end version: user had a session, backend restarted, user returned.
        They should land back in their conversation — not see a fresh greeting.

        This test will only pass when BOTH issues are resolved:
          - state survives across process restarts (persistent checkpointer)
          - the reconnect handler detects existing state and skips re-init

        Currently FAILS for both reasons.

        Real example: thread e97a9b78 — 10 re-inits with no user interaction,
        each caused by the backend touching the session after a restart.
        """
        thread_id = "user-post-restart"

        # First process: seed the session as if the user already onboarded
        graph_p1 = _make_graph()
        graph_p1.update_state(_config(f"{thread_id}:mini"), {
            "onboarding_profile": "Daniel — rural patient, transportation barriers",
            "messages": [HumanMessage(content="How long does TB treatment take?")],
        })

        # Restart: new graph instance — with MemorySaver this has no prior state
        graph_p2 = _make_graph()
        sent: List[str] = []
        cl_mock = _make_cl_mock(thread_id, sent)

        p1, p2, p3, p4 = _on_chat_start_patches(graph_p2, cl_mock)
        with p1, p2, p3, p4:
            from chainlit_app import on_chat_start
            await on_chat_start()

        assert len(sent) == 0, (
            f"User reconnecting after a backend restart should not see a new greeting. "
            f"Got {len(sent)} message(s) — session was fully re-initialized. "
            f"First message: {sent[0][:80] if sent else 'n/a'}"
        )


# ---------------------------------------------------------------------------
# Scenario 4: Backend error during a message — session must stay intact
# ---------------------------------------------------------------------------

class TestBackendErrorDoesNotResetSession:
    """
    When Bedrock fails while the user is mid-conversation, the user should see
    a transient error message and be able to retry. The session must not restart.

    If a Bedrock failure triggered re-initialization, users would lose their
    entire conversation context every time AWS throttles or has a blip.
    """

    @pytest.mark.asyncio
    async def test_throttling_error_shows_inline_error_not_a_fresh_greeting(self):
        """
        Bedrock returns ThrottlingException. The user should see a localized error
        message in the chat, not a full re-init (banner + opener).

        The error message should be in Spanish (the app's language) and indicate
        a temporary problem, not start a new conversation.
        """
        graph = _make_graph()
        graph.invoke = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("ThrottlingException: Too many requests")
        )

        sent: List[str] = []
        init_fired = []

        async def track_init():
            init_fired.append(True)

        user_session = _FakeUserSession({
            "graph": graph,
            "thread_id": "mid-convo-error:mini",
            "variant": "mini",
        })
        cl_mock = MagicMock()
        cl_mock.user_session = user_session
        cl_mock.Message.side_effect = lambda content="": _capture_message(content, sent)

        with patch("chainlit_app.cl", new=cl_mock), \
             patch("chainlit_app.log_session_event"), \
             patch("chainlit_app.on_chat_start", new=track_init):

            from chainlit_app import on_message
            msg = MagicMock()
            msg.content = "What are the side effects of isoniazid?"
            await on_message(msg)

        assert not init_fired, (
            f"on_chat_start should not be triggered by a Bedrock error. "
            f"Was called {len(init_fired)} time(s)."
        )
        assert sent, "User should receive an error message, not silence."
        assert any(
            "lo siento" in m.lower() or "problema" in m.lower() or "intenta" in m.lower()
            for m in sent
        ), (
            f"Error message should be in Spanish and indicate a temporary problem. "
            f"Got: {sent}"
        )
