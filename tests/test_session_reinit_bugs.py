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

Fix summary (all 8 tests pass):
  - SqliteSaver replaces MemorySaver in graph.py (state survives process restart)
  - Idempotency guard + per-thread lock in on_chat_start (Type A, Type B short-gap)
  - @cl.data_layer + @cl.header_auth_callback + @cl.on_chat_resume (Type B long-gap)
  - session.thread_id (stable conversation ID) replaces session.id in on_chat_start
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

def _build_minimal_graph(checkpointer: Any) -> Any:
    """Compile a minimal graph with the given checkpointer. No Bedrock needed."""
    from langgraph.graph import StateGraph, MessagesState, END, START

    class MinimalState(MessagesState, total=False):
        onboarding_profile: str

    g = StateGraph(MinimalState)
    g.add_node("noop", lambda state: {})
    g.add_edge(START, "noop")
    g.add_edge("noop", END)
    return g.compile(checkpointer=checkpointer)


def _make_graph() -> Any:
    """
    Graph with an isolated in-memory checkpointer.
    Used for tests that operate on a single process lifetime (no restart simulation)
    — idempotency, concurrent connects, mid-conversation reconnect.
    """
    return _build_minimal_graph(MemorySaver())


@contextmanager
def _shared_sqlite_checkpointer() -> Generator[str, None, None]:
    """
    Yield a temp-file path that two sequential graph instances can share,
    simulating a backend restart against the same persistent store.
    The file is deleted after the test.
    """
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        yield db_path
    finally:
        os.unlink(db_path)


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
      .send()   → appends current content to `sent` and records the slot index
      .update() → overwrites the specific slot this instance owns

    Slot-based update (not sent[-1]) is required for concurrent on_message
    tests where two messages are in flight and each must update its own slot.
    """
    instance = AsyncMock()
    instance.content = content
    _slot: List[int] = []  # mutable closure; populated on first send()

    async def _send():
        _slot.append(len(sent))
        sent.append(instance.content)

    async def _update():
        if _slot:
            sent[_slot[0]] = instance.content

    instance.send = _send
    instance.update = _update
    return instance


def _make_cl_mock(session_id: str, sent: List[str]) -> MagicMock:
    """
    Build an explicit MagicMock for the chainlit module.

    Must be passed as `new=` to patch() — Chainlit's module-level __getattr__
    raises KeyError (not AttributeError) for unknown attrs, which breaks
    mock's auto-introspection when called without `new=`.

    session.thread_id = session_id simulates the data layer making the
    conversation ID stable across reconnects (same as session.id here).
    """
    cl_mock = MagicMock()
    session = MagicMock()
    session.id = session_id
    session.thread_id = session_id  # stable conversation ID from data layer
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

        Two separate graph instances sharing the same SQLite file simulate two
        separate process lifetimes against the same persistent store.
        """
        thread_id = "user-persisted:mini"

        with _shared_sqlite_checkpointer() as db_path:
            # First process lifetime: user had a session
            with SqliteSaver.from_conn_string(db_path) as cp1:
                graph_p1 = _build_minimal_graph(cp1)
                graph_p1.update_state(_config(thread_id), {
                    "onboarding_profile": "Sofia — Sofia's full profile content here",
                })

            # Backend restarts: new connection to the same file, new graph instance
            with SqliteSaver.from_conn_string(db_path) as cp2:
                graph_p2 = _build_minimal_graph(cp2)
                recovered = _read_profile(graph_p2, thread_id)

        assert recovered is not None, (
            "User's session profile was not accessible after a backend restart. "
            "This means any reconnect triggers a full re-init, as if the user is new."
        )

    @pytest.mark.asyncio
    async def test_user_sees_no_greeting_when_reconnecting_after_backend_restart(self):
        """
        End-to-end version: user had a session, backend restarted, user returned.
        They should land back in their conversation — not see a fresh greeting.

        Requires both fixes together:
          - state survives process restart (persistent checkpointer)
          - on_chat_start detects existing state and skips re-init (idempotency guard)

        Real example: thread e97a9b78 — 10 re-inits with no user interaction,
        each caused by the backend restarting and re-touching the session.
        """
        thread_id = "user-post-restart"
        sent: List[str] = []

        with _shared_sqlite_checkpointer() as db_path:
            # First process: user onboarded and had a conversation
            with SqliteSaver.from_conn_string(db_path) as cp1:
                graph_p1 = _build_minimal_graph(cp1)
                graph_p1.update_state(_config(f"{thread_id}:mini"), {
                    "onboarding_profile": "Daniel — rural patient, transportation barriers",
                    "messages": [HumanMessage(content="How long does TB treatment take?")],
                })

            # Backend restarts: new connection, new graph instance, same DB file
            with SqliteSaver.from_conn_string(db_path) as cp2:
                graph_p2 = _build_minimal_graph(cp2)
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


# ---------------------------------------------------------------------------
# Scenario 5: User returns after session expiry — on_chat_resume fires
# ---------------------------------------------------------------------------

class TestOnChatResumeForReturningUser:
    """
    When the Chainlit data layer is configured, returning users trigger
    on_chat_resume instead of on_chat_start. on_chat_resume must:
      - restore graph / variant / thread_id into user_session
      - send NO messages (user lands back in their existing conversation)

    This covers Type B long-gap crashes (>1h idle) where session.id changed
    but on_chat_resume correctly links back to the same conversation thread.

    Reference crash data: thread 1e3e514d (35 re-inits over 12 days),
    thread e97a9b78 (10 re-inits with zero user messages).
    """

    def _make_resume_thread_dict(self, thread_id: str) -> dict:
        """Minimal ThreadDict as Chainlit would pass to on_chat_resume."""
        return {
            "id": thread_id,
            "createdAt": "2025-01-01T00:00:00Z",
            "name": "Test thread",
            "userId": "user-1",
            "userIdentifier": "anonymous",
            "tags": [],
            "metadata": {},
            "steps": [],
            "elements": [],
        }

    @pytest.mark.asyncio
    async def test_on_chat_resume_sends_no_messages(self):
        """
        A user whose session expired and reconnects should not see any new
        messages — they're landing back in their existing conversation.
        """
        graph = _make_graph()
        thread_id = "user-long-idle-return"
        sent: List[str] = []
        cl_mock = _make_cl_mock(thread_id, sent)

        with patch("chainlit_app.cl", new=cl_mock), \
             patch("chainlit_app.select_graph", return_value=(graph, "mini")), \
             patch("chainlit_app.log_session_event"):

            from chainlit_app import on_chat_resume
            await on_chat_resume(self._make_resume_thread_dict(thread_id))

        assert len(sent) == 0, (
            f"on_chat_resume should send no messages to a returning user. "
            f"Got {len(sent)} message(s): {[m[:60] for m in sent]}"
        )

    @pytest.mark.asyncio
    async def test_on_chat_resume_restores_session_state(self):
        """
        After on_chat_resume, the user_session must have graph, variant, and
        thread_id populated so that on_message can function immediately.
        """
        graph = _make_graph()
        thread_id = "user-resume-state-check"
        sent: List[str] = []
        cl_mock = _make_cl_mock(thread_id, sent)

        with patch("chainlit_app.cl", new=cl_mock), \
             patch("chainlit_app.select_graph", return_value=(graph, "mini")), \
             patch("chainlit_app.log_session_event"):

            from chainlit_app import on_chat_resume
            await on_chat_resume(self._make_resume_thread_dict(thread_id))

        assert cl_mock.user_session.get("graph") is graph, (
            "on_chat_resume must restore graph into user_session."
        )
        assert cl_mock.user_session.get("variant") == "mini", (
            "on_chat_resume must restore variant into user_session."
        )
        assert cl_mock.user_session.get("thread_id") == f"{thread_id}:mini", (
            "on_chat_resume must restore thread_id into user_session."
        )


# ---------------------------------------------------------------------------
# Scenario 6: Two messages sent before the first reply arrives
# ---------------------------------------------------------------------------

class TestConcurrentOnMessage:
    """
    A user types a follow-up message before the first reply finishes streaming
    (slow Bedrock response, or double-tap on mobile). Both messages must complete
    without errors. They must be serialized — not interleaved — so graph state
    stays coherent.

    The per-session lock in on_message exists for this case. This test verifies
    it works as intended: second call waits, both succeed, no errors are swallowed.
    """

    @pytest.mark.asyncio
    async def test_two_concurrent_messages_both_complete_without_error(self):
        """
        Fire two on_message calls concurrently on the same session.
        Both should receive a reply. Neither should error or be dropped.
        """
        import asyncio as _asyncio
        import time as _time

        call_log: List[str] = []

        def _slow_invoke(graph, text, thread_id):
            # Runs in a real thread via anyio.to_thread.run_sync — use time.sleep,
            # not asyncio.sleep.
            call_log.append(f"start:{text}")
            _time.sleep(0.05)  # simulate slow Bedrock
            call_log.append(f"end:{text}")
            from langchain_core.messages import AIMessage as _AI
            return {"messages": [_AI(content=f"reply to: {text}")]}

        graph = _make_graph()
        sent: List[str] = []

        user_session = _FakeUserSession({
            "graph": graph,
            "thread_id": "concurrent-msgs:mini",
            "variant": "mini",
        })
        cl_mock = MagicMock()
        cl_mock.user_session = user_session
        cl_mock.Message.side_effect = lambda content="": _capture_message(content, sent)

        with patch("chainlit_app.cl", new=cl_mock), \
             patch("chainlit_app.log_session_event"), \
             patch("chainlit_app.invoke_graph_with_retry", side_effect=_slow_invoke):

            from chainlit_app import on_message
            msg1, msg2 = MagicMock(), MagicMock()
            msg1.content = "first message"
            msg2.content = "second message"

            await _asyncio.gather(on_message(msg1), on_message(msg2))

        # Both slots must have been filled with real replies (blank "" placeholders
        # are replaced in-place by update(), so sent should contain two reply strings)
        assert len(sent) == 2, (
            f"Expected 2 slots in sent (one per message). Got {len(sent)}: {sent}"
        )
        assert all(m.startswith("reply to:") for m in sent), (
            f"Both slots should contain replies. Got: {sent}"
        )

        # Calls must have been serialized: one invoke fully completes before the next starts
        assert call_log.index("end:first message") < call_log.index("start:second message") or \
               call_log.index("end:second message") < call_log.index("start:first message"), (
            f"Messages were interleaved — session lock did not serialize them. "
            f"Call log: {call_log}"
        )
