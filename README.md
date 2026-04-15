# TB-TST Helper (LangGraph + AWS Bedrock)

A small TB treatment-support chatbot with:
- safety + routing classifier
- FAQ (RAG) + DBT responses
- crisis fallback message
- per-thread memory via `thread_id`

## Setup

Create a `.env` (or export env vars):

**Required**
- `AWS_REGION` (default: `us-west-2`, fallback: `AWS_DEFAULT_REGION`)
- `BEDROCK_MODEL_ID` (default: `anthropic.claude-3-5-sonnet-20240620-v1:0`)

**Optional**
- `AWS_PROFILE` (local dev only)
- `IS_EC2=1` (on EC2: ignore `AWS_PROFILE` and use the instance role)

Install deps (using `uv`):

```bash
uv sync
````

## Run (CLI)

```bash
uv run python -m tbtst_bot.app --variant mini
uv run python -m tbtst_bot.app --variant full
```

## Logging

**Deployment (quiet)**

```bash
export TBTST_LOG_LEVEL=WARNING
export TBTST_DEBUG_METRICS=0
```

**Testing (verbose)**

```bash
export TBTST_LOG_LEVEL=INFO
export TBTST_DEBUG_METRICS=1
```

## Prompts

Prompt files live in `prompts/` (plain `.txt`).

If you use Python `.format()` on prompt strings that contain literal JSON, escape braces:

* `{{` and `}}` for literal `{` and `}`
* keep placeholders like `{user_text}` unescaped

## Baselines

Scripts live under `baselines/`:

```bash
uv run python baselines/<script_name>.py
```

## Testing

Unit tests (no Bedrock calls):

```bash
uv run pytest -q
```

Integration tests (calls Bedrock):

```bash
RUN_BEDROCK_TESTS=1 uv run pytest -q -m integration
```

### Session reinit crash tests (`tests/test_session_reinit_bugs.py`)

Regression tests for two crash patterns identified from production transcripts
(40 threads, 10 confirmed mid-conversation crashes).

**What they test (by user scenario):**

| Scenario | Test | Status |
|---|---|---|
| New user opens chat, sees one greeting | `test_new_user_sees_exactly_one_greeting` | passes |
| Two rapid connections → one greeting, not two | `test_two_rapid_connections_show_one_greeting_not_two` | **failing** |
| User returns after idle → no repeated greeting | `test_user_returning_after_idle_sees_no_new_greeting` | **failing** |
| Mid-conversation reconnect → no re-init | `test_user_who_was_mid_conversation_is_not_reset_by_reconnect` | **failing** |
| User context survives backend restart | `test_user_context_is_accessible_after_backend_restarts` | **failing** |
| No greeting shown after backend restart | `test_user_sees_no_greeting_when_reconnecting_after_backend_restart` | **failing** |
| Bedrock error shows error message, not re-init | `test_throttling_error_shows_inline_error_not_a_fresh_greeting` | passes |

**Two fixes needed to clear all failures:**

1. **Persistent checkpointer** — replace `MemorySaver()` in `graph.py:1523` with a
   checkpointer backed by a shared store (e.g. `SqliteSaver`, `PostgresSaver`).
   Without this, all LangGraph state is lost on every worker restart, making
   every reconnect look like a new session.

2. **Idempotency guard in `on_chat_start`** — before seeding and sending, check
   whether the graph already has `onboarding_profile` for this `thread_id`.
   If it does, return early. This gate needs a per-thread lock to be safe under
   concurrent connections.
