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

## Session state

LangGraph state is persisted to `./data/langgraph_checkpoints.db` (SQLite) so sessions survive process restarts. Override the path with `TBTST_CHECKPOINT_DB`.

For multi-instance deployments, point `TBTST_CHECKPOINT_DB` at a network-mounted path or migrate to `PostgresSaver` with a shared DB.

## Testing

Unit tests (no Bedrock calls):

```bash
uv run pytest -q
```

Integration tests (calls Bedrock):

```bash
RUN_BEDROCK_TESTS=1 uv run pytest -q -m integration
```

### Session reinit regression tests (`tests/test_session_reinit_bugs.py`)

Covers crash patterns from production (40 threads, 10 confirmed mid-conversation crashes):

| Scenario | Test |
|---|---|
| New user sees exactly one greeting | `test_new_user_sees_exactly_one_greeting` |
| Rapid double-connect → one greeting | `test_two_rapid_connections_show_one_greeting_not_two` |
| Idle reconnect → no repeated greeting | `test_user_returning_after_idle_sees_no_new_greeting` |
| Mid-conversation reconnect → no re-init | `test_user_who_was_mid_conversation_is_not_reset_by_reconnect` |
| User context survives backend restart | `test_user_context_is_accessible_after_backend_restarts` |
| No greeting after backend restart | `test_user_sees_no_greeting_when_reconnecting_after_backend_restart` |
| Bedrock error → inline error, not re-init | `test_throttling_error_shows_inline_error_not_a_fresh_greeting` |
| Long-idle resume sends no messages | `test_on_chat_resume_sends_no_messages` |
| Long-idle resume restores session state | `test_on_chat_resume_restores_session_state` |

All 9 pass. Fixes applied:

- `SqliteSaver` replaces `MemorySaver` in `graph.py` — state survives process restart
- Idempotency guard + per-thread lock in `on_chat_start` — covers Type A (race) and Type B short-gap reconnects
- `@cl.data_layer` + `@cl.header_auth_callback` + `@cl.on_chat_resume` in `chainlit_app.py` — covers Type B long-gap reconnects (session expiry)
- `session.thread_id` replaces `session.id` as the thread identifier — stable conversation ID across WebSocket reconnects

### Chainlit DB schema (data layer)

Run once before first deploy (or after pointing at a new DB):

```bash
uv run python scripts/init_chainlit_db.py
```

Uses `TBTST_CHAINLIT_DB` (explicit async URL) → `DATABASE_URL` (Postgres, auto-upgraded to asyncpg) → SQLite fallback at `./data/chainlit.db`.
