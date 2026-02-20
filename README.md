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
```
