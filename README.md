Here’s a small, practical `README.md` you can paste in:

````md
# TB-TST Helper (LangGraph + AWS Bedrock)

A small TB treatment-support chatbot with:
- a single Bedrock classifier call (safety + route)
- two specialist tools: **FAQ** and **DBT**
- crisis handling (static crisis message)
- LangGraph memory via `thread_id`

## Setup

Create a `.env` (or export env vars):

- `AWS_REGION` (default: `us-west-2`)
- `BEDROCK_MODEL_ID` (e.g. `anthropic.claude-3-5-sonnet-20240620-v1:0`)
- `AWS_PROFILE` (optional)

Install deps (using `uv`):

```bash
uv sync
````

## Run the app (CLI)

```bash
uv run -m tbtst_bot.app
```

## Prompts

Prompt files live in `prompts/` (plain `.txt`).

If a prompt contains literal JSON and you use Python `.format()`, escape braces:

* use `{{` and `}}` for literal `{` and `}`
* keep `{user_text}` (or other placeholders) unescaped

## Run baselines

Baselines live under `baselines/`.

Typical pattern:

```bash
uv run python baselines/<script_name>.py
```

(See `baselines/` for available scripts and their flags.)

## Testing

### Unit tests (fast, no Bedrock calls)

```bash
uv run pytest -q
```

### Integration tests (calls Bedrock; costs money)

Classifier integration tests are skipped unless you opt in:

```bash
RUN_BEDROCK_TESTS=1 uv run pytest -q -m integration
```

Run only integration tests:

```bash
RUN_BEDROCK_TESTS=1 uv run pytest -q tests/test_classifier_integration.py
```

```
::contentReference[oaicite:0]{index=0}
```
❯ RUN_BEDROCK_TESTS=1 uv run pytest -q -m integration
