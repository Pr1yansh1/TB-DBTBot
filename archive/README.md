# Archive

[← Back to README](../README.md)

Older bot versions kept for reference. Nothing here is used by the active bot (`chainlit_app.py` at repo root).

---

## Version timeline

| Version | Folder | Key characteristic | Status |
|---|---|---|---|
| MVP Baseline | `mvp-baseline/` | Single giant prompt, direct API call, no graph | Archived |
| MVP Supervisor | `mvp-supervisor/` | LangGraph with a supervisor node routing to specialist tools | Archived |
| DBT Mini | `dbt-mini/` | Current graph shape, single flat DBT node | Archived |
| DBT RAG | `dbt-rag/` | Experimental: DBT manual fed into DBT node via RAG | Abandoned |
| **DBT Full** | `../src/tbtst_bot/` | DBT subgraph with 4 module nodes (dt/er/ie/mind) | **Active** |

---

## Running an archived version

All archived bots share the same virtual environment and `tbtst_bot` package as the active bot. Run from the **repo root**:

```bash
# DBT Mini
ARCHIVE_VERSION=dbt-mini chainlit run archive/chainlit_app.py

# Override port to run alongside the active bot
ARCHIVE_VERSION=dbt-mini chainlit run archive/chainlit_app.py --port 8001
```

`ARCHIVE_VERSION` options: `dbt-mini`

The `mvp-baseline` scripts can be run directly:
```bash
uv run python archive/mvp-baseline/sonnet_plain.py
uv run python archive/mvp-baseline/sonnet_prompt.py
```

---

## `mvp-baseline/`

The oldest version. No LangGraph, no routing, no safety classifier. A single system prompt is passed with every message and the model responds directly.

| File | Description |
|---|---|
| `giant_prompt.txt` | The monolithic system prompt |
| `sonnet_plain.py` | Bare API call — no framework, no state |
| `sonnet_prompt.py` | Same with a small prompt templating layer |
| `TODO.txt` | Notes from this phase |

---

## `mvp-supervisor/`

First LangGraph version. Introduced a **supervisor node** that classified each message and called one of three specialist tool-nodes: `faq_tool`, `dbt_tool`, `psychoed_tool`. The supervisor was a single LLM call that both classified intent and selected the tool — routing and classification were not separated.

This architecture was replaced because:
- The supervisor LLM call was doing two jobs (classify + select) in one step with no retry or fallback.
- Safety classification was not an independent first step — a classification mistake could bypass the crisis path.
- `psychoed_tool` (psychoeducation) was redundant; its content was folded into the FAQ node.

| File | Description |
|---|---|
| `supervisor_system.txt` | Supervisor node prompt — routing + tool selection in one call |
| `tb_info_system.txt` | TB information specialist node prompt |

---

## `dbt-mini/`

Introduced the current graph architecture: classify first (with safety), then route to one of three specialist nodes (TB FAQ, DBT, misc). The DBT handling was a **single flat node** with one system prompt covering all four DBT modules (distress tolerance, mindfulness, emotion regulation, interpersonal effectiveness) in one call.

Replaced by DBT Full because a single prompt covering all four modules either:
- Became too long and hit token limits (see `dbt-modules-v1/` — 674 lines)
- Or was trimmed to the point where module-specific guidance became superficial

DBT Full routes to a dedicated node per module, each with a focused prompt.

| Path | Description |
|---|---|
| `prompts/dbt_system.txt` | The single-node DBT prompt — still compiled into `graph.py` as `GRAPH_DBT_MINI` |
| `prompts/dbt-modules-v1/` | Earlier, more verbose module prompts (~66% longer than current) — the versions that hit token limits |

---

## `dbt-rag/`

Experimental attempt to use the Spanish DBT skills manual as a retrieval corpus for the DBT node. Abandoned. See [`docs/rag.md`](../docs/rag.md#archived-dbt-rag-failed-experiment) for a full explanation of what was tried and why it was abandoned.

| Path | Description |
|---|---|
| `DBT-RAG-documents/Manual-2023-enumerado-.pdf` | Source: Spanish DBT skills manual |
| `DBT-RAG-documents/processed/` | Preprocessing outputs: chunks, blocks, docs, units, viz, index |
| `DBT-RAG-documents/processed_fichas_only/` | Subset: only the skill worksheets (*fichas*) |
| `scripts/preprocess_dbt_pdf.py` | The preprocessing pipeline that produced the above |
