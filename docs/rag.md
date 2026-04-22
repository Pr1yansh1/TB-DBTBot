# RAG Documentation

[← Back to README](../README.md)

---

## Active TB RAG

The bot uses retrieval-augmented generation to answer factual TB questions. All source documents are in Spanish and cover TB diagnosis, treatment, side effects, and patient education. The vector index is built at startup and cached locally.

### Source documents (`src/TB-RAG-documents/`)

| File | Source | Topic |
|---|---|---|
| `MayoClinic_TB_Diagnostico_Tratamiento_GENERAL.txt` | Mayo Clinic | Active TB diagnosis and treatment |
| `MayoClinic_TB_Diagnostico_Tratamiento_LATENT.txt` | Mayo Clinic | Latent TB infection (LTBI) |
| `Preguntas_y_respuestas_TB_CDC_GENERAL.txt` | CDC | General TB Q&A |
| `Preguntas_y_respuestas_TB_CDC_LATENT_LTBI.txt` | CDC | Latent TB Q&A |
| `abece-tuberculosis-msps copy.txt` | MSPS (Colombia) | TB basics |
| `guia-de-participacion-comunitaria-y-tuberculosis copy.txt` | — | Community participation guide |
| `hospital_muniz_2011_guia_tuberculosis copy.txt` | Hospital Muñiz | Clinical TB guide |
| `tratamiento_y_prevencion_de_la_tuberculosis_2015 copy.txt` | — | Treatment and prevention |
| `tuberculosis-material-educativo-escuelas-docentes copy.txt` | — | Educational material |
| `tuberculosis-prevenible-y-curable-8825-diptico copy.txt` | — | Patient-facing pamphlet |

### GENERAL vs LATENT split

**Why two corpora?**

During development, the classifier and retrieval operated on a single merged index. Queries about general active TB consistently pulled in chunks from latent TB documents (LTBI), producing answers that mixed active and latent treatment protocols — incorrect and potentially harmful for a patient on active TB treatment.

The core problem: latent TB documents discuss *preventive* treatment (e.g. isoniazid monotherapy for 9 months), while active TB documents discuss *curative* multi-drug regimens. If a patient on active treatment asks "how long does my treatment last?" and the retrieval returns a LTBI chunk, the answer ("9 months of isoniazid") contradicts their actual 6-month RIPE protocol.

**How the split works:**

Documents are tagged at index-build time based on filename:
- `"latent"` — filenames containing `latent` or `latente`
- `"general"` — everything else

At query time, the classifier emits a `tb_topic` field (`"general"` or `"latent"`). This field is threaded through graph state and passed to `retrieve_tb_docs`:

```python
retrieve_tb_docs(query="...", allow_latent=True)   # latent questions: all docs
retrieve_tb_docs(query="...", allow_latent=False)  # general questions: only general docs
```

When `allow_latent=False` (the default), a ChromaDB metadata filter `{"tb_topic": "general"}` excludes latent documents from retrieval. This is enforced in `rag_utils.py:get_retriever()` and tested in `test_graph_routing.py::TestStateFlow`.

### Embedding model

`hiiamsid/sentence_similarity_spanish_es` — a Spanish sentence similarity model from HuggingFace. Set via `TBTST_EMBED_MODEL`. The model is loaded once and cached (`@lru_cache`) so it is not reloaded per request.

### Index lifecycle

The Chroma index is persisted to `cache/chroma/` (configurable via `CHROMA_PERSIST_DIR`). On startup, `rag_utils.py` computes a SHA-256 fingerprint over all source `.txt` files (path, size, mtime) and compares it to the saved fingerprint at `cache/chroma/_docs_fingerprint.sha256`.

- If the fingerprint matches and the index directory is non-empty → load existing index, no rebuild.
- If the fingerprint changed, the index is missing, or `RAG_FORCE_REBUILD=1` → delete the old index and rebuild from scratch.

This means **adding, editing, or removing a source document automatically triggers a rebuild** on the next startup without manual intervention.

**Manual rebuild:**

```bash
RAG_FORCE_REBUILD=1 chainlit run chainlit_app.py
```

Or delete the index directory directly:

```bash
rm -rf cache/chroma/
```

### Retrieval parameters

| Variable | Default | Notes |
|---|---|---|
| `RAG_K` | `4` | Number of chunks retrieved per query |
| `RAG_CHUNK_SIZE` | `900` | Tokens per chunk (tiktoken encoder) |
| `RAG_CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `RAG_DOCS_DIR` | `src/TB-RAG-documents` | Source document directory |
| `CHROMA_PERSIST_DIR` | `./cache/chroma` | Index persistence path |
| `RAG_FORCE_REBUILD` | `0` | Set to `1` to force rebuild on next start |

### EC2 warmup

On EC2, the embedding model and vector store are warmed up at server start via `warmup_rag()` (called from `chainlit_app.py`). Without warmup, the first user query would trigger a slow cold load of the SentenceTransformer model. The warmup is cached — calling it multiple times is safe.

---

## Archived: DBT RAG (failed experiment)

Early in development, we attempted to feed the Spanish DBT skills manual directly into the DBT node via RAG — the hypothesis being that retrieved exercise text would improve the specificity of skill coaching responses.

### What was tried

The DBT manual (`archive/dbt-rag/DBT-RAG-documents/Manual-2023-enumerado-.pdf`) was preprocessed into chunks using `archive/dbt-rag/scripts/preprocess_dbt_pdf.py`. The preprocessing pipeline produced several output formats still present in `archive/dbt-rag/DBT-RAG-documents/`:

| Output | Description |
|---|---|
| `processed/chunks/` | Raw text chunks (`.txt`) from each page/section |
| `processed/blocks/blocks.jsonl` | Structured block-level extraction (paragraph, header, list) |
| `processed/docs/` | Full-page reconstructions |
| `processed/units/` | Semantic unit groupings (skill, worksheet, handout) |
| `processed/viz/` | Visualization outputs from the preprocessing run |
| `processed/index.json` | Chunk metadata index |
| `processed_fichas_only/` | Subset: only the *fichas* (skill worksheets), not the instructional text |

### Why it was abandoned

The retrieval approach didn't improve DBT response quality and introduced new problems:

1. **Exercise text is procedural, not conversational.** DBT worksheets are designed for therapist-guided sessions with forms to fill out. Retrieved chunks read as clinical instructions, not empathetic conversation. The LLM had to work against the retrieved text to sound natural.

2. **Chunk boundaries broke skill coherence.** A DBT skill like TIP (Temperature, Intense Exercise, Paced Breathing) spans multiple worksheet pages. Retrieval returned partial descriptions that cut off mid-skill, producing incomplete or misleading coaching instructions.

3. **The manual is in a different register than patient conversations.** The manual is written for clinicians and trained DBT practitioners. Injecting it into a patient-facing conversation produced overly technical, jargon-heavy responses.

4. **Manually crafted prompts outperformed retrieval.** The current `prompts/DBT/` prompts are hand-written distillations of each module's key skills and techniques. They are shorter (405 lines total vs 674 for the verbose v1 versions), consistently formatted, and significantly easier to maintain and tune than retrieval-dependent prompts.

The preprocessed outputs are kept in `archive/dbt-rag/` as a record of the experiment.
