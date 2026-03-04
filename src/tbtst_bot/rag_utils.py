# src/tbtst_bot/rag_utils.py
# Chroma version, consistent with docs in src/TB-RAG-documents
from __future__ import annotations

import hashlib
import os
import shutil
from functools import lru_cache
from typing import Any, Dict, List

from langchain.tools import tool
from langchain_core.documents import Document

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -------------------------
# Paths (robust to CWD)
# -------------------------
HERE = os.path.dirname(os.path.abspath(__file__))  # .../src/tbtst_bot
SRC_DIR = os.path.abspath(os.path.join(HERE, ".."))  # .../src

DEFAULT_DOCS_DIR = os.path.join(SRC_DIR, "TB-RAG-documents")
DEFAULT_PERSIST_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "cache", "chroma"))

# -------------------------
# Config
# -------------------------
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "hiiamsid/sentence_similarity_spanish_es",
)

RAG_DOCS_DIR = os.getenv("RAG_DOCS_DIR", DEFAULT_DOCS_DIR)
RAG_PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", DEFAULT_PERSIST_DIR)

RAG_K = int(os.getenv("RAG_K", "4"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))

# Optional manual "force rebuild now"
RAG_FORCE_REBUILD = os.getenv("RAG_FORCE_REBUILD", "0") == "1"


# -------------------------
# Fingerprint-based rebuild
# -------------------------
def _iter_txt_files(root_dir: str) -> List[str]:
    out: List[str] = []
    for base, _dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                out.append(os.path.join(base, f))
    out.sort()
    return out


def _docs_fingerprint() -> str:
    """
    Computes a stable fingerprint based on:
      - relative path
      - file size
      - mtime
    This is fast and good enough for "docs changed => rebuild index".
    """
    if not os.path.isdir(RAG_DOCS_DIR):
        raise FileNotFoundError(
            f"RAG_DOCS_DIR does not exist: {RAG_DOCS_DIR}\n"
            f"Expected TXT files under: {DEFAULT_DOCS_DIR}\n"
            f"Override with env var RAG_DOCS_DIR if needed."
        )

    files = _iter_txt_files(RAG_DOCS_DIR)
    h = hashlib.sha256()
    h.update(os.path.abspath(RAG_DOCS_DIR).encode("utf-8"))
    for path in files:
        rel = os.path.relpath(path, RAG_DOCS_DIR)
        try:
            st = os.stat(path)
            payload = f"{rel}|{st.st_size}|{int(st.st_mtime)}"
        except FileNotFoundError:
            # If a file disappears mid-walk, just skip; fingerprint will differ next run anyway.
            continue
        h.update(payload.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _fingerprint_path() -> str:
    return os.path.join(RAG_PERSIST_DIR, "_docs_fingerprint.sha256")


def _read_saved_fingerprint() -> str:
    fp = _fingerprint_path()
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _write_saved_fingerprint(val: str) -> None:
    os.makedirs(RAG_PERSIST_DIR, exist_ok=True)
    with open(_fingerprint_path(), "w", encoding="utf-8") as f:
        f.write(val)


def _persist_dir_has_index() -> bool:
    return os.path.isdir(RAG_PERSIST_DIR) and bool(os.listdir(RAG_PERSIST_DIR))


def _maybe_rebuild_needed() -> bool:
    if RAG_FORCE_REBUILD:
        return True

    cur = _docs_fingerprint()
    saved = _read_saved_fingerprint()

    # If no saved fingerprint OR differs -> rebuild.
    if not saved or saved != cur:
        return True

    # If fingerprint matches but index dir missing -> rebuild.
    if not _persist_dir_has_index():
        return True

    return False


# -------------------------
# Document loading & metadata
# -------------------------
def _tag_tb_topic(filename: str) -> str:
    """
    Tag docs as 'latent' vs 'general' based on the DOC itself (filename).
    This is NOT query keyword matching.
    """
    b = (filename or "").lower()
    if "latente" in b or "latent" in b:
        return "latent"
    return "general"


def _load_documents() -> List[Document]:
    loader = DirectoryLoader(
        RAG_DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()

    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown_source"
        base = os.path.basename(src)
        d.metadata["source"] = base
        d.metadata["filename"] = base
        d.metadata["tb_topic"] = _tag_tb_topic(base)

    return docs


def _split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)


def load_or_create_vectorstore() -> Chroma:
    """
    Creates/loads persisted Chroma. Rebuilds automatically if docs changed since last build.
    """
    embeddings = _get_embeddings()

    rebuild = _maybe_rebuild_needed()
    cur_fp = _docs_fingerprint()

    if rebuild:
        # Clean persist dir to avoid mixed indexes / stale docs.
        if os.path.isdir(RAG_PERSIST_DIR):
            shutil.rmtree(RAG_PERSIST_DIR, ignore_errors=True)
        os.makedirs(RAG_PERSIST_DIR, exist_ok=True)

        docs = _load_documents()
        chunks = _split_documents(docs)

        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=RAG_PERSIST_DIR,
        )
        vs.persist()
        _write_saved_fingerprint(cur_fp)
        return vs

    # Load existing
    return Chroma(
        persist_directory=RAG_PERSIST_DIR,
        embedding_function=embeddings,
    )


@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    return load_or_create_vectorstore()


@lru_cache(maxsize=2)
def get_retriever(*, allow_latent: bool):
    """
    Two cached retrievers:
      - allow_latent=False -> filter to tb_topic='general'
      - allow_latent=True  -> no filter (general + latent)
    """
    vs = _get_vectorstore()
    if allow_latent:
        return vs.as_retriever(search_kwargs={"k": RAG_K})
    return vs.as_retriever(search_kwargs={"k": RAG_K, "filter": {"tb_topic": "general"}})


@tool
def retrieve_tb_docs(query: str, allow_latent: bool = False) -> dict:
    """
    Retrieve relevant TB snippets from the local document collection.

    allow_latent:
      - False: exclude latent TB docs (default)
      - True: include latent TB docs
    """
    retriever = get_retriever(allow_latent=bool(allow_latent))
    docs = retriever.invoke(query)

    sources: List[Dict[str, Any]] = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("filename") or d.metadata.get("source") or "unknown"
        sources.append(
            {
                "id": i,
                "title": src,
                "filename": src,
                "excerpt": (d.page_content or "")[:1200],
            }
        )

    return {"sources": sources}
