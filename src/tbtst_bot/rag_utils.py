# src/tbtst_bot/rag_utils.py (Chroma version, consistent with docs in src/TB-RAG-documents)
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

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

# Allow overrides, but default matches your repo structure
RAG_DOCS_DIR = os.getenv("RAG_DOCS_DIR", DEFAULT_DOCS_DIR)

# Chroma persists to a folder
RAG_PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", DEFAULT_PERSIST_DIR)

RAG_K = int(os.getenv("RAG_K", "4"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))


def _load_documents() -> List[Document]:
    if not os.path.isdir(RAG_DOCS_DIR):
        raise FileNotFoundError(
            f"RAG_DOCS_DIR does not exist: {RAG_DOCS_DIR}\n"
            f"Expected TXT files under: {DEFAULT_DOCS_DIR}\n"
            f"Override with env var RAG_DOCS_DIR if needed."
        )

    loader = DirectoryLoader(
        RAG_DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()

    # Normalize "source" to a readable filename for citations/debugging
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown_source"
        d.metadata["source"] = os.path.basename(src)

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
    embeddings = _get_embeddings()

    # If the persist dir exists and has content, load existing index
    if os.path.isdir(RAG_PERSIST_DIR) and os.listdir(RAG_PERSIST_DIR):
        return Chroma(
            persist_directory=RAG_PERSIST_DIR,
            embedding_function=embeddings,
        )

    docs = _load_documents()
    chunks = _split_documents(docs)

    os.makedirs(RAG_PERSIST_DIR, exist_ok=True)

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=RAG_PERSIST_DIR,
    )
    # Persist explicitly
    vs.persist()
    return vs


@lru_cache(maxsize=1)
def get_retriever():
    vs = load_or_create_vectorstore()
    return vs.as_retriever(search_kwargs={"k": RAG_K})



@tool
def retrieve_tb_docs(query: str) -> dict:
    """Retrieve relevant TB snippets from the local document collection."""
    retriever = get_retriever()
    docs = retriever.invoke(query)

    sources = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        sources.append(
            {
                "id": i,
                "title": src,          # you can improve later
                "filename": src,       # keep consistent with graph.py expectations
                "excerpt": d.page_content[:1200],  # cap to keep prompts smaller
            }
        )

    return {"sources": sources}

