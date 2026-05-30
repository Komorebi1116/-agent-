from __future__ import annotations

from vs_agent.config import Settings
from vs_agent.retrieval.chroma_store import ChromaVectorStore
from vs_agent.retrieval.embedder import Embedder
from vs_agent.storage.sqlite_store import SQLiteStore


def create_vector_store(settings: Settings, store: SQLiteStore, embedder: Embedder) -> ChromaVectorStore:
    return ChromaVectorStore(settings=settings, embedder=embedder)

