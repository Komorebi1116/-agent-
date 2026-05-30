from __future__ import annotations

from vs_agent.config import Settings
from vs_agent.models import RetrievalHit
from vs_agent.retrieval.embedder import create_embedder
from vs_agent.retrieval.reranker import rerank_hits
from vs_agent.retrieval.vector_store import create_vector_store
from vs_agent.storage.sqlite_store import SQLiteStore


class Retriever:
    def __init__(self, store: SQLiteStore, settings: Settings):
        self.store = store
        embedder = create_embedder(settings)
        self.vector_store = create_vector_store(settings, store, embedder)

    def retrieve(self, query: str, limit: int = 8) -> list[RetrievalHit]:
        card_rows = self.vector_store.query(query, kind="card", limit=limit * 2)
        chunk_rows = self.vector_store.query(query, kind="chunk", limit=limit * 2)
        hits: list[RetrievalHit] = []

        for row in card_rows:
            card = self.store.get_card(row["owner_id"])
            if not card or card.user_status == "ignored":
                continue
            paper = self.store.get_paper(card.paper_id)
            if not paper:
                continue
            hits.append(RetrievalHit(kind="card", score=float(row["score"]), paper=paper, card=card))

        for row in chunk_rows:
            chunk = self.store.get_chunk(row["owner_id"])
            if not chunk:
                continue
            paper = self.store.get_paper(chunk.paper_id)
            if not paper:
                continue
            hits.append(RetrievalHit(kind="chunk", score=float(row["score"]), paper=paper, chunk=chunk))

        deduped = dedupe_hits(rerank_hits(hits))
        return deduped[:limit]


def dedupe_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    seen: set[tuple[str, str]] = set()
    result: list[RetrievalHit] = []
    for hit in hits:
        owner = hit.card.card_id if hit.card else hit.chunk.chunk_id if hit.chunk else ""
        key = (hit.kind, owner)
        if key in seen:
            continue
        seen.add(key)
        result.append(hit)
    return result
