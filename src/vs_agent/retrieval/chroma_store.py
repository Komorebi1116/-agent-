from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vs_agent.config import Settings
from vs_agent.models import Chunk, MethodCard
from vs_agent.retrieval.embedder import Embedder


VectorKind = Literal["chunk", "card"]


@dataclass(frozen=True)
class ChromaQueryRow:
    kind: VectorKind
    owner_id: str
    paper_id: str
    score: float
    text: str
    metadata: dict


class ChromaVectorStore:
    def __init__(self, settings: Settings, embedder: Embedder):
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("ChromaDB 未安装，请先运行 pip install -r requirements.txt") from exc

        self.settings = settings
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.chunk_collection = self.client.get_or_create_collection(name="paper_chunks")
        self.card_collection = self.client.get_or_create_collection(name="method_cards")

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)
        self.chunk_collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "kind": "chunk",
                    "owner_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "page": chunk.page if chunk.page is not None else -1,
                    "section": chunk.section or "",
                    "embedding_model": self.embedder.name,
                }
                for chunk in chunks
            ],
        )

    def add_cards(self, cards: list[MethodCard]) -> None:
        if not cards:
            return

        texts = [card_to_embedding_text(card) for card in cards]
        embeddings = self.embedder.embed_batch(texts)
        self.card_collection.upsert(
            ids=[card.card_id for card in cards],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "kind": "card",
                    "owner_id": card.card_id,
                    "paper_id": card.paper_id,
                    "page": card.evidence_page if card.evidence_page is not None else card.page if card.page is not None else -1,
                    "status": card.user_status,
                    "source": card.source,
                    "embedding_model": self.embedder.name,
                }
                for card in cards
            ],
        )

    def query(self, query: str, kind: VectorKind, limit: int = 8) -> list[dict]:
        collection = self._collection_for_kind(kind)
        query_embedding = self.embedder.embed(query)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["metadatas", "documents", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]

        rows: list[dict] = []
        for owner_id, metadata, document, distance in zip(ids, metadatas, documents, distances):
            metadata = metadata or {}
            rows.append(
                {
                    "kind": kind,
                    "owner_id": metadata.get("owner_id") or owner_id,
                    "paper_id": metadata.get("paper_id", ""),
                    "score": distance_to_score(float(distance)),
                    "text": document or "",
                    "metadata": metadata,
                }
            )
        return rows

    def delete_paper(self, paper_id: str) -> None:
        for collection in (self.chunk_collection, self.card_collection):
            try:
                collection.delete(where={"paper_id": paper_id})
            except Exception:
                pass

    def _collection_for_kind(self, kind: VectorKind):
        if kind == "chunk":
            return self.chunk_collection
        if kind == "card":
            return self.card_collection
        raise ValueError("kind must be 'chunk' or 'card'")


def card_to_embedding_text(card: MethodCard) -> str:
    return " ".join(
        [
            card.title,
            card.card_type,
            card.task_type,
            card.input_output or "",
            card.core_problem,
            card.proposed_solution,
            card.model_or_module or "",
            card.technical_details,
            card.loss_design or "",
            card.training_setting or "",
            " ".join(card.evaluation_metrics),
            card.why_it_works or "",
            card.reusable_value_for_user_project,
            card.limitation or "",
            card.evidence_quote,
            card.task or "",
            card.model_name or "",
            card.module_type or "",
            card.problem_target or "",
            " ".join(card.metric_related),
            card.evidence_summary,
            card.reusable_point,
            card.source_quote,
            " ".join(card.tags),
        ]
    )


def distance_to_score(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))
