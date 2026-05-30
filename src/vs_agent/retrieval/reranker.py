from __future__ import annotations

from vs_agent.models import RetrievalHit


def rerank_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    def key(hit: RetrievalHit) -> tuple[float, float]:
        status_boost = 0.0
        if hit.card and hit.card.user_status == "saved":
            status_boost = 0.25
        elif hit.card and hit.card.user_status == "candidate":
            status_boost = 0.08
        kind_boost = 0.05 if hit.kind == "card" else 0.0
        return (hit.score + status_boost + kind_boost, status_boost)

    return sorted(hits, key=key, reverse=True)

