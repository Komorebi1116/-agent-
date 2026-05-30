from __future__ import annotations

import re

from vs_agent.models import MethodCard


def dedupe_method_cards(cards: list[MethodCard], max_cards: int) -> list[MethodCard]:
    deduped: list[MethodCard] = []
    by_key: dict[str, MethodCard] = {}
    for card in cards:
        key = card_dedupe_key_from_card(card)
        if key in by_key:
            existing = by_key[key]
            existing.source_chunk_ids = sorted(set(existing.source_chunk_ids + card.source_chunk_ids))
            if len(card.source_quote) > len(existing.source_quote):
                existing.source_quote = card.source_quote
                existing.page = card.page
            existing.tags = sorted(set(existing.tags + card.tags))
            existing.metric_related = sorted(set(existing.metric_related + card.metric_related))
            continue
        by_key[key] = card
        deduped.append(card)
        if len(deduped) >= max_cards:
            break
    return deduped


def card_dedupe_key(
    title: str,
    model_name: str | None,
    module_type: str | None,
    problem_target: str | None,
) -> str:
    normalized_title = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    normalized_title = re.sub(r"\b(component|module|design|method|candidate)\b", "", normalized_title)
    normalized_title = re.sub(r"\s+", " ", normalized_title).strip()
    return "|".join(
        [
            normalized_title,
            (model_name or "").lower().strip(),
            (module_type or "").lower().strip(),
            (problem_target or "").lower().strip(),
        ]
    )


def card_dedupe_key_from_card(card: MethodCard) -> str:
    return card_dedupe_key(
        card.title,
        card.model_or_module or card.model_name,
        card.card_type or card.module_type,
        card.core_problem or card.problem_target,
    )
