from __future__ import annotations

from vs_agent.models import MethodCard
from vs_agent.storage.sqlite_store import SQLiteStore


class CardService:
    def __init__(self, store: SQLiteStore):
        self.store = store

    def save_cards(self, cards: list[MethodCard]) -> None:
        self.store.upsert_cards(cards)

    def set_status(self, card_id: str, status: str) -> None:
        if status not in {"candidate", "saved", "ignored"}:
            raise ValueError("status must be candidate, saved, or ignored")
        self.store.update_card_status(card_id, status)

    def set_tags(self, card_id: str, tags_text: str) -> None:
        tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
        self.store.update_card_tags(card_id, tags)

    def rename(self, card_id: str, title: str) -> None:
        self.store.update_card_title(card_id, title.strip())

    def list_active(self) -> list[MethodCard]:
        return self.store.list_cards(include_ignored=False)

