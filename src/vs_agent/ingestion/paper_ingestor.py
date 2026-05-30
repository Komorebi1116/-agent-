from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path

from vs_agent.cards.card_extractor import MethodCardExtractor
from vs_agent.config import Settings
from vs_agent.ingestion.chunker import chunk_pages
from vs_agent.ingestion.pdf_parser import parse_pdf
from vs_agent.models import Paper, utc_now_iso
from vs_agent.retrieval.embedder import create_embedder
from vs_agent.retrieval.vector_store import create_vector_store
from vs_agent.storage.sqlite_store import SQLiteStore


logger = logging.getLogger(__name__)


class PaperIngestor:
    def __init__(self, settings: Settings, store: SQLiteStore):
        self.settings = settings
        self.store = store
        self.embedder = create_embedder(settings)
        self.vector_store = create_vector_store(settings, store, self.embedder)
        self.card_extractor = MethodCardExtractor(settings)
        self.last_timings: dict[str, float] = {}
        self.last_duplicate = False
        self.last_card_extraction_warning: str | None = None
        self.last_card_extraction_error: str | None = None
        self.last_card_extraction_method = "LLM"
        self.last_generated_card_count = 0
        self.last_evidence_passed_count = 0

    def ingest_pdf(self, pdf_path: Path | str) -> Paper:
        self.last_timings = {}
        self.last_duplicate = False
        self.last_card_extraction_warning = None
        self.last_card_extraction_error = None
        self.last_generated_card_count = 0
        self.last_evidence_passed_count = 0
        pdf_path = Path(pdf_path)
        original_filename = pdf_path.name
        filename_title = filename_to_title(original_filename)

        started = time.perf_counter()
        file_md5 = compute_file_md5(pdf_path)
        self._record_timing("md5_seconds", started)

        existing = self.store.get_paper_by_md5(file_md5)
        if existing and existing.status == "processed":
            self.last_duplicate = True
            logger.info("PDF already ingested: file_md5=%s paper_id=%s", file_md5, existing.paper_id)
            return existing

        paper_id = existing.paper_id if existing else f"paper_{uuid.uuid4().hex[:16]}"
        if existing:
            self.store.delete_paper_artifacts(paper_id)
            if hasattr(self.vector_store, "delete_paper"):
                self.vector_store.delete_paper(paper_id)

        started = time.perf_counter()
        parsed = parse_pdf(pdf_path)
        self._record_timing("pdf_parse_seconds", started)

        paper = Paper(
            paper_id=paper_id,
            title=filename_title,
            authors=parsed.authors,
            year=parsed.year,
            file_path=str(pdf_path),
            upload_time=utc_now_iso(),
            status="processing",
            file_md5=file_md5,
            original_filename=original_filename,
            filename_title=filename_title,
        )
        self.store.upsert_paper(paper)
        try:
            started = time.perf_counter()
            chunks = chunk_pages(paper_id, parsed.pages)
            self._record_timing("chunk_split_seconds", started)

            started = time.perf_counter()
            self.store.upsert_chunks(chunks)
            self._record_timing("chunk_sqlite_seconds", started)

            started = time.perf_counter()
            self.vector_store.add_chunks(chunks)
            self._record_timing("chunk_chroma_seconds", started)

            started = time.perf_counter()
            cards = self.card_extractor.extract(paper_id, chunks)
            self.last_card_extraction_warning = self.card_extractor.last_warning
            self.last_card_extraction_error = self.card_extractor.last_error
            self.last_generated_card_count = len(cards)
            self.last_evidence_passed_count = len([card for card in cards if card.evidence_check_passed])
            self._record_timing("method_card_extract_seconds", started)

            started = time.perf_counter()
            self.store.upsert_cards(cards)
            self._record_timing("card_sqlite_seconds", started)

            started = time.perf_counter()
            self.vector_store.add_cards(cards)
            self._record_timing("card_chroma_seconds", started)

            self.store.update_paper_status(paper_id, "processed")
            paper.status = "processed"
            logger.info(
                "PDF ingest finished: paper_id=%s file_md5=%s timings=%s",
                paper_id,
                file_md5,
                self.last_timings,
            )
        except Exception:
            self.last_card_extraction_error = self.card_extractor.last_error or self.last_card_extraction_error
            self.store.update_paper_status(paper_id, "failed")
            paper.status = "failed"
            raise
        return paper

    def _record_timing(self, key: str, started: float) -> None:
        elapsed = time.perf_counter() - started
        self.last_timings[key] = elapsed
        logger.info("ingest timing %s=%.3fs", key, elapsed)


def compute_file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filename_to_title(filename: str) -> str:
    path = Path(filename)
    if path.suffix.lower() == ".pdf":
        return path.name[: -len(path.suffix)]
    return path.name
