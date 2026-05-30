from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: list[str]
    year: str | None
    file_path: str
    upload_time: str
    status: str
    file_md5: str | None = None
    original_filename: str | None = None
    filename_title: str | None = None


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    section: str | None
    page: int | None
    text: str
    embedding_id: str | None = None


@dataclass
class MethodCard:
    card_id: str
    paper_id: str
    title: str
    card_type: str = ""
    task_type: str = ""
    input_output: str | None = None
    core_problem: str = ""
    proposed_solution: str = ""
    model_or_module: str | None = None
    technical_details: str = ""
    loss_design: str | None = None
    training_setting: str | None = None
    evaluation_metrics: list[str] = field(default_factory=list)
    why_it_works: str | None = None
    reusable_value_for_user_project: str = ""
    limitation: str | None = None
    evidence_quote: str = ""
    evidence_page: int | None = None
    source_chunk_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence_check_passed: bool = False
    evidence_issue: str | None = None
    tags: list[str] = field(default_factory=list)
    user_status: str = "candidate"
    source: str = "llm"
    # Backward-compatible fields used by older retrieval/answer UI code.
    task: str | None = None
    model_name: str | None = None
    module_type: str | None = None
    problem_target: str | None = None
    metric_related: list[str] = field(default_factory=list)
    evidence_summary: str = ""
    reusable_point: str = ""
    source_quote: str = ""
    page: int | None = None


@dataclass
class Citation:
    paper_id: str
    paper_title: str
    page: int | None
    quote: str
    chunk_id: str | None = None
    card_id: str | None = None


@dataclass
class RetrievalHit:
    kind: str
    score: float
    paper: Paper
    chunk: Chunk | None = None
    card: MethodCard | None = None


@dataclass
class ChatAnswer:
    answer: str
    used_cards: list[str]
    used_chunks: list[str]
    citations: list[Citation]
    related_cards: list[MethodCard]
    intent: str = "qa"
    table_rows: list[dict[str, Any]] = field(default_factory=list)
