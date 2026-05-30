from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from vs_agent.cards.card_utils import dedupe_method_cards
from vs_agent.models import Chunk, MethodCard, Paper


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _loads_list(value: str | None) -> list:
    if not value:
        return []
    loaded = json.loads(value)
    return loaded if isinstance(loaded, list) else []


class SQLiteStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL,
                    year TEXT,
                    file_path TEXT NOT NULL,
                    upload_time TEXT NOT NULL,
                    status TEXT NOT NULL,
                    file_md5 TEXT,
                    original_filename TEXT,
                    filename_title TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    section TEXT,
                    page INTEGER,
                    text TEXT NOT NULL,
                    embedding_id TEXT,
                    FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );

                CREATE TABLE IF NOT EXISTS method_cards (
                    card_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    task TEXT,
                    model_name TEXT,
                    module_type TEXT,
                    problem_target TEXT,
                    metric_related_json TEXT NOT NULL,
                    evidence_summary TEXT NOT NULL,
                    reusable_point TEXT NOT NULL,
                    source_chunk_ids_json TEXT NOT NULL,
                    source_quote TEXT NOT NULL,
                    page INTEGER,
                    tags_json TEXT NOT NULL,
                    user_status TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'unknown',
                    card_type TEXT NOT NULL DEFAULT '',
                    task_type TEXT NOT NULL DEFAULT '',
                    input_output TEXT,
                    core_problem TEXT NOT NULL DEFAULT '',
                    proposed_solution TEXT NOT NULL DEFAULT '',
                    model_or_module TEXT,
                    technical_details TEXT NOT NULL DEFAULT '',
                    loss_design TEXT,
                    training_setting TEXT,
                    evaluation_metrics_json TEXT NOT NULL DEFAULT '[]',
                    why_it_works TEXT,
                    reusable_value_for_user_project TEXT NOT NULL DEFAULT '',
                    limitation TEXT,
                    evidence_quote TEXT NOT NULL DEFAULT '',
                    evidence_page INTEGER,
                    confidence REAL NOT NULL DEFAULT 0,
                    evidence_check_passed INTEGER NOT NULL DEFAULT 0,
                    evidence_issue TEXT,
                    FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                );

                CREATE TABLE IF NOT EXISTS vectors (
                    embedding_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                );
                """
            )
            paper_columns = {row["name"] for row in conn.execute("PRAGMA table_info(papers)").fetchall()}
            if "file_md5" not in paper_columns:
                conn.execute("ALTER TABLE papers ADD COLUMN file_md5 TEXT")
            if "original_filename" not in paper_columns:
                conn.execute("ALTER TABLE papers ADD COLUMN original_filename TEXT")
            if "filename_title" not in paper_columns:
                conn.execute("ALTER TABLE papers ADD COLUMN filename_title TEXT")
            card_columns = {row["name"] for row in conn.execute("PRAGMA table_info(method_cards)").fetchall()}
            if "source" not in card_columns:
                conn.execute("ALTER TABLE method_cards ADD COLUMN source TEXT NOT NULL DEFAULT 'unknown'")
            for column_name, column_sql in {
                "card_type": "TEXT NOT NULL DEFAULT ''",
                "task_type": "TEXT NOT NULL DEFAULT ''",
                "input_output": "TEXT",
                "core_problem": "TEXT NOT NULL DEFAULT ''",
                "proposed_solution": "TEXT NOT NULL DEFAULT ''",
                "model_or_module": "TEXT",
                "technical_details": "TEXT NOT NULL DEFAULT ''",
                "loss_design": "TEXT",
                "training_setting": "TEXT",
                "evaluation_metrics_json": "TEXT NOT NULL DEFAULT '[]'",
                "why_it_works": "TEXT",
                "reusable_value_for_user_project": "TEXT NOT NULL DEFAULT ''",
                "limitation": "TEXT",
                "evidence_quote": "TEXT NOT NULL DEFAULT ''",
                "evidence_page": "INTEGER",
                "confidence": "REAL NOT NULL DEFAULT 0",
                "evidence_check_passed": "INTEGER NOT NULL DEFAULT 0",
                "evidence_issue": "TEXT",
            }.items():
                if column_name not in card_columns:
                    conn.execute(f"ALTER TABLE method_cards ADD COLUMN {column_name} {column_sql}")
            conn.execute("DELETE FROM method_cards WHERE source='rule_based'")
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_file_md5
                ON papers(file_md5)
                WHERE file_md5 IS NOT NULL
                """
            )

    def upsert_paper(self, paper: Paper) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title=excluded.title,
                    authors_json=excluded.authors_json,
                    year=excluded.year,
                    file_path=excluded.file_path,
                    upload_time=excluded.upload_time,
                    status=excluded.status,
                    file_md5=excluded.file_md5,
                    original_filename=excluded.original_filename,
                    filename_title=excluded.filename_title
                """,
                (
                    paper.paper_id,
                    paper.title,
                    _dumps(paper.authors),
                    paper.year,
                    paper.file_path,
                    paper.upload_time,
                    paper.status,
                    paper.file_md5,
                    paper.original_filename,
                    paper.filename_title,
                ),
            )

    def update_paper_status(self, paper_id: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE papers SET status=? WHERE paper_id=?", (status, paper_id))

    def list_papers(self) -> list[Paper]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM papers ORDER BY upload_time DESC").fetchall()
        return [self._paper_from_row(row) for row in rows]

    def get_paper(self, paper_id: str) -> Paper | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        return self._paper_from_row(row) if row else None

    def get_paper_by_file_md5(self, file_md5: str) -> Paper | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM papers WHERE file_md5=?", (file_md5,)).fetchone()
        return self._paper_from_row(row) if row else None

    def get_paper_by_md5(self, file_md5: str) -> Paper | None:
        return self.get_paper_by_file_md5(file_md5)

    def delete_paper_artifacts(self, paper_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM vectors WHERE paper_id=?", (paper_id,))
            conn.execute("DELETE FROM method_cards WHERE paper_id=?", (paper_id,))
            conn.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    paper_id=excluded.paper_id,
                    section=excluded.section,
                    page=excluded.page,
                    text=excluded.text,
                    embedding_id=excluded.embedding_id
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.paper_id,
                        chunk.section,
                        chunk.page,
                        chunk.text,
                        chunk.embedding_id,
                    )
                    for chunk in chunks
                ],
            )

    def list_chunks(self, paper_id: str | None = None) -> list[Chunk]:
        query = "SELECT * FROM chunks"
        params: tuple[str, ...] = ()
        if paper_id:
            query += " WHERE paper_id=?"
            params = (paper_id,)
        query += " ORDER BY paper_id, page, chunk_id"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return self._chunk_from_row(row) if row else None

    def upsert_cards(self, cards: Iterable[MethodCard]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO method_cards (
                    card_id, paper_id, title, task, model_name, module_type, problem_target,
                    metric_related_json, evidence_summary, reusable_point, source_chunk_ids_json,
                    source_quote, page, tags_json, user_status, source,
                    card_type, task_type, input_output, core_problem, proposed_solution,
                    model_or_module, technical_details, loss_design, training_setting,
                    evaluation_metrics_json, why_it_works, reusable_value_for_user_project,
                    limitation, evidence_quote, evidence_page, confidence,
                    evidence_check_passed, evidence_issue
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(card_id) DO UPDATE SET
                    paper_id=excluded.paper_id,
                    title=excluded.title,
                    task=excluded.task,
                    model_name=excluded.model_name,
                    module_type=excluded.module_type,
                    problem_target=excluded.problem_target,
                    metric_related_json=excluded.metric_related_json,
                    evidence_summary=excluded.evidence_summary,
                    reusable_point=excluded.reusable_point,
                    source_chunk_ids_json=excluded.source_chunk_ids_json,
                    source_quote=excluded.source_quote,
                    page=excluded.page,
                    tags_json=excluded.tags_json,
                    user_status=excluded.user_status,
                    source=excluded.source,
                    card_type=excluded.card_type,
                    task_type=excluded.task_type,
                    input_output=excluded.input_output,
                    core_problem=excluded.core_problem,
                    proposed_solution=excluded.proposed_solution,
                    model_or_module=excluded.model_or_module,
                    technical_details=excluded.technical_details,
                    loss_design=excluded.loss_design,
                    training_setting=excluded.training_setting,
                    evaluation_metrics_json=excluded.evaluation_metrics_json,
                    why_it_works=excluded.why_it_works,
                    reusable_value_for_user_project=excluded.reusable_value_for_user_project,
                    limitation=excluded.limitation,
                    evidence_quote=excluded.evidence_quote,
                    evidence_page=excluded.evidence_page,
                    confidence=excluded.confidence,
                    evidence_check_passed=excluded.evidence_check_passed,
                    evidence_issue=excluded.evidence_issue
                """,
                [
                    (
                        card.card_id,
                        card.paper_id,
                        card.title,
                        card.task or card.task_type,
                        card.model_name or card.model_or_module,
                        card.module_type or card.card_type,
                        card.problem_target or card.core_problem,
                        _dumps(card.metric_related or card.evaluation_metrics),
                        card.evidence_summary or card.proposed_solution,
                        card.reusable_point or card.reusable_value_for_user_project,
                        _dumps(card.source_chunk_ids),
                        card.source_quote or card.evidence_quote,
                        card.page if card.page is not None else card.evidence_page,
                        _dumps(card.tags),
                        card.user_status,
                        card.source,
                        card.card_type,
                        card.task_type,
                        card.input_output,
                        card.core_problem,
                        card.proposed_solution,
                        card.model_or_module,
                        card.technical_details,
                        card.loss_design,
                        card.training_setting,
                        _dumps(card.evaluation_metrics),
                        card.why_it_works,
                        card.reusable_value_for_user_project,
                        card.limitation,
                        card.evidence_quote or card.source_quote,
                        card.evidence_page if card.evidence_page is not None else card.page,
                        card.confidence,
                        1 if card.evidence_check_passed else 0,
                        card.evidence_issue,
                    )
                    for card in cards
                ],
            )

    def list_cards(self, paper_id: str | None = None, include_ignored: bool = True) -> list[MethodCard]:
        clauses: list[str] = []
        params: list[str] = []
        if paper_id:
            clauses.append("paper_id=?")
            params.append(paper_id)
        if not include_ignored:
            clauses.append("user_status != 'ignored'")
        query = "SELECT * FROM method_cards"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " AND source != 'rule_based'" if clauses else " WHERE source != 'rule_based'"
        query += " ORDER BY user_status DESC, title"
        with self.connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return dedupe_method_cards([self._card_from_row(row) for row in rows], max_cards=200)

    def get_card(self, card_id: str) -> MethodCard | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM method_cards WHERE card_id=?", (card_id,)).fetchone()
        return self._card_from_row(row) if row else None

    def update_card_status(self, card_id: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE method_cards SET user_status=? WHERE card_id=?", (status, card_id))

    def update_card_tags(self, card_id: str, tags: list[str]) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE method_cards SET tags_json=? WHERE card_id=?", (_dumps(tags), card_id))

    def update_card_title(self, card_id: str, title: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE method_cards SET title=? WHERE card_id=?", (title, card_id))

    def upsert_vector(
        self,
        embedding_id: str,
        kind: str,
        owner_id: str,
        paper_id: str,
        text: str,
        metadata: dict,
        embedding: list[float],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO vectors VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(embedding_id) DO UPDATE SET
                    kind=excluded.kind,
                    owner_id=excluded.owner_id,
                    paper_id=excluded.paper_id,
                    text=excluded.text,
                    metadata_json=excluded.metadata_json,
                    embedding_json=excluded.embedding_json
                """,
                (embedding_id, kind, owner_id, paper_id, text, _dumps(metadata), _dumps(embedding)),
            )

    def list_vectors(self, kind: str | None = None) -> list[dict]:
        query = "SELECT * FROM vectors"
        params: tuple[str, ...] = ()
        if kind:
            query += " WHERE kind=?"
            params = (kind,)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "embedding_id": row["embedding_id"],
                "kind": row["kind"],
                "owner_id": row["owner_id"],
                "paper_id": row["paper_id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata_json"]),
                "embedding": json.loads(row["embedding_json"]),
            }
            for row in rows
        ]

    def reset(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                DELETE FROM vectors;
                DELETE FROM method_cards;
                DELETE FROM chunks;
                DELETE FROM papers;
                """
            )

    @staticmethod
    def _paper_from_row(row: sqlite3.Row) -> Paper:
        return Paper(
            paper_id=row["paper_id"],
            title=row["title"],
            authors=_loads_list(row["authors_json"]),
            year=row["year"],
            file_path=row["file_path"],
            upload_time=row["upload_time"],
            status=row["status"],
            file_md5=row["file_md5"] if "file_md5" in row.keys() else None,
            original_filename=row["original_filename"] if "original_filename" in row.keys() else None,
            filename_title=row["filename_title"] if "filename_title" in row.keys() else None,
        )

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            paper_id=row["paper_id"],
            section=row["section"],
            page=row["page"],
            text=row["text"],
            embedding_id=row["embedding_id"],
        )

    @staticmethod
    def _card_from_row(row: sqlite3.Row) -> MethodCard:
        return MethodCard(
            card_id=row["card_id"],
            paper_id=row["paper_id"],
            title=row["title"],
            card_type=row["card_type"] if "card_type" in row.keys() else row["module_type"] or "",
            task_type=row["task_type"] if "task_type" in row.keys() else row["task"] or "",
            input_output=row["input_output"] if "input_output" in row.keys() else None,
            core_problem=row["core_problem"] if "core_problem" in row.keys() else row["problem_target"] or "",
            proposed_solution=row["proposed_solution"] if "proposed_solution" in row.keys() else row["evidence_summary"] or "",
            model_or_module=row["model_or_module"] if "model_or_module" in row.keys() else row["model_name"],
            technical_details=row["technical_details"] if "technical_details" in row.keys() else "",
            loss_design=row["loss_design"] if "loss_design" in row.keys() else None,
            training_setting=row["training_setting"] if "training_setting" in row.keys() else None,
            evaluation_metrics=_loads_list(row["evaluation_metrics_json"]) if "evaluation_metrics_json" in row.keys() else _loads_list(row["metric_related_json"]),
            why_it_works=row["why_it_works"] if "why_it_works" in row.keys() else None,
            reusable_value_for_user_project=row["reusable_value_for_user_project"] if "reusable_value_for_user_project" in row.keys() else row["reusable_point"] or "",
            limitation=row["limitation"] if "limitation" in row.keys() else None,
            evidence_quote=row["evidence_quote"] if "evidence_quote" in row.keys() else row["source_quote"] or "",
            evidence_page=row["evidence_page"] if "evidence_page" in row.keys() else row["page"],
            confidence=float(row["confidence"]) if "confidence" in row.keys() else 0.0,
            evidence_check_passed=bool(row["evidence_check_passed"]) if "evidence_check_passed" in row.keys() else False,
            evidence_issue=row["evidence_issue"] if "evidence_issue" in row.keys() else None,
            source_chunk_ids=_loads_list(row["source_chunk_ids_json"]),
            tags=_loads_list(row["tags_json"]),
            user_status=row["user_status"],
            source=row["source"] if "source" in row.keys() else "unknown",
            task=row["task"],
            model_name=row["model_name"],
            module_type=row["module_type"],
            problem_target=row["problem_target"],
            metric_related=_loads_list(row["metric_related_json"]),
            evidence_summary=row["evidence_summary"],
            reusable_point=row["reusable_point"],
            source_quote=row["source_quote"],
            page=row["page"],
        )
