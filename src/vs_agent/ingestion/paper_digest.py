from __future__ import annotations

import re
from dataclasses import dataclass, field

from vs_agent.models import Chunk


FIELD_LIMITS = {
    "abstract_text": 1400,
    "introduction_text": 1800,
    "method_text": 3200,
    "loss_text": 1400,
    "experiment_text": 2200,
    "result_text": 2200,
    "discussion_text": 1200,
    "conclusion_text": 1000,
    "other_relevant_text": 1400,
}
MAX_CHUNK_INDEX_ITEMS = 40
CHUNK_PREVIEW_CHARS = 220


@dataclass
class DigestChunkIndex:
    chunk_id: str
    page: int | None
    section: str | None
    text_preview: str


@dataclass
class PaperDigest:
    paper_id: str
    title_candidate: str
    abstract_text: str = ""
    introduction_text: str = ""
    method_text: str = ""
    loss_text: str = ""
    experiment_text: str = ""
    result_text: str = ""
    discussion_text: str = ""
    conclusion_text: str = ""
    other_relevant_text: str = ""
    chunk_index: list[DigestChunkIndex] = field(default_factory=list)

    def to_prompt_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title_candidate": self.title_candidate,
            "abstract_text": self.abstract_text,
            "introduction_text": self.introduction_text,
            "method_text": self.method_text,
            "loss_text": self.loss_text,
            "experiment_text": self.experiment_text,
            "result_text": self.result_text,
            "discussion_text": self.discussion_text,
            "conclusion_text": self.conclusion_text,
            "other_relevant_text": self.other_relevant_text,
            "chunk_index": [item.__dict__ for item in self.chunk_index],
        }


def build_paper_digest(paper_id: str, chunks: list[Chunk]) -> PaperDigest:
    digest = PaperDigest(paper_id=paper_id, title_candidate="")
    buckets: dict[str, list[str]] = {key: [] for key in FIELD_LIMITS}

    for chunk in chunks:
        text = normalize_digest_text(chunk.text)
        if not text:
            continue
        if len(digest.chunk_index) < MAX_CHUNK_INDEX_ITEMS:
            digest.chunk_index.append(
                DigestChunkIndex(
                    chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    section=chunk.section,
                    text_preview=text[:CHUNK_PREVIEW_CHARS],
                )
            )
        bucket = classify_section(chunk.section)
        buckets[bucket].append(f"[chunk_id={chunk.chunk_id}, page={chunk.page}, section={chunk.section}]\n{text}")

    # Preserve method/loss/experiment/result first by giving them their own buckets.
    for field_name, parts in buckets.items():
        setattr(digest, field_name, limit_text("\n\n".join(parts), FIELD_LIMITS[field_name]))

    if chunks:
        digest.title_candidate = chunks[0].text.splitlines()[0][:180].strip()
    return digest


def classify_section(section: str | None) -> str:
    if not section:
        return "other_relevant_text"
    lower = section.lower()
    if "abstract" in lower:
        return "abstract_text"
    if "intro" in lower:
        return "introduction_text"
    if any(term in lower for term in ["method", "approach", "architecture", "model", "proposed"]):
        return "method_text"
    if "loss" in lower or "objective" in lower:
        return "loss_text"
    if any(term in lower for term in ["experiment", "implementation", "training", "dataset", "ablation"]):
        return "experiment_text"
    if any(term in lower for term in ["result", "evaluation", "metric"]):
        return "result_text"
    if "discussion" in lower:
        return "discussion_text"
    if "conclusion" in lower:
        return "conclusion_text"
    return "other_relevant_text"


def normalize_digest_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def limit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "\n[TRUNCATED]"
