from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from vs_agent.ingestion.pdf_parser import ParsedPage
from vs_agent.models import Chunk


SECTION_RE = re.compile(
    r"^\s*((\d+(\.\d+)*)\s+)?(abstract|introduction|related work|method|methods|materials|experiments?|results?|discussion|conclusion|references)\b",
    re.I,
)


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 2200
    overlap_chars: int = 240
    min_chars: int = 280


def normalize_text(text: str) -> str:
    text = text.replace("-\n", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section(line: str, current: str | None) -> str | None:
    clean = line.strip()
    if not clean:
        return current
    match = SECTION_RE.match(clean)
    if match:
        return clean[:120]
    if len(clean) < 80 and clean.isupper() and len(clean.split()) <= 8:
        return clean.title()
    return current


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n|(?<=[.!?。！？])\s+(?=[A-Z(])", text)
    return [part.strip() for part in parts if part.strip()]


def chunk_pages(paper_id: str, pages: list[ParsedPage], config: ChunkingConfig | None = None) -> list[Chunk]:
    cfg = config or ChunkingConfig()
    chunks: list[Chunk] = []
    section: str | None = None

    for parsed_page in pages:
        text = normalize_text(parsed_page.text)
        if not text:
            continue
        for line in text.splitlines()[:25]:
            section = detect_section(line, section)

        buffer = ""
        for para in _split_paragraphs(text):
            section = detect_section(para, section)
            if len(buffer) + len(para) + 2 <= cfg.max_chars:
                buffer = f"{buffer}\n\n{para}".strip()
                continue
            if len(buffer) >= cfg.min_chars:
                chunks.append(_make_chunk(paper_id, parsed_page.page, section, buffer))
                buffer = (buffer[-cfg.overlap_chars :] + "\n\n" + para).strip()
            else:
                buffer = f"{buffer}\n\n{para}".strip()

            while len(buffer) > cfg.max_chars:
                part = buffer[: cfg.max_chars]
                chunks.append(_make_chunk(paper_id, parsed_page.page, section, part))
                buffer = buffer[cfg.max_chars - cfg.overlap_chars :].strip()

        if len(buffer) >= cfg.min_chars:
            chunks.append(_make_chunk(paper_id, parsed_page.page, section, buffer))

    return chunks


def _make_chunk(paper_id: str, page: int, section: str | None, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"chunk_{uuid.uuid4().hex[:16]}",
        paper_id=paper_id,
        section=section,
        page=page,
        text=normalize_text(text),
    )

