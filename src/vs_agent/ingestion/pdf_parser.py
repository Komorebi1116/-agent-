from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedPage:
    page: int
    text: str


@dataclass
class ParsedPaper:
    title: str
    authors: list[str]
    year: str | None
    pages: list[ParsedPage]


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guess_title(metadata_title: str | None, first_page_text: str) -> str:
    if metadata_title and metadata_title.strip() and metadata_title.lower() != "untitled":
        return metadata_title.strip()
    lines = [line.strip() for line in first_page_text.splitlines() if len(line.strip()) > 8]
    for line in lines[:12]:
        if not re.search(r"^(abstract|keywords|introduction)\b", line, re.I):
            return line[:240]
    return "Untitled paper"


def _guess_authors(metadata_author: str | None) -> list[str]:
    if not metadata_author:
        return []
    parts = re.split(r"[,;]|\band\b", metadata_author)
    return [part.strip() for part in parts if part.strip()]


def _guess_year(text: str, metadata: dict) -> str | None:
    for value in (metadata.get("creationDate"), metadata.get("modDate")):
        if value:
            match = re.search(r"(19|20)\d{2}", value)
            if match:
                return match.group(0)
    match = re.search(r"\b(19|20)\d{2}\b", text[:4000])
    return match.group(0) if match else None


def parse_pdf(path: Path | str) -> ParsedPaper:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("PDF 解析需要安装 PyMuPDF，请先运行 pip install -r requirements.txt") from exc

    pdf_path = Path(path)
    with fitz.open(pdf_path) as doc:
        pages = [
            ParsedPage(page=index + 1, text=_clean_text(page.get_text("text")))
            for index, page in enumerate(doc)
        ]
        first_text = pages[0].text if pages else ""
        metadata = doc.metadata or {}
        return ParsedPaper(
            title=_guess_title(metadata.get("title"), first_text),
            authors=_guess_authors(metadata.get("author")),
            year=_guess_year(first_text, metadata),
            pages=pages,
        )
