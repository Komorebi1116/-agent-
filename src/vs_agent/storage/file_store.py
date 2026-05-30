from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import BinaryIO


def safe_filename(filename: str) -> str:
    stem = Path(filename).stem or "paper"
    suffix = Path(filename).suffix or ".pdf"
    stem = re.sub(r"[^\w\-.()\u4e00-\u9fff]+", "_", stem, flags=re.UNICODE).strip("_")
    return f"{stem[:120]}{suffix.lower()}"


class FileStore:
    def __init__(self, papers_dir: Path | str):
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, uploaded_file: BinaryIO, filename: str, paper_id: str) -> Path:
        paper_dir = self.papers_dir / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        target = paper_dir / safe_filename(filename)
        with target.open("wb") as fh:
            shutil.copyfileobj(uploaded_file, fh)
        return target

