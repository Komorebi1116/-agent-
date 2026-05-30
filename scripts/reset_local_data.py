from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vs_agent.config import load_settings
from vs_agent.storage.sqlite_store import SQLiteStore


def main() -> None:
    settings = load_settings()
    SQLiteStore(settings.db_path).reset()
    for folder in (settings.papers_dir, settings.exports_dir, settings.chroma_dir):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    print(f"Reset local data under {settings.data_dir}")


if __name__ == "__main__":
    main()
