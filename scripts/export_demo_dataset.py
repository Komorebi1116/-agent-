from __future__ import annotations

import json
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
    store = SQLiteStore(settings.db_path)
    payload = {
        "papers": [paper.__dict__ for paper in store.list_papers()],
        "chunks": [chunk.__dict__ for chunk in store.list_chunks()],
        "method_cards": [card.__dict__ for card in store.list_cards()],
    }
    output = settings.exports_dir / "demo_dataset.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

