from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    papers_dir: Path
    chroma_dir: Path
    exports_dir: Path
    db_path: Path
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    embedding_model: str
    llm_timeout_seconds: int
    llm_max_retries: int

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key and self.openai_model)

    @property
    def remote_embedding_enabled(self) -> bool:
        return bool(self.openai_api_key and self.embedding_model)


def load_settings() -> Settings:
    if load_dotenv:
        load_dotenv(PROJECT_ROOT / ".env")

    data_dir = Path(os.getenv("VS_AGENT_DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    papers_dir = data_dir / "papers"
    chroma_dir = data_dir / "chroma"
    exports_dir = data_dir / "exports"

    for path in (data_dir, papers_dir, chroma_dir, exports_dir):
        path.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        papers_dir=papers_dir,
        chroma_dir=chroma_dir,
        exports_dir=exports_dir,
        db_path=data_dir / "app.db",
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.getenv(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/"),
        openai_model=os.getenv("OPENAI_MODEL", "qwen-plus").strip(),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-v4").strip(),
        llm_timeout_seconds=int(os.getenv("OPENAI_TIMEOUT_SECONDS", "180")),
        llm_max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
    )
