from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from vs_agent.config import Settings


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+\-_.]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")
REMOTE_EMBEDDING_BATCH_SIZE = 10
REMOTE_EMBEDDING_DIMENSIONS = 1024


class Embedder(Protocol):
    name: str

    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    def similarity(self, left: list[float], right: list[float]) -> float:
        ...


class LocalHashEmbedder:
    """Small deterministic embedder for a no-key, no-model-download MVP."""

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions
        self.name = f"local-hash-{dimensions}"

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def similarity(self, left: list[float], right: list[float]) -> float:
        return cosine_similarity(left, right)


class OpenAICompatibleEmbedder:
    def __init__(self, settings: Settings):
        if not settings.remote_embedding_enabled:
            raise RuntimeError("Embedding is not configured. Set OPENAI_API_KEY and OPENAI_EMBEDDING_MODEL.")
        self.settings = settings
        self.name = settings.embedding_model

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned_texts = [text.strip() or " " for text in texts]
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("远程向量化需要 requests，请先运行 pip install -r requirements.txt") from exc

        embeddings: list[list[float]] = []
        for batch in _batched(cleaned_texts, REMOTE_EMBEDDING_BATCH_SIZE):
            response = requests.post(
                f"{self.settings.openai_base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.embedding_model,
                    "input": batch,
                    "dimensions": REMOTE_EMBEDDING_DIMENSIONS,
                    "encoding_format": "float",
                },
                timeout=60,
            )
            if not response.ok:
                raise RuntimeError(
                    f"Embedding API 请求失败：HTTP {response.status_code}，响应：{response.text[:1000]}"
                )
            data = response.json()
            items = data["data"]
            if all("index" in item for item in items):
                items = sorted(items, key=lambda item: item["index"])
            embeddings.extend([[float(value) for value in item["embedding"]] for item in items])
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding API 返回数量异常：请求 {len(texts)} 条，返回 {len(embeddings)} 条")
        return embeddings

    def similarity(self, left: list[float], right: list[float]) -> float:
        return cosine_similarity(left, right)


def create_embedder(settings: Settings | None = None) -> Embedder:
    if settings and settings.remote_embedding_enabled:
        return OpenAICompatibleEmbedder(settings)
    return LocalHashEmbedder()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def tokenize(text: str) -> list[str]:
    words = [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
    grams: list[str] = []
    for index in range(len(words) - 1):
        if len(words[index]) == 1 and "\u4e00" <= words[index] <= "\u9fff":
            grams.append(words[index] + words[index + 1])
    return words + grams
