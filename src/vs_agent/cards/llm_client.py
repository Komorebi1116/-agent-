from __future__ import annotations

import json
import time
from dataclasses import dataclass

from vs_agent.config import Settings


@dataclass
class LLMClient:
    settings: Settings
    timeout_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is None:
            self.timeout_seconds = self.settings.llm_timeout_seconds

    def chat_json(self, system_prompt: str, user_prompt: str) -> object:
        content = self.chat_text(system_prompt, user_prompt, response_format={"type": "json_object"})
        return json.loads(_extract_json(content))

    def chat_text(self, system_prompt: str, user_prompt: str, response_format: dict | None = None) -> str:
        if not self.settings.llm_enabled:
            raise RuntimeError("LLM is not configured. Set OPENAI_API_KEY and OPENAI_MODEL.")
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("LLM 调用需要 requests，请先运行 pip install -r requirements.txt") from exc

        payload = {
            "model": self.settings.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        if response_format:
            payload["response_format"] = response_format
        response = None
        last_error: Exception | None = None
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                response = requests.post(
                    f"{self.settings.openai_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if not response.ok:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
                break
            except Exception as exc:
                last_error = exc
                if attempt >= self.settings.llm_max_retries:
                    raise RuntimeError(
                        f"LLM请求失败，model={self.settings.openai_model}, "
                        f"timeout={self.timeout_seconds}s, attempts={attempt + 1}: {exc}"
                    ) from exc
                time.sleep(min(2**attempt, 4))
        if response is None:
            raise RuntimeError(f"LLM请求失败：{last_error}")
        data = response.json()
        return data["choices"][0]["message"]["content"]


def _extract_json(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.strip("`")
        clean = clean.removeprefix("json").strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end >= start:
        return clean[start : end + 1]
    return clean
