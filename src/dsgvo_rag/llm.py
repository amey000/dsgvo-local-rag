"""Minimal client for the local Ollama HTTP API."""

from __future__ import annotations

from typing import Protocol

import httpx


class LLM(Protocol):
    model: str

    def chat(self, messages: list[dict[str, str]]) -> str: ...


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        num_ctx: int = 8192,
        timeout_s: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.options = {"temperature": temperature, "num_ctx": num_ctx}
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": self.options,
            },
        )
        if response.status_code == 404:
            raise RuntimeError(
                f"Modell '{self.model}' ist in Ollama nicht vorhanden – "
                "einmalig `make pull-model` ausführen."
            )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def available_models(self) -> list[str]:
        response = self._client.get("/api/tags", timeout=5)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    def is_ready(self) -> bool:
        try:
            names = self.available_models()
        except httpx.HTTPError:
            return False
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in names
