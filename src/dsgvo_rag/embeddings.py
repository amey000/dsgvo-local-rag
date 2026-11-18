"""Embeddings with intfloat/multilingual-e5-large (1024 dims, runs locally on CPU)."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class E5Embedder:
    """E5 models expect ``passage:`` / ``query:`` prefixes; vectors are L2-normalised (cosine)."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self.dim: int = int(get_dim())

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode([f"passage: {t}" for t in texts]) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"query: {text}"])[0]
