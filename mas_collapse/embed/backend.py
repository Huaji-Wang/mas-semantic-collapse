from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np


class Embedder(ABC):
    name: str

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized float32 array of shape (n, d)."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32)


class BgeM3Embedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.name = f"bge_m3:{model_name}"
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


class OpenAIEmbedder(Embedder):
    """Reserved Kong-aligned backend: text-embedding-3-large."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        batch_size: int = 64,
    ):
        from openai import OpenAI

        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for openai embedding backend")
        self.name = f"openai:{model}"
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            resp = self.client.embeddings.create(model=self.model, input=batch)
            # API returns in order
            out.extend([d.embedding for d in sorted(resp.data, key=lambda x: x.index)])
        return _l2_normalize(np.asarray(out, dtype=np.float32))


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words hashing for pipeline smoke tests (not for papers)."""

    def __init__(self, dim: int = 256):
        self.name = f"hashing:{dim}"
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                out[i, hash(tok) % self.dim] += 1.0
        return _l2_normalize(out)


def build_embedder(cfg: dict[str, Any]) -> Embedder:
    emb = cfg["embedding"]
    backend = emb.get("backend", "bge_m3")
    if backend in {"bge_m3", "bge-m3"}:
        return BgeM3Embedder(
            model_name=emb.get("model_name", "BAAI/bge-m3"),
            device=emb.get("device", "cpu"),
            batch_size=int(emb.get("batch_size", 32)),
        )
    if backend in {"openai", "openai_text_embedding_3_large", "text-embedding-3-large"}:
        return OpenAIEmbedder(
            api_key=emb.get("openai_api_key", ""),
            model=emb.get("openai_model", "text-embedding-3-large"),
            batch_size=int(emb.get("batch_size", 64)),
        )
    if backend in {"hashing", "smoke"}:
        return HashingEmbedder(dim=int(emb.get("hash_dim", 256)))
    raise ValueError(f"Unknown embedding backend: {backend}")

