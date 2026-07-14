"""Text embedding for hierarchical RAG (TF-IDF default, optional sentence-transformers)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextEmbedder:
    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self._model = None
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: np.ndarray | None = None
        self._texts: list[str] = []

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        try:
            import sentence_transformers  # noqa: F401

            return "sentence_transformers"
        except ImportError:
            return "tfidf"

    def fit(self, texts: list[str]) -> None:
        self._texts = texts
        backend = self._resolve_backend()
        if backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._matrix = np.array(self._model.encode(texts, show_progress_bar=False))
            self.backend = "sentence_transformers"
        else:
            self._vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
            self._matrix = self._vectorizer.fit_transform(texts).astype(np.float32)
            self.backend = "tfidf"

    def encode_query(self, query: str) -> np.ndarray:
        if self.backend == "sentence_transformers":
            return np.array(self._model.encode([query])[0])
        assert self._vectorizer is not None
        return self._vectorizer.transform([query]).astype(np.float32)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        if self._matrix is None:
            raise RuntimeError("Embedder not fitted")
        q = self.encode_query(query)
        if self.backend == "sentence_transformers":
            q = q.reshape(1, -1)
            sims = cosine_similarity(q, self._matrix)[0]
        else:
            sims = (self._matrix @ q.T).toarray().ravel()
        ranked = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in ranked]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "texts": self._texts,
            "vectorizer": self._vectorizer,
            "matrix": self._matrix if self.backend == "sentence_transformers" else self._matrix.toarray(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.backend = payload["backend"]
        self._texts = payload["texts"]
        self._vectorizer = payload["vectorizer"]
        mat = payload["matrix"]
        if self.backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._matrix = mat
        else:
            from scipy.sparse import csr_matrix

            self._matrix = csr_matrix(mat)
