"""Embedding backends for memboot."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np

from memboot.exceptions import EmbedError


class BaseEmbedder(ABC):
    """Abstract embedding backend."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (N, dim) array, L2-normalized."""

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed_texts([text])[0]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w{2,}\b", text.lower())


class TfidfEmbedder(BaseEmbedder):
    """Numpy-only TF-IDF embedder. No sklearn dependency."""

    def __init__(self, max_features: int = 512) -> None:
        self._max_features = max_features
        self._vocabulary: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._fitted = False

    @property
    def dim(self) -> int:
        return self._max_features

    def fit(self, texts: list[str]) -> None:
        """Build vocabulary and IDF weights from corpus."""
        if not texts:
            raise EmbedError("Cannot fit on empty corpus")

        n_docs = len(texts)
        doc_freq: dict[str, int] = {}
        for text in texts:
            tokens = set(_tokenize(text))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Select top features by document frequency
        sorted_tokens = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)
        top_tokens = sorted_tokens[: self._max_features]

        self._vocabulary = {token: idx for idx, (token, _) in enumerate(top_tokens)}

        # Compute IDF: log(N / (1 + df)) + 1
        idf = np.zeros(len(self._vocabulary), dtype=np.float32)
        for token, idx in self._vocabulary.items():
            df = doc_freq[token]
            idf[idx] = np.log(n_docs / (1 + df)) + 1.0

        self._idf = idf
        self._fitted = True

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Compute TF-IDF vectors, L2-normalized."""
        if not self._fitted or self._idf is None:
            raise EmbedError("Embedder not fitted. Call fit() first.")

        n_texts = len(texts)
        dim = len(self._vocabulary)
        matrix = np.zeros((n_texts, dim), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = _tokenize(text)
            if not tokens:
                continue
            token_count: dict[str, int] = {}
            for token in tokens:
                token_count[token] = token_count.get(token, 0) + 1

            total = len(tokens)
            for token, count in token_count.items():
                if token in self._vocabulary:
                    idx = self._vocabulary[token]
                    tf = count / total
                    matrix[i, idx] = tf * self._idf[idx]

        # L2 normalize rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix = matrix / norms

        # Pad or truncate to max_features
        if dim < self._max_features:
            padded = np.zeros((n_texts, self._max_features), dtype=np.float32)
            padded[:, :dim] = matrix
            matrix = padded

        return matrix

    def save_state(self) -> dict:
        """Serialize vocabulary and IDF for persistence."""
        if not self._fitted or self._idf is None:
            raise EmbedError("Cannot save state: not fitted")
        return {
            "max_features": self._max_features,
            "vocabulary": self._vocabulary,
            "idf": self._idf.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict) -> TfidfEmbedder:
        """Restore embedder from saved state."""
        embedder = cls(max_features=state["max_features"])
        embedder._vocabulary = state["vocabulary"]
        embedder._idf = np.array(state["idf"], dtype=np.float32)
        embedder._fitted = True
        return embedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """High-quality embeddings via sentence-transformers (optional)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedError(
                "sentence-transformers not installed. Install with: pip install memboot[embed]"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts using sentence-transformers."""
        return self._model.encode(texts, normalize_embeddings=True)


def get_embedder(backend: str = "tfidf", **kwargs) -> BaseEmbedder:
    """Factory function to create an embedder."""
    if backend == "tfidf":
        return TfidfEmbedder(**kwargs)
    elif backend == "sentence-transformers":
        return SentenceTransformerEmbedder(**kwargs)
    else:
        raise EmbedError(f"Unknown embedding backend: {backend}")
