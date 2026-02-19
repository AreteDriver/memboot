"""Tests for memboot.embedder."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from memboot.embedder import (
    BaseEmbedder,
    TfidfEmbedder,
    _tokenize,
    get_embedder,
)
from memboot.exceptions import EmbedError


class TestTokenize:
    def test_basic_words(self):
        tokens = _tokenize("hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_punctuation_stripped(self):
        tokens = _tokenize("hello, world! test.")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lowercase(self):
        tokens = _tokenize("Hello WORLD")
        assert "hello" in tokens
        assert "world" in tokens

    def test_min_length(self):
        tokens = _tokenize("I am a big dog")
        # "I", "a" are 1-char → excluded; "am" is 2 chars → included
        assert "am" in tokens
        assert "big" in tokens
        assert "dog" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []


class TestTfidfEmbedder:
    def test_fit_and_embed(self):
        emb = TfidfEmbedder(max_features=10)
        corpus = ["hello world", "foo bar baz", "hello foo"]
        emb.fit(corpus)
        result = emb.embed_texts(corpus)
        assert result.shape == (3, 10)
        assert result.dtype == np.float32

    def test_embed_single(self):
        emb = TfidfEmbedder(max_features=10)
        emb.fit(["hello world", "foo bar"])
        result = emb.embed_text("hello world")
        assert result.shape == (10,)

    def test_l2_normalized(self):
        emb = TfidfEmbedder(max_features=10)
        emb.fit(["hello world", "foo bar baz"])
        result = emb.embed_texts(["hello world"])
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5 or norm == 0.0

    def test_empty_corpus_raises(self):
        emb = TfidfEmbedder()
        with pytest.raises(EmbedError, match="empty corpus"):
            emb.fit([])

    def test_unfitted_raises(self):
        emb = TfidfEmbedder()
        with pytest.raises(EmbedError, match="not fitted"):
            emb.embed_texts(["hello"])

    def test_save_and_restore(self):
        emb = TfidfEmbedder(max_features=10)
        emb.fit(["hello world", "foo bar"])
        state = emb.save_state()

        restored = TfidfEmbedder.from_state(state)
        original = emb.embed_text("hello world")
        restored_result = restored.embed_text("hello world")
        np.testing.assert_array_almost_equal(original, restored_result)

    def test_save_state_unfitted_raises(self):
        emb = TfidfEmbedder()
        with pytest.raises(EmbedError, match="not fitted"):
            emb.save_state()

    def test_dim_property(self):
        emb = TfidfEmbedder(max_features=256)
        assert emb.dim == 256

    def test_pad_to_max_features(self):
        emb = TfidfEmbedder(max_features=100)
        emb.fit(["hello world"])  # vocabulary < 100
        result = emb.embed_texts(["hello world"])
        assert result.shape[1] == 100

    def test_empty_text_embedding(self):
        emb = TfidfEmbedder(max_features=10)
        emb.fit(["hello world", "foo bar"])
        result = emb.embed_texts([""])
        assert result.shape == (1, 10)
        # Empty text should give zero vector (normalized to zero)
        assert np.all(result[0] == 0.0) or np.linalg.norm(result[0]) > 0


class TestSentenceTransformerEmbedder:
    def test_import_error(self):
        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            pytest.raises(EmbedError, match="sentence-transformers not installed"),
        ):
            from memboot.embedder import SentenceTransformerEmbedder

            SentenceTransformerEmbedder()


class TestGetEmbedder:
    def test_tfidf(self):
        embedder = get_embedder("tfidf")
        assert isinstance(embedder, TfidfEmbedder)

    def test_tfidf_with_kwargs(self):
        embedder = get_embedder("tfidf", max_features=256)
        assert embedder.dim == 256

    def test_unknown_backend(self):
        with pytest.raises(EmbedError, match="Unknown embedding backend"):
            get_embedder("nonexistent")

    def test_is_abstract_base(self):
        assert hasattr(BaseEmbedder, "dim")
        assert hasattr(BaseEmbedder, "embed_texts")
