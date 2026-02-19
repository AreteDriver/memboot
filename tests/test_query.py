"""Tests for memboot.query."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from memboot.embedder import TfidfEmbedder
from memboot.exceptions import QueryError
from memboot.models import Memory, MemoryType
from memboot.query import _restore_embedder, cosine_similarity, search
from memboot.store import MembootStore


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b)) < 1e-5

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) < 0


class TestRestoreEmbedder:
    def test_tfidf_with_state(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        store = MembootStore(db_path)
        emb = TfidfEmbedder(max_features=10)
        emb.fit(["hello world", "foo bar"])
        store.set_meta("embedding_backend", "tfidf")
        store.set_meta("tfidf_state", json.dumps(emb.save_state()))

        restored = _restore_embedder(store)
        assert isinstance(restored, TfidfEmbedder)
        store.close()

    def test_tfidf_missing_state(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        store = MembootStore(db_path)
        store.set_meta("embedding_backend", "tfidf")
        # No tfidf_state set

        with pytest.raises(QueryError, match="No TF-IDF state"):
            _restore_embedder(store)
        store.close()

    def test_default_backend(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        store = MembootStore(db_path)
        # No backend set — defaults to tfidf
        emb = TfidfEmbedder(max_features=10)
        emb.fit(["test"])
        store.set_meta("tfidf_state", json.dumps(emb.save_state()))
        restored = _restore_embedder(store)
        assert isinstance(restored, TfidfEmbedder)
        store.close()


class TestSearch:
    def _setup_indexed_project(self, tmp_path: Path) -> Path:
        """Create a project with indexed chunks and memories."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "test.py").write_text("def hello(): return 'hello'\n")

        from memboot.indexer import index_project

        index_project(project)
        return project

    def test_search_returns_results(self, tmp_path: Path, monkeypatch):
        db_dir = tmp_path / ".memboot"
        monkeypatch.setattr(
            "memboot.indexer.Path.expanduser",
            lambda self: db_dir if str(self) == "~/.memboot" else Path.home(),
        )
        monkeypatch.setattr(
            "memboot.query.get_db_path",
            lambda p: db_dir / f"{p.name}.db",
        )
        project = tmp_path / "proj"
        project.mkdir()
        (project / "test.py").write_text("def hello(): return 'hello'\n")

        from memboot.indexer import index_project

        # Patch indexer's get_db_path too
        with patch("memboot.indexer.get_db_path", lambda p: db_dir / f"{p.name}.db"):
            index_project(project)

        results = search("hello", project, top_k=5)
        assert len(results) >= 1
        assert all(r.score >= 0 for r in results)

    def test_no_index_raises(self, tmp_path: Path, monkeypatch):
        project = tmp_path / "no_index"
        project.mkdir()
        monkeypatch.setattr(
            "memboot.query.get_db_path",
            lambda p: tmp_path / "nonexistent.db",
        )
        with pytest.raises(QueryError, match="No index found"):
            search("test", project)

    def test_top_k_limiting(self, tmp_path: Path, monkeypatch):
        db_dir = tmp_path / ".memboot"
        monkeypatch.setattr(
            "memboot.query.get_db_path",
            lambda p: db_dir / f"{p.name}.db",
        )
        project = tmp_path / "proj"
        project.mkdir()
        # Create many chunks
        content = "\n\n".join(f"def func_{i}(): pass" for i in range(20))
        (project / "funcs.py").write_text(content)

        with patch("memboot.indexer.get_db_path", lambda p: db_dir / f"{p.name}.db"):
            from memboot.indexer import index_project

            index_project(project)

        results = search("func", project, top_k=3)
        assert len(results) <= 3

    def test_include_memories_false(self, tmp_path: Path, monkeypatch):
        db_dir = tmp_path / ".memboot"
        db_path = db_dir / "proj.db"
        monkeypatch.setattr(
            "memboot.query.get_db_path",
            lambda p: db_path,
        )
        project = tmp_path / "proj"
        project.mkdir()
        (project / "test.py").write_text("def hello(): pass\n")

        with patch("memboot.indexer.get_db_path", lambda p: db_path):
            from memboot.indexer import index_project

            index_project(project)

        # Add a memory
        store = MembootStore(db_path)
        emb_state = store.get_meta("tfidf_state")
        emb = TfidfEmbedder.from_state(json.loads(emb_state))
        vec = emb.embed_text("test memory")
        mem = Memory(
            id="m1",
            content="test memory",
            memory_type=MemoryType.NOTE,
            embedding=vec.tolist(),
        )
        store.add_memory(mem)
        store.close()

        results = search("test", project, include_memories=False)
        assert all(not r.source.startswith("memory:") for r in results)

    def test_search_includes_memories(self, tmp_path: Path, monkeypatch):
        db_dir = tmp_path / ".memboot"
        db_path = db_dir / "proj.db"
        monkeypatch.setattr("memboot.query.get_db_path", lambda p: db_path)
        project = tmp_path / "proj"
        project.mkdir()
        (project / "test.py").write_text("def hello(): pass\n")

        with patch("memboot.indexer.get_db_path", lambda p: db_path):
            from memboot.indexer import index_project

            index_project(project)

        # Add a memory with embedding
        store = MembootStore(db_path)
        emb_state = store.get_meta("tfidf_state")
        emb = TfidfEmbedder.from_state(json.loads(emb_state))
        vec = emb.embed_text("hello world note")
        mem = Memory(
            id="m1",
            content="hello world note",
            memory_type=MemoryType.NOTE,
            embedding=vec.tolist(),
        )
        store.add_memory(mem)
        store.close()

        results = search("hello", project, include_memories=True)
        sources = [r.source for r in results]
        assert any(s.startswith("memory:") for s in sources)
