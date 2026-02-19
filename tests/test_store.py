"""Tests for memboot.store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from memboot.models import Chunk, ChunkType, Memory, MemoryType
from memboot.store import MembootStore


@pytest.fixture
def store(tmp_db_path: Path) -> MembootStore:
    s = MembootStore(tmp_db_path)
    yield s
    s.close()


def _make_chunk(
    chunk_id: str = "c1",
    content: str = "def foo(): pass",
    source_file: str = "foo.py",
    embedding: list[float] | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        content=content,
        source_file=source_file,
        start_line=1,
        end_line=5,
        chunk_type=ChunkType.FUNCTION,
        embedding=embedding or [0.1, 0.2, 0.3],
    )


def _make_memory(
    mem_id: str = "m1",
    content: str = "A note",
    memory_type: MemoryType = MemoryType.NOTE,
    embedding: list[float] | None = None,
    tags: list[str] | None = None,
) -> Memory:
    return Memory(
        id=mem_id,
        content=content,
        memory_type=memory_type,
        embedding=embedding or [0.4, 0.5, 0.6],
        tags=tags or [],
    )


class TestStoreInit:
    def test_creates_db_file(self, tmp_db_path: Path):
        store = MembootStore(tmp_db_path)
        assert tmp_db_path.exists()
        store.close()

    def test_creates_parent_dirs(self, tmp_path: Path):
        db_path = tmp_path / "sub" / "dir" / "test.db"
        store = MembootStore(db_path)
        assert db_path.exists()
        store.close()

    def test_wal_mode(self, store: MembootStore):
        conn = store._get_conn()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


class TestChunkOps:
    def test_add_and_count(self, store: MembootStore):
        chunks = [_make_chunk("c1"), _make_chunk("c2")]
        added = store.add_chunks(chunks)
        assert added == 2
        assert store.count_chunks() == 2

    def test_get_chunk_found(self, store: MembootStore):
        store.add_chunks([_make_chunk("c1", content="hello")])
        chunk = store.get_chunk("c1")
        assert chunk is not None
        assert chunk.content == "hello"
        assert chunk.chunk_type == ChunkType.FUNCTION

    def test_get_chunk_not_found(self, store: MembootStore):
        assert store.get_chunk("nonexistent") is None

    def test_get_chunks_by_file(self, store: MembootStore):
        store.add_chunks(
            [
                _make_chunk("c1", source_file="a.py"),
                _make_chunk("c2", source_file="a.py"),
                _make_chunk("c3", source_file="b.py"),
            ]
        )
        results = store.get_chunks_by_file("a.py")
        assert len(results) == 2

    def test_clear_chunks(self, store: MembootStore):
        store.add_chunks([_make_chunk("c1"), _make_chunk("c2")])
        deleted = store.clear_chunks()
        assert deleted == 2
        assert store.count_chunks() == 0

    def test_get_all_chunk_embeddings(self, store: MembootStore):
        emb = [1.0, 2.0, 3.0]
        store.add_chunks([_make_chunk("c1", embedding=emb)])
        results = store.get_all_chunk_embeddings()
        assert len(results) == 1
        chunk_id, arr = results[0]
        assert chunk_id == "c1"
        np.testing.assert_array_almost_equal(arr, np.array(emb, dtype=np.float32))

    def test_numpy_blob_roundtrip(self, store: MembootStore):
        original = [0.123456, 0.789012, 0.345678]
        store.add_chunks([_make_chunk("c1", embedding=original)])
        chunk = store.get_chunk("c1")
        assert chunk is not None
        assert chunk.embedding is not None
        np.testing.assert_array_almost_equal(chunk.embedding, original, decimal=5)

    def test_replace_on_duplicate_id(self, store: MembootStore):
        store.add_chunks([_make_chunk("c1", content="old")])
        store.add_chunks([_make_chunk("c1", content="new")])
        assert store.count_chunks() == 1
        chunk = store.get_chunk("c1")
        assert chunk is not None
        assert chunk.content == "new"

    def test_chunk_without_embedding(self, store: MembootStore):
        chunk = Chunk(
            id="c1",
            content="test",
            source_file="t.py",
            chunk_type=ChunkType.WINDOW,
            embedding=None,
        )
        store.add_chunks([chunk])
        result = store.get_chunk("c1")
        assert result is not None
        assert result.embedding is None


class TestMemoryOps:
    def test_add_and_get(self, store: MembootStore):
        store.add_memory(_make_memory("m1", content="A decision"))
        mem = store.get_memory("m1")
        assert mem is not None
        assert mem.content == "A decision"

    def test_get_memory_not_found(self, store: MembootStore):
        assert store.get_memory("nonexistent") is None

    def test_list_memories_all(self, store: MembootStore):
        store.add_memory(_make_memory("m1", memory_type=MemoryType.NOTE))
        store.add_memory(_make_memory("m2", memory_type=MemoryType.DECISION))
        mems = store.list_memories()
        assert len(mems) == 2

    def test_list_memories_filtered(self, store: MembootStore):
        store.add_memory(_make_memory("m1", memory_type=MemoryType.NOTE))
        store.add_memory(_make_memory("m2", memory_type=MemoryType.DECISION))
        notes = store.list_memories(MemoryType.NOTE)
        assert len(notes) == 1
        assert notes[0].memory_type == MemoryType.NOTE

    def test_delete_memory_exists(self, store: MembootStore):
        store.add_memory(_make_memory("m1"))
        assert store.delete_memory("m1") is True
        assert store.get_memory("m1") is None

    def test_delete_memory_not_exists(self, store: MembootStore):
        assert store.delete_memory("nonexistent") is False

    def test_count_and_clear_memories(self, store: MembootStore):
        store.add_memory(_make_memory("m1"))
        store.add_memory(_make_memory("m2"))
        assert store.count_memories() == 2
        deleted = store.clear_memories()
        assert deleted == 2
        assert store.count_memories() == 0

    def test_get_all_memory_embeddings(self, store: MembootStore):
        emb = [0.7, 0.8, 0.9]
        store.add_memory(_make_memory("m1", embedding=emb))
        results = store.get_all_memory_embeddings()
        assert len(results) == 1
        mem_id, arr = results[0]
        assert mem_id == "m1"
        np.testing.assert_array_almost_equal(arr, np.array(emb, dtype=np.float32))

    def test_memory_tags_roundtrip(self, store: MembootStore):
        store.add_memory(_make_memory("m1", tags=["a", "b"]))
        mem = store.get_memory("m1")
        assert mem is not None
        assert mem.tags == ["a", "b"]


class TestMetaOps:
    def test_set_and_get(self, store: MembootStore):
        store.set_meta("key1", "value1")
        assert store.get_meta("key1") == "value1"

    def test_get_not_found(self, store: MembootStore):
        assert store.get_meta("nonexistent") is None

    def test_overwrite(self, store: MembootStore):
        store.set_meta("key", "old")
        store.set_meta("key", "new")
        assert store.get_meta("key") == "new"


class TestLifecycle:
    def test_reset_clears_all(self, store: MembootStore):
        store.add_chunks([_make_chunk("c1")])
        store.add_memory(_make_memory("m1"))
        store.set_meta("k", "v")
        store.reset()
        assert store.count_chunks() == 0
        assert store.count_memories() == 0
        assert store.get_meta("k") is None

    def test_close_idempotent(self, tmp_db_path: Path):
        store = MembootStore(tmp_db_path)
        store.close()
        store.close()  # should not raise
