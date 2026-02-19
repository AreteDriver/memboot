"""Tests for memboot.memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from memboot.memory import delete_memory, list_memories, remember
from memboot.models import MemoryType


@pytest.fixture
def indexed_project(tmp_path: Path, monkeypatch):
    """Create and index a project, return (project_path, db_path)."""
    db_dir = tmp_path / ".memboot"
    db_path = db_dir / "proj.db"

    monkeypatch.setattr(
        "memboot.indexer.get_db_path",
        lambda p: db_path,
    )
    monkeypatch.setattr(
        "memboot.memory.get_db_path",
        lambda p: db_path,
    )

    project = tmp_path / "proj"
    project.mkdir()
    (project / "test.py").write_text("def hello(): return 'hello'\n")

    from memboot.indexer import index_project

    index_project(project)
    return project, db_path


class TestRemember:
    def test_stores_memory(self, indexed_project):
        project, db_path = indexed_project
        mem = remember("Important decision", MemoryType.DECISION, project)
        assert mem.id
        assert mem.content == "Important decision"
        assert mem.memory_type == MemoryType.DECISION

    def test_embedding_attached(self, indexed_project):
        project, db_path = indexed_project
        mem = remember("Test note", MemoryType.NOTE, project)
        assert mem.embedding is not None
        assert len(mem.embedding) > 0

    def test_tags_preserved(self, indexed_project):
        project, db_path = indexed_project
        mem = remember("Tagged note", MemoryType.NOTE, project, tags=["arch", "db"])
        assert mem.tags == ["arch", "db"]

    def test_without_existing_state(self, tmp_path: Path, monkeypatch):
        """When no TF-IDF state exists, should fit on just the content."""
        db_dir = tmp_path / ".memboot"
        db_path = db_dir / "proj.db"
        monkeypatch.setattr(
            "memboot.memory.get_db_path",
            lambda p: db_path,
        )
        project = tmp_path / "proj"
        project.mkdir()

        # Store is created fresh, no tfidf_state
        mem = remember("A note without prior indexing", MemoryType.NOTE, project)
        assert mem.embedding is not None


class TestListMemories:
    def test_returns_all(self, indexed_project):
        project, db_path = indexed_project
        remember("Note 1", MemoryType.NOTE, project)
        remember("Decision 1", MemoryType.DECISION, project)
        mems = list_memories(project)
        assert len(mems) == 2

    def test_filtered_by_type(self, indexed_project):
        project, db_path = indexed_project
        remember("Note 1", MemoryType.NOTE, project)
        remember("Decision 1", MemoryType.DECISION, project)
        notes = list_memories(project, MemoryType.NOTE)
        assert len(notes) == 1
        assert notes[0].memory_type == MemoryType.NOTE

    def test_empty_project(self, tmp_path: Path, monkeypatch):
        db_path = tmp_path / "nonexistent.db"
        monkeypatch.setattr(
            "memboot.memory.get_db_path",
            lambda p: db_path,
        )
        mems = list_memories(tmp_path)
        assert mems == []


class TestDeleteMemory:
    def test_existing_memory(self, indexed_project):
        project, db_path = indexed_project
        mem = remember("To delete", MemoryType.NOTE, project)
        result = delete_memory(mem.id, project)
        assert result is True

    def test_nonexistent_memory(self, indexed_project):
        project, db_path = indexed_project
        result = delete_memory("nonexistent-id", project)
        assert result is False

    def test_missing_project(self, tmp_path: Path, monkeypatch):
        db_path = tmp_path / "nonexistent.db"
        monkeypatch.setattr(
            "memboot.memory.get_db_path",
            lambda p: db_path,
        )
        result = delete_memory("any-id", tmp_path)
        assert result is False
