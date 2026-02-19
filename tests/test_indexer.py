"""Tests for memboot.indexer."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from memboot.exceptions import IndexingError
from memboot.indexer import (
    _categorize_files,
    _should_ignore,
    compute_project_hash,
    discover_files,
    get_db_path,
    index_project,
)
from memboot.models import MembootConfig
from memboot.store import MembootStore


class TestComputeProjectHash:
    def test_deterministic(self, tmp_path: Path):
        h1 = compute_project_hash(tmp_path)
        h2 = compute_project_hash(tmp_path)
        assert h1 == h2

    def test_different_paths(self, tmp_path: Path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        assert compute_project_hash(a) != compute_project_hash(b)

    def test_short_hash(self, tmp_path: Path):
        h = compute_project_hash(tmp_path)
        assert len(h) == 12
        assert h.isalnum()


class TestGetDbPath:
    def test_returns_path_in_memboot_home(self, tmp_path: Path):
        with patch("memboot.indexer.Path.expanduser", return_value=tmp_path / ".memboot"):
            db = get_db_path(tmp_path / "project")
            assert ".memboot" in str(db.parent.name) or str(db).endswith(".db")

    def test_db_extension(self, tmp_path: Path):
        with patch("memboot.indexer.Path.expanduser", return_value=tmp_path / ".memboot"):
            db = get_db_path(tmp_path)
            assert str(db).endswith(".db")


class TestShouldIgnore:
    def test_matches_name_pattern(self):
        assert _should_ignore(Path("__pycache__/foo.pyc"), ["__pycache__"])

    def test_matches_parent_pattern(self):
        assert _should_ignore(Path(".git/objects/abc"), [".git"])

    def test_no_match(self):
        assert not _should_ignore(Path("src/main.py"), ["__pycache__", ".git"])

    def test_glob_pattern(self):
        assert _should_ignore(Path("mypackage.egg-info/PKG-INFO"), ["*.egg-info"])


class TestDiscoverFiles:
    def test_finds_matching_extensions(self, tmp_project_dir: Path):
        config = MembootConfig()
        files = discover_files(tmp_project_dir, config)
        extensions = {f.suffix for f in files}
        assert ".py" in extensions
        assert ".md" in extensions

    def test_skips_ignored_patterns(self, tmp_path: Path):
        project = tmp_path / "proj"
        project.mkdir()
        (project / "good.py").write_text("x = 1\n")
        pycache = project / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.py").write_text("cached\n")

        config = MembootConfig()
        files = discover_files(project, config)
        assert all("__pycache__" not in str(f) for f in files)

    def test_sorted_output(self, tmp_project_dir: Path):
        config = MembootConfig()
        files = discover_files(tmp_project_dir, config)
        assert files == sorted(files)

    def test_empty_dir(self, tmp_path: Path):
        project = tmp_path / "empty"
        project.mkdir()
        config = MembootConfig()
        files = discover_files(project, config)
        assert files == []


class TestIndexProject:
    def test_full_pipeline(self, tmp_project_dir: Path, monkeypatch):
        # Redirect db storage to temp dir
        db_dir = tmp_project_dir.parent / ".memboot"
        monkeypatch.setattr(
            "memboot.indexer.Path.expanduser",
            lambda self: db_dir if str(self) == "~/.memboot" else Path.home(),
        )
        info = index_project(tmp_project_dir)
        assert info.chunk_count > 0
        assert info.embedding_backend == "tfidf"
        assert info.embedding_dim > 0

    def test_not_a_directory(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(IndexingError, match="Not a directory"):
            index_project(f)

    def test_empty_project(self, tmp_path: Path, monkeypatch):
        project = tmp_path / "empty"
        project.mkdir()
        db_dir = tmp_path / ".memboot"
        monkeypatch.setattr(
            "memboot.indexer.Path.expanduser",
            lambda self: db_dir if str(self) == "~/.memboot" else Path.home(),
        )
        info = index_project(project)
        assert info.chunk_count == 0

    def test_force_reindex(self, tmp_project_dir: Path, monkeypatch):
        db_dir = tmp_project_dir.parent / ".memboot"
        monkeypatch.setattr(
            "memboot.indexer.Path.expanduser",
            lambda self: db_dir if str(self) == "~/.memboot" else Path.home(),
        )
        index_project(tmp_project_dir)
        info2 = index_project(tmp_project_dir, force=True)
        assert info2.chunk_count > 0

    def test_metadata_on_first_run(self, tmp_project_dir: Path, monkeypatch):
        db_dir = tmp_project_dir.parent / ".memboot"
        monkeypatch.setattr(
            "memboot.indexer.Path.expanduser",
            lambda self: db_dir if str(self) == "~/.memboot" else Path.home(),
        )
        info = index_project(tmp_project_dir)
        assert info.metadata["new_files"] > 0
        assert info.metadata["changed_files"] == 0
        assert info.metadata["unchanged_files"] == 0
        assert info.metadata["deleted_files"] == 0
        assert info.metadata["new_chunks"] == info.chunk_count


class TestCategorizeFiles:
    def test_all_new(self, tmp_path: Path):
        project = tmp_path / "proj"
        project.mkdir()
        (project / "a.py").write_text("x = 1\n")
        files = [project / "a.py"]
        unchanged, changed, new, deleted = _categorize_files(files, project, {})
        assert len(new) == 1
        assert len(unchanged) == 0
        assert len(changed) == 0
        assert len(deleted) == 0

    def test_unchanged(self, tmp_path: Path):
        project = tmp_path / "proj"
        project.mkdir()
        f = project / "a.py"
        f.write_text("x = 1\n")
        stat = f.stat()
        stored = {"a.py": (stat.st_mtime, stat.st_size, 3)}
        unchanged, changed, new, deleted = _categorize_files([f], project, stored)
        assert len(unchanged) == 1
        assert len(changed) == 0
        assert len(new) == 0

    def test_changed_mtime(self, tmp_path: Path):
        project = tmp_path / "proj"
        project.mkdir()
        f = project / "a.py"
        f.write_text("x = 1\n")
        stat = f.stat()
        stored = {"a.py": (stat.st_mtime - 100, stat.st_size, 3)}
        unchanged, changed, new, deleted = _categorize_files([f], project, stored)
        assert len(changed) == 1
        assert len(unchanged) == 0

    def test_changed_size(self, tmp_path: Path):
        project = tmp_path / "proj"
        project.mkdir()
        f = project / "a.py"
        f.write_text("x = 1\n")
        stat = f.stat()
        stored = {"a.py": (stat.st_mtime, stat.st_size + 50, 3)}
        unchanged, changed, new, deleted = _categorize_files([f], project, stored)
        assert len(changed) == 1

    def test_deleted(self, tmp_path: Path):
        project = tmp_path / "proj"
        project.mkdir()
        stored = {"old.py": (1000.0, 100, 5)}
        unchanged, changed, new, deleted = _categorize_files([], project, stored)
        assert deleted == ["old.py"]

    def test_mixed(self, tmp_path: Path):
        project = tmp_path / "proj"
        project.mkdir()
        f_unchanged = project / "keep.py"
        f_unchanged.write_text("x\n")
        f_new = project / "new.py"
        f_new.write_text("y\n")
        stat = f_unchanged.stat()
        stored = {
            "keep.py": (stat.st_mtime, stat.st_size, 2),
            "gone.py": (500.0, 50, 1),
        }
        unchanged, changed, new, deleted = _categorize_files(
            [f_unchanged, f_new],
            project,
            stored,
        )
        assert len(unchanged) == 1
        assert len(new) == 1
        assert deleted == ["gone.py"]


class TestIncrementalReindex:
    """Integration tests for incremental reindexing."""

    @pytest.fixture
    def _db_dir(self, tmp_path, monkeypatch):
        """Redirect memboot DB storage to temp dir."""
        db_dir = tmp_path / ".memboot"
        monkeypatch.setattr(
            "memboot.indexer.Path.expanduser",
            lambda self: db_dir if str(self) == "~/.memboot" else Path.home(),
        )
        return db_dir

    def test_second_run_no_changes(self, tmp_project_dir: Path, _db_dir):
        info1 = index_project(tmp_project_dir)
        assert info1.chunk_count > 0

        info2 = index_project(tmp_project_dir)
        assert info2.chunk_count == info1.chunk_count
        assert info2.metadata["new_chunks"] == 0
        assert info2.metadata["unchanged_files"] > 0
        assert info2.metadata["changed_files"] == 0

    def test_modified_file_reindexed(self, tmp_project_dir: Path, _db_dir):
        index_project(tmp_project_dir)

        # Modify a file — ensure mtime changes
        f = tmp_project_dir / "main.py"
        time.sleep(0.05)
        f.write_text('def hello():\n    return "hi"\n\ndef extra():\n    return "more"\n')

        info2 = index_project(tmp_project_dir)
        assert info2.metadata["changed_files"] == 1
        assert info2.metadata["new_chunks"] > 0

    def test_new_file_indexed(self, tmp_project_dir: Path, _db_dir):
        info1 = index_project(tmp_project_dir)

        # Add a new file
        (tmp_project_dir / "extra.py").write_text('def extra():\n    return "new"\n')

        info2 = index_project(tmp_project_dir)
        assert info2.metadata["new_files"] == 1
        assert info2.chunk_count > info1.chunk_count

    def test_deleted_file_chunks_removed(self, tmp_project_dir: Path, _db_dir):
        info1 = index_project(tmp_project_dir)

        # Delete a file
        (tmp_project_dir / "notes.txt").unlink()

        info2 = index_project(tmp_project_dir)
        assert info2.metadata["deleted_files"] == 1
        assert info2.chunk_count < info1.chunk_count

    def test_force_reindexes_everything(self, tmp_project_dir: Path, _db_dir):
        index_project(tmp_project_dir)

        info2 = index_project(tmp_project_dir, force=True)
        assert info2.chunk_count > 0
        # Force treats all as new
        assert info2.metadata["new_files"] > 0
        assert info2.metadata["unchanged_files"] == 0

    def test_file_meta_stored(self, tmp_project_dir: Path, _db_dir):
        index_project(tmp_project_dir)
        from memboot.indexer import get_db_path

        store = MembootStore(get_db_path(tmp_project_dir))
        meta = store.get_all_file_meta()
        store.close()
        assert len(meta) > 0
        for _, (mtime, size, _count) in meta.items():
            assert mtime > 0
            assert size > 0

    def test_force_clears_file_meta(self, tmp_project_dir: Path, _db_dir):
        index_project(tmp_project_dir)
        index_project(tmp_project_dir, force=True)
        from memboot.indexer import get_db_path

        store = MembootStore(get_db_path(tmp_project_dir))
        meta = store.get_all_file_meta()
        store.close()
        # After force, file_meta is repopulated for all files
        assert len(meta) > 0
