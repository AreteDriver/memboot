"""Tests for memboot.indexer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from memboot.exceptions import IndexingError
from memboot.indexer import (
    _should_ignore,
    compute_project_hash,
    discover_files,
    get_db_path,
    index_project,
)
from memboot.models import MembootConfig


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
