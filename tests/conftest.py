"""Shared fixtures for memboot tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from memboot.models import Chunk, ChunkType, MembootConfig, Memory, MemoryType


@pytest.fixture
def sample_config() -> MembootConfig:
    """Default MembootConfig."""
    return MembootConfig()


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Temp SQLite database path."""
    return tmp_path / "test.db"


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """Temp project directory with sample files."""
    project = tmp_path / "project"
    project.mkdir()

    # Sample Python file
    (project / "main.py").write_text(
        'def hello():\n    """Say hello."""\n    return "hello"\n\n\n'
        'def goodbye():\n    """Say goodbye."""\n    return "goodbye"\n'
    )

    # Sample Markdown file
    (project / "README.md").write_text(
        "# My Project\n\nA sample project.\n\n## Usage\n\nRun the thing.\n"
    )

    # Sample YAML file
    (project / "config.yaml").write_text("name: test\nversion: 1\nsettings:\n  debug: true\n")

    # Sample JSON file
    (project / "data.json").write_text('{"key1": "value1", "key2": [1, 2, 3]}\n')

    # Sample text file
    (project / "notes.txt").write_text("Some notes here.\nMore notes.\n")

    return project


@pytest.fixture
def sample_chunk() -> Chunk:
    """Pre-built Chunk instance."""
    return Chunk(
        id="chunk-001",
        content="def foo(): pass",
        source_file="src/foo.py",
        start_line=1,
        end_line=1,
        chunk_type=ChunkType.FUNCTION,
        embedding=[0.1, 0.2, 0.3],
    )


@pytest.fixture
def sample_memory() -> Memory:
    """Pre-built Memory instance."""
    return Memory(
        id="mem-001",
        content="Always use pytest fixtures for test isolation.",
        memory_type=MemoryType.PATTERN,
        embedding=[0.4, 0.5, 0.6],
        tags=["testing", "patterns"],
    )


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Sample L2-normalized embedding vector."""
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return vec / np.linalg.norm(vec)
