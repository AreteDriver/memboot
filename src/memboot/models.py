"""Pydantic v2 models for memboot."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ChunkType(StrEnum):
    """Chunk origin type."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    MARKDOWN_SECTION = "markdown_section"
    YAML_KEY = "yaml_key"
    JSON_KEY = "json_key"
    WINDOW = "window"


class MemoryType(StrEnum):
    """Episodic memory type."""

    DECISION = "decision"
    NOTE = "note"
    OBSERVATION = "observation"
    PATTERN = "pattern"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A single chunk of indexed content."""

    id: str
    content: str
    source_file: str
    start_line: int | None = None
    end_line: int | None = None
    chunk_type: ChunkType
    embedding: list[float] | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class Memory(BaseModel):
    """An episodic memory entry."""

    id: str
    content: str
    memory_type: MemoryType
    embedding: list[float] | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class SearchResult(BaseModel):
    """A single search result with score."""

    content: str
    source: str
    score: float
    chunk_type: ChunkType | None = None
    start_line: int | None = None
    end_line: int | None = None


class ProjectInfo(BaseModel):
    """Metadata about an indexed project."""

    project_path: str
    project_hash: str
    db_path: str
    chunk_count: int = 0
    memory_count: int = 0
    last_indexed: str | None = None
    embedding_dim: int = 0
    embedding_backend: str = "tfidf"


class MembootConfig(BaseModel):
    """Configuration for indexing and querying."""

    max_chunk_tokens: int = 500
    overlap_tokens: int = 50
    embedding_backend: str = "tfidf"
    max_features: int = 512
    file_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py",
            ".md",
            ".yaml",
            ".yml",
            ".json",
            ".txt",
            ".toml",
            ".cfg",
            ".ini",
            ".rst",
        ]
    )
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "venv",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            ".eggs",
            "*.egg-info",
            "dist",
            "build",
            ".tox",
        ]
    )
