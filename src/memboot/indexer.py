"""Project indexing pipeline: discover -> chunk -> embed -> store."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from uuid import uuid4

from memboot.chunker import chunk_file
from memboot.embedder import TfidfEmbedder, get_embedder
from memboot.exceptions import IndexingError
from memboot.models import Chunk, MembootConfig, ProjectInfo
from memboot.store import MembootStore


def compute_project_hash(project_path: Path) -> str:
    """Hash a project path to a short identifier."""
    return hashlib.sha256(str(project_path.resolve()).encode()).hexdigest()[:12]


def get_db_path(project_path: Path) -> Path:
    """Derive the SQLite database path for a project."""
    memboot_home = Path("~/.memboot").expanduser()
    memboot_home.mkdir(parents=True, exist_ok=True)
    return memboot_home / f"{compute_project_hash(project_path)}.db"


def _should_ignore(path: Path, ignore_patterns: list[str]) -> bool:
    """Check if a path matches any ignore pattern."""
    for pattern in ignore_patterns:
        if fnmatch(path.name, pattern):
            return True
        for parent in path.parents:
            if fnmatch(parent.name, pattern):
                return True
    return False


def discover_files(project_path: Path, config: MembootConfig) -> list[Path]:
    """Walk project directory and collect indexable files."""
    files: list[Path] = []
    for path in sorted(project_path.rglob("*")):
        if not path.is_file():
            continue
        if _should_ignore(path.relative_to(project_path), config.ignore_patterns):
            continue
        if path.suffix.lower() in config.file_extensions:
            files.append(path)
    return files


def index_project(
    project_path: Path,
    config: MembootConfig | None = None,
    force: bool = False,
) -> ProjectInfo:
    """Full indexing pipeline: discover -> chunk -> embed -> store."""
    config = config or MembootConfig()
    project_path = project_path.resolve()

    if not project_path.is_dir():
        raise IndexingError(f"Not a directory: {project_path}")

    db_path = get_db_path(project_path)
    store = MembootStore(db_path)

    if force:
        store.clear_chunks()

    # Discover files
    files = discover_files(project_path, config)
    if not files:
        store.close()
        return ProjectInfo(
            project_path=str(project_path),
            project_hash=compute_project_hash(project_path),
            db_path=str(db_path),
        )

    # Chunk all files
    all_chunks: list[Chunk] = []
    for file_path in files:
        results = chunk_file(file_path, config)
        for result in results:
            chunk = Chunk(
                id=str(uuid4()),
                content=result.content,
                source_file=str(file_path.relative_to(project_path)),
                start_line=result.start_line,
                end_line=result.end_line,
                chunk_type=result.chunk_type,
            )
            all_chunks.append(chunk)

    if not all_chunks:
        store.close()
        return ProjectInfo(
            project_path=str(project_path),
            project_hash=compute_project_hash(project_path),
            db_path=str(db_path),
        )

    # Embed
    embedder = get_embedder(config.embedding_backend, max_features=config.max_features)
    if isinstance(embedder, TfidfEmbedder):
        embedder.fit([c.content for c in all_chunks])
        store.set_meta("tfidf_state", json.dumps(embedder.save_state()))

    embeddings = embedder.embed_texts([c.content for c in all_chunks])
    for i, chunk in enumerate(all_chunks):
        chunk.embedding = embeddings[i].tolist()

    # Store
    store.add_chunks(all_chunks)
    store.set_meta("embedding_dim", str(embedder.dim))
    store.set_meta("embedding_backend", config.embedding_backend)
    store.set_meta("last_indexed", datetime.now(UTC).isoformat())
    store.set_meta("project_path", str(project_path))

    info = ProjectInfo(
        project_path=str(project_path),
        project_hash=compute_project_hash(project_path),
        db_path=str(db_path),
        chunk_count=store.count_chunks(),
        memory_count=store.count_memories(),
        last_indexed=datetime.now(UTC).isoformat(),
        embedding_dim=embedder.dim,
        embedding_backend=config.embedding_backend,
    )

    store.close()
    return info
