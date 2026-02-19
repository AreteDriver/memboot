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


def _categorize_files(
    files: list[Path],
    project_path: Path,
    stored_meta: dict[str, tuple[float, int, int]],
) -> tuple[list[Path], list[Path], list[Path], list[str]]:
    """Categorize discovered files against stored metadata.

    Returns (unchanged, changed, new, deleted_rel_paths).
    """
    unchanged: list[Path] = []
    changed: list[Path] = []
    new: list[Path] = []
    seen: set[str] = set()

    for file_path in files:
        rel = str(file_path.relative_to(project_path))
        seen.add(rel)
        stat = file_path.stat()

        if rel in stored_meta:
            stored_mtime, stored_size, _ = stored_meta[rel]
            if stat.st_mtime == stored_mtime and stat.st_size == stored_size:
                unchanged.append(file_path)
            else:
                changed.append(file_path)
        else:
            new.append(file_path)

    deleted = [p for p in stored_meta if p not in seen]
    return unchanged, changed, new, deleted


def index_project(
    project_path: Path,
    config: MembootConfig | None = None,
    force: bool = False,
) -> ProjectInfo:
    """Indexing pipeline with incremental support.

    On first run or with force=True, indexes everything. On subsequent runs,
    only re-chunks files whose mtime or size changed.
    """
    config = config or MembootConfig()
    project_path = project_path.resolve()

    if not project_path.is_dir():
        raise IndexingError(f"Not a directory: {project_path}")

    db_path = get_db_path(project_path)
    store = MembootStore(db_path)

    if force:
        store.clear_chunks()  # Also clears file_meta

    # Discover files
    files = discover_files(project_path, config)
    if not files:
        store.close()
        return ProjectInfo(
            project_path=str(project_path),
            project_hash=compute_project_hash(project_path),
            db_path=str(db_path),
        )

    # Categorize files against stored metadata
    stored_meta = store.get_all_file_meta()

    if force or not stored_meta:
        # First run or forced: treat everything as new
        files_to_process = files
        deleted_files: list[str] = []
        unchanged_count = 0
        changed_count = 0
        new_count = len(files)
    else:
        unchanged, changed, new_files, deleted_files = _categorize_files(
            files,
            project_path,
            stored_meta,
        )
        files_to_process = changed + new_files
        unchanged_count = len(unchanged)
        changed_count = len(changed)
        new_count = len(new_files)

        # Remove chunks for changed + deleted files
        for rel_path in deleted_files:
            store.delete_chunks_by_file(rel_path)
            store.delete_file_meta(rel_path)
        for file_path in changed:
            rel = str(file_path.relative_to(project_path))
            store.delete_chunks_by_file(rel)

    # Nothing to process — everything unchanged
    if not files_to_process:
        info = ProjectInfo(
            project_path=str(project_path),
            project_hash=compute_project_hash(project_path),
            db_path=str(db_path),
            chunk_count=store.count_chunks(),
            memory_count=store.count_memories(),
            last_indexed=store.get_meta("last_indexed"),
            embedding_dim=int(store.get_meta("embedding_dim") or 0),
            embedding_backend=config.embedding_backend,
            metadata={
                "unchanged_files": unchanged_count,
                "changed_files": 0,
                "new_files": 0,
                "deleted_files": len(deleted_files),
                "new_chunks": 0,
            },
        )
        store.close()
        return info

    # Chunk only files that need processing
    new_chunks: list[Chunk] = []
    file_chunk_counts: dict[str, int] = {}
    for file_path in files_to_process:
        rel = str(file_path.relative_to(project_path))
        results = chunk_file(file_path, config)
        count = 0
        for result in results:
            chunk = Chunk(
                id=str(uuid4()),
                content=result.content,
                source_file=rel,
                start_line=result.start_line,
                end_line=result.end_line,
                chunk_type=result.chunk_type,
            )
            new_chunks.append(chunk)
            count += 1
        file_chunk_counts[rel] = count

    # Embed new chunks
    embedder = get_embedder(config.embedding_backend, max_features=config.max_features)
    if new_chunks:
        if isinstance(embedder, TfidfEmbedder):
            stored_state = store.get_meta("tfidf_state")
            if force or not stored_state:
                # First run or forced: fit on all content
                embedder.fit([c.content for c in new_chunks])
                store.set_meta("tfidf_state", json.dumps(embedder.save_state()))
            else:
                # Incremental: restore vocabulary, new terms get zero weight
                embedder = TfidfEmbedder.from_state(json.loads(stored_state))

        embeddings = embedder.embed_texts([c.content for c in new_chunks])
        for i, chunk in enumerate(new_chunks):
            chunk.embedding = embeddings[i].tolist()

    # Store new chunks
    store.add_chunks(new_chunks)

    # Update file metadata for all processed files
    for file_path in files_to_process:
        rel = str(file_path.relative_to(project_path))
        stat = file_path.stat()
        store.set_file_meta(rel, stat.st_mtime, stat.st_size, file_chunk_counts.get(rel, 0))

    # Update store metadata
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
        metadata={
            "unchanged_files": unchanged_count,
            "changed_files": changed_count,
            "new_files": new_count,
            "deleted_files": len(deleted_files),
            "new_chunks": len(new_chunks),
        },
    )

    store.close()
    return info
