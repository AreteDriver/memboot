"""Episodic memory CRUD operations."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from memboot.embedder import TfidfEmbedder, get_embedder
from memboot.indexer import get_db_path
from memboot.models import Memory, MemoryType
from memboot.store import MembootStore


def _restore_embedder(store: MembootStore):
    """Restore embedder from store metadata."""
    backend = store.get_meta("embedding_backend") or "tfidf"
    if backend == "tfidf":
        state_json = store.get_meta("tfidf_state")
        if state_json:
            return TfidfEmbedder.from_state(json.loads(state_json))
    return get_embedder(backend)


def remember(
    content: str,
    memory_type: MemoryType,
    project_path: Path,
    tags: list[str] | None = None,
) -> Memory:
    """Store an episodic memory."""
    db_path = get_db_path(project_path.resolve())
    store = MembootStore(db_path)
    try:
        embedder = _restore_embedder(store)
        # For TF-IDF without state, fit on just this text (won't be great but works)
        if isinstance(embedder, TfidfEmbedder) and not embedder._fitted:
            embedder.fit([content])
        embedding = embedder.embed_text(content)
        memory = Memory(
            id=str(uuid4()),
            content=content,
            memory_type=memory_type,
            embedding=embedding.tolist(),
            tags=tags or [],
        )
        store.add_memory(memory)
        return memory
    finally:
        store.close()


def list_memories(
    project_path: Path,
    memory_type: MemoryType | None = None,
) -> list[Memory]:
    """List all memories for a project."""
    db_path = get_db_path(project_path.resolve())
    if not db_path.exists():
        return []
    store = MembootStore(db_path)
    try:
        return store.list_memories(memory_type)
    finally:
        store.close()


def delete_memory(memory_id: str, project_path: Path) -> bool:
    """Delete a memory by ID."""
    db_path = get_db_path(project_path.resolve())
    if not db_path.exists():
        return False
    store = MembootStore(db_path)
    try:
        return store.delete_memory(memory_id)
    finally:
        store.close()
