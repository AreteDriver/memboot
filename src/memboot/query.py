"""Similarity search across chunks and memories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from memboot.embedder import TfidfEmbedder, get_embedder
from memboot.exceptions import QueryError
from memboot.indexer import get_db_path
from memboot.models import SearchResult
from memboot.store import MembootStore


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors (= dot product)."""
    return float(np.dot(a, b))


def _restore_embedder(store: MembootStore):
    """Restore the embedder from stored state."""
    backend = store.get_meta("embedding_backend") or "tfidf"
    if backend == "tfidf":
        state_json = store.get_meta("tfidf_state")
        if state_json is None:
            raise QueryError("No TF-IDF state found. Run 'memboot init' first.")
        state = json.loads(state_json)
        return TfidfEmbedder.from_state(state)
    else:
        return get_embedder(backend)


def search(
    query_text: str,
    project_path: Path,
    top_k: int = 5,
    include_memories: bool = True,
) -> list[SearchResult]:
    """Search chunks and memories by similarity."""
    db_path = get_db_path(project_path.resolve())
    if not db_path.exists():
        raise QueryError(f"No index found for {project_path}. Run 'memboot init' first.")

    store = MembootStore(db_path)

    try:
        embedder = _restore_embedder(store)
        query_vec = embedder.embed_text(query_text)

        scored: list[tuple[str, float, str]] = []  # (id, score, type)

        # Score chunks
        for chunk_id, emb in store.get_all_chunk_embeddings():
            score = cosine_similarity(query_vec, emb)
            scored.append((chunk_id, score, "chunk"))

        # Score memories
        if include_memories:
            for mem_id, emb in store.get_all_memory_embeddings():
                score = cosine_similarity(query_vec, emb)
                scored.append((mem_id, score, "memory"))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        # Hydrate results
        results: list[SearchResult] = []
        for item_id, score, item_type in top:
            if item_type == "memory":
                mem = store.get_memory(item_id)
                if mem:
                    results.append(
                        SearchResult(
                            content=mem.content,
                            source=f"memory:{item_id}",
                            score=round(score, 4),
                        )
                    )
            else:
                chunk = store.get_chunk(item_id)
                if chunk:
                    results.append(
                        SearchResult(
                            content=chunk.content,
                            source=chunk.source_file,
                            score=round(score, 4),
                            chunk_type=chunk.chunk_type,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                        )
                    )

        return results
    finally:
        store.close()
