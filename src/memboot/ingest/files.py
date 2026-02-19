"""Ingest supported file types into project memory."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from memboot.chunker import chunk_file
from memboot.embedder import TfidfEmbedder, get_embedder
from memboot.exceptions import IngestError
from memboot.indexer import get_db_path
from memboot.models import Chunk, MembootConfig
from memboot.store import MembootStore


def ingest_file(
    file_path: Path,
    project_path: Path,
    config: MembootConfig | None = None,
) -> list[Chunk]:
    """Ingest a single external file into the project memory."""
    config = config or MembootConfig()

    if not file_path.is_file():
        raise IngestError(f"File not found: {file_path}")

    if file_path.suffix.lower() not in config.file_extensions:
        raise IngestError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported: {', '.join(config.file_extensions)}"
        )

    db_path = get_db_path(project_path.resolve())
    store = MembootStore(db_path)

    try:
        results = chunk_file(file_path, config)
        if not results:
            return []

        chunks: list[Chunk] = []
        for result in results:
            chunk = Chunk(
                id=str(uuid4()),
                content=result.content,
                source_file=str(file_path),
                start_line=result.start_line,
                end_line=result.end_line,
                chunk_type=result.chunk_type,
            )
            chunks.append(chunk)

        # Embed
        backend = store.get_meta("embedding_backend") or "tfidf"
        if backend == "tfidf":
            state_json = store.get_meta("tfidf_state")
            if state_json:
                embedder = TfidfEmbedder.from_state(json.loads(state_json))
            else:
                embedder = TfidfEmbedder()
                embedder.fit([c.content for c in chunks])
        else:
            embedder = get_embedder(backend)

        embeddings = embedder.embed_texts([c.content for c in chunks])
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i].tolist()

        store.add_chunks(chunks)
        return chunks
    finally:
        store.close()
