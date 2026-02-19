"""Web URL ingestion (Pro feature)."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from memboot.chunker import _chunk_window
from memboot.embedder import TfidfEmbedder, get_embedder
from memboot.exceptions import IngestError
from memboot.indexer import get_db_path
from memboot.models import Chunk, ChunkType, MembootConfig
from memboot.store import MembootStore


def ingest_url(
    url: str,
    project_path: Path,
    config: MembootConfig | None = None,
) -> list[Chunk]:
    """Ingest content from a URL into project memory.

    Fetches and extracts main content using trafilatura, chunks via sliding
    window, embeds, and stores. Requires Pro tier.
    """
    try:
        import trafilatura
    except ImportError as exc:
        raise IngestError(
            "Web ingestion requires trafilatura. Install with: pip install memboot[web]"
        ) from exc

    config = config or MembootConfig()

    # Fetch and extract content
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as exc:
        raise IngestError(f"Failed to fetch {url}: {exc}") from exc

    if downloaded is None:
        raise IngestError(f"Could not download content from {url}")

    text = trafilatura.extract(downloaded)
    if not text or not text.strip():
        raise IngestError(f"No extractable content from {url}")

    # Chunk the extracted text
    chunk_results = _chunk_window(text, config)

    if not chunk_results:
        return []

    chunks: list[Chunk] = []
    for result in chunk_results:
        chunk = Chunk(
            id=str(uuid4()),
            content=result.content,
            source_file=url,
            start_line=result.start_line,
            end_line=result.end_line,
            chunk_type=ChunkType.WINDOW,
        )
        chunks.append(chunk)

    # Embed and store
    db_path = get_db_path(project_path.resolve())
    store = MembootStore(db_path)

    try:
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
