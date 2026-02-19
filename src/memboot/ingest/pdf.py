"""PDF ingestion (Pro feature)."""

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


def ingest_pdf(
    file_path: Path,
    project_path: Path,
    config: MembootConfig | None = None,
) -> list[Chunk]:
    """Ingest a PDF file into project memory.

    Extracts text page-by-page using pdfplumber, chunks via sliding window,
    embeds, and stores. Requires Pro tier.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise IngestError(
            "PDF ingestion requires pdfplumber. Install with: pip install memboot[pdf]"
        ) from exc

    config = config or MembootConfig()

    if not file_path.is_file():
        raise IngestError(f"File not found: {file_path}")

    # Extract text from PDF
    pages_text: list[str] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(text)
    except Exception as exc:
        raise IngestError(f"Failed to read PDF {file_path}: {exc}") from exc

    if not pages_text:
        return []

    # Chunk the combined text
    full_text = "\n\n".join(pages_text)
    chunk_results = _chunk_window(full_text, config)

    if not chunk_results:
        return []

    chunks: list[Chunk] = []
    for result in chunk_results:
        chunk = Chunk(
            id=str(uuid4()),
            content=result.content,
            source_file=str(file_path),
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
