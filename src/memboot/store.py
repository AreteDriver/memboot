"""SQLite-backed vector store for memboot."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from memboot.exceptions import StoreError
from memboot.models import Chunk, ChunkType, Memory, MemoryType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_file TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    chunk_type TEXT,
    embedding BLOB,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    embedding BLOB,
    tags TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS file_meta (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0
);
"""


class MembootStore:
    """SQLite store with numpy embedding support."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    # -- Chunk operations --

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Insert chunks. Returns count added."""
        conn = self._get_conn()
        added = 0
        for chunk in chunks:
            emb_blob = (
                np.array(chunk.embedding, dtype=np.float32).tobytes() if chunk.embedding else None
            )
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO chunks "
                    "(id, content, source_file, start_line, end_line, "
                    "chunk_type, embedding, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.id,
                        chunk.content,
                        chunk.source_file,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.chunk_type.value if chunk.chunk_type else None,
                        emb_blob,
                        chunk.created_at,
                    ),
                )
                added += 1
            except sqlite3.Error as exc:
                raise StoreError(f"Failed to add chunk {chunk.id}: {exc}") from exc
        conn.commit()
        return added

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a chunk by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def get_chunks_by_file(self, source_file: str) -> list[Chunk]:
        """Retrieve all chunks from a specific file."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM chunks WHERE source_file = ?", (source_file,)).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def count_chunks(self) -> int:
        """Count total chunks."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def clear_chunks(self) -> int:
        """Delete all chunks and file metadata. Returns chunk count deleted."""
        count = self.count_chunks()
        conn = self._get_conn()
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM file_meta")
        conn.commit()
        return count

    def get_all_chunk_embeddings(self) -> list[tuple[str, np.ndarray]]:
        """Load all chunk embeddings for vector search."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        results = []
        for chunk_id, emb_blob in rows:
            arr = np.frombuffer(emb_blob, dtype=np.float32)
            results.append((chunk_id, arr))
        return results

    # -- Memory operations --

    def add_memory(self, memory: Memory) -> None:
        """Insert a single memory."""
        conn = self._get_conn()
        emb_blob = (
            np.array(memory.embedding, dtype=np.float32).tobytes() if memory.embedding else None
        )
        tags_json = json.dumps(memory.tags) if memory.tags else "[]"
        try:
            conn.execute(
                "INSERT OR REPLACE INTO memories "
                "(id, content, memory_type, embedding, tags, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    memory.id,
                    memory.content,
                    memory.memory_type.value,
                    emb_blob,
                    tags_json,
                    memory.created_at,
                ),
            )
            conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"Failed to add memory {memory.id}: {exc}") from exc

    def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def list_memories(self, memory_type: MemoryType | None = None) -> list[Memory]:
        """List all memories, optionally filtered by type."""
        conn = self._get_conn()
        if memory_type:
            rows = conn.execute(
                "SELECT * FROM memories WHERE memory_type = ? ORDER BY created_at DESC",
                (memory_type.value,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM memories ORDER BY created_at DESC").fetchall()
        return [self._row_to_memory(r) for r in rows]

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory. Returns True if found and deleted."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        return cursor.rowcount > 0

    def count_memories(self) -> int:
        """Count total memories."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    def clear_memories(self) -> int:
        """Delete all memories. Returns count deleted."""
        count = self.count_memories()
        conn = self._get_conn()
        conn.execute("DELETE FROM memories")
        conn.commit()
        return count

    def get_all_memory_embeddings(self) -> list[tuple[str, np.ndarray]]:
        """Load all memory embeddings for vector search."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()
        results = []
        for mem_id, emb_blob in rows:
            arr = np.frombuffer(emb_blob, dtype=np.float32)
            results.append((mem_id, arr))
        return results

    # -- Meta operations --

    def set_meta(self, key: str, value: str) -> None:
        """Set a metadata key-value pair."""
        conn = self._get_conn()
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        conn.commit()

    def get_meta(self, key: str) -> str | None:
        """Get a metadata value by key."""
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    # -- File meta operations --

    def set_file_meta(self, path: str, mtime: float, size: int, chunk_count: int) -> None:
        """Store file metadata for incremental reindexing."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO file_meta (path, mtime, size, chunk_count) VALUES (?, ?, ?, ?)",
            (path, mtime, size, chunk_count),
        )
        conn.commit()

    def get_all_file_meta(self) -> dict[str, tuple[float, int, int]]:
        """Get all stored file metadata. Returns {path: (mtime, size, chunk_count)}."""
        conn = self._get_conn()
        rows = conn.execute("SELECT path, mtime, size, chunk_count FROM file_meta").fetchall()
        return {path: (mtime, size, chunk_count) for path, mtime, size, chunk_count in rows}

    def delete_file_meta(self, path: str) -> bool:
        """Delete file metadata for a path. Returns True if found."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM file_meta WHERE path = ?", (path,))
        conn.commit()
        return cursor.rowcount > 0

    def clear_file_meta(self) -> None:
        """Delete all file metadata."""
        conn = self._get_conn()
        conn.execute("DELETE FROM file_meta")
        conn.commit()

    def delete_chunks_by_file(self, source_file: str) -> int:
        """Delete all chunks for a specific file. Returns count deleted."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM chunks WHERE source_file = ?", (source_file,))
        conn.commit()
        return cursor.rowcount

    # -- Lifecycle --

    def reset(self) -> None:
        """Drop and recreate all tables."""
        conn = self._get_conn()
        conn.executescript(
            "DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS memories;"
            " DROP TABLE IF EXISTS meta; DROP TABLE IF EXISTS file_meta;"
        )
        conn.commit()
        self._init_db()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- Internal helpers --

    @staticmethod
    def _row_to_chunk(row: tuple) -> Chunk:
        chunk_id, content, source_file, start_line, end_line, chunk_type, emb_blob, created_at = row
        embedding = None
        if emb_blob is not None:
            embedding = np.frombuffer(emb_blob, dtype=np.float32).tolist()
        return Chunk(
            id=chunk_id,
            content=content,
            source_file=source_file,
            start_line=start_line,
            end_line=end_line,
            chunk_type=ChunkType(chunk_type) if chunk_type else ChunkType.WINDOW,
            embedding=embedding,
            created_at=created_at,
        )

    @staticmethod
    def _row_to_memory(row: tuple) -> Memory:
        mem_id, content, memory_type, emb_blob, tags_json, created_at = row
        embedding = None
        if emb_blob is not None:
            embedding = np.frombuffer(emb_blob, dtype=np.float32).tolist()
        tags = json.loads(tags_json) if tags_json else []
        return Memory(
            id=mem_id,
            content=content,
            memory_type=MemoryType(memory_type),
            embedding=embedding,
            tags=tags,
            created_at=created_at,
        )
