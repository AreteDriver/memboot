"""Exception hierarchy for memboot."""

from __future__ import annotations


class MembootError(Exception):
    """Base exception for all memboot errors."""


class ChunkError(MembootError):
    """Error during chunking."""


class EmbedError(MembootError):
    """Error during embedding."""


class StoreError(MembootError):
    """Error in the storage layer."""


class IndexingError(MembootError):
    """Error during project indexing."""


class IngestError(MembootError):
    """Error during file ingestion."""


class QueryError(MembootError):
    """Error during similarity search."""


class LicenseError(MembootError):
    """Error related to licensing."""
