"""Tests for memboot.exceptions."""

from __future__ import annotations

import pytest

from memboot.exceptions import (
    ChunkError,
    EmbedError,
    IndexingError,
    IngestError,
    LicenseError,
    MembootError,
    QueryError,
    StoreError,
)


class TestExceptionHierarchy:
    def test_memboot_error_is_exception(self):
        assert issubclass(MembootError, Exception)

    @pytest.mark.parametrize(
        "exc_class",
        [ChunkError, EmbedError, StoreError, IndexingError, IngestError, QueryError, LicenseError],
    )
    def test_subclass_of_memboot_error(self, exc_class):
        assert issubclass(exc_class, MembootError)

    @pytest.mark.parametrize(
        "exc_class",
        [ChunkError, EmbedError, StoreError, IndexingError, IngestError, QueryError, LicenseError],
    )
    def test_message_preserved(self, exc_class):
        msg = "something went wrong"
        exc = exc_class(msg)
        assert str(exc) == msg

    def test_raise_and_catch_as_memboot_error(self):
        with pytest.raises(MembootError):
            raise ChunkError("test")

    def test_raise_and_catch_specific(self):
        with pytest.raises(StoreError):
            raise StoreError("db error")
