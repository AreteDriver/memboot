"""Tests for memboot.ingest modules."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memboot.exceptions import IngestError
from memboot.ingest.files import ingest_file
from memboot.store import MembootStore


@pytest.fixture
def indexed_store(tmp_path: Path) -> tuple[Path, Path]:
    """Create a project with a pre-fitted TF-IDF store."""
    from memboot.embedder import TfidfEmbedder

    db_path = tmp_path / "test.db"
    store = MembootStore(db_path)

    emb = TfidfEmbedder(max_features=10)
    emb.fit(["hello world", "foo bar baz"])
    store.set_meta("tfidf_state", json.dumps(emb.save_state()))
    store.set_meta("embedding_backend", "tfidf")
    store.close()

    project = tmp_path / "proj"
    project.mkdir()
    return project, db_path


class TestIngestFile:
    def test_success(self, indexed_store, monkeypatch):
        project, db_path = indexed_store
        monkeypatch.setattr("memboot.ingest.files.get_db_path", lambda p: db_path)

        f = project / "test.py"
        f.write_text("def hello():\n    return 'hello'\n")

        chunks = ingest_file(f, project)
        assert len(chunks) >= 1
        assert all(c.embedding is not None for c in chunks)

    def test_unsupported_extension(self, indexed_store, monkeypatch):
        project, db_path = indexed_store
        monkeypatch.setattr("memboot.ingest.files.get_db_path", lambda p: db_path)

        f = project / "data.xlsx"
        f.write_text("fake xlsx")

        with pytest.raises(IngestError, match="Unsupported file type"):
            ingest_file(f, project)

    def test_file_not_found(self, indexed_store, monkeypatch):
        project, db_path = indexed_store
        monkeypatch.setattr("memboot.ingest.files.get_db_path", lambda p: db_path)

        with pytest.raises(IngestError, match="File not found"):
            ingest_file(project / "nonexistent.py", project)

    def test_empty_file(self, indexed_store, monkeypatch):
        project, db_path = indexed_store
        monkeypatch.setattr("memboot.ingest.files.get_db_path", lambda p: db_path)

        f = project / "empty.py"
        f.write_text("")

        chunks = ingest_file(f, project)
        assert chunks == []

    def test_without_tfidf_state(self, tmp_path: Path, monkeypatch):
        """When no TF-IDF state, should create fresh embedder."""
        db_path = tmp_path / "test.db"
        store = MembootStore(db_path)
        store.close()

        project = tmp_path / "proj"
        project.mkdir()
        monkeypatch.setattr("memboot.ingest.files.get_db_path", lambda p: db_path)

        f = project / "test.py"
        f.write_text("x = 1\ny = 2\n")

        chunks = ingest_file(f, project)
        assert len(chunks) >= 1


class TestIngestPdf:
    def test_import_error(self, tmp_path: Path):
        with (
            patch.dict("sys.modules", {"pdfplumber": None}),
            pytest.raises(IngestError, match="pdfplumber"),
        ):
            from importlib import reload

            import memboot.ingest.pdf

            reload(memboot.ingest.pdf)
            memboot.ingest.pdf.ingest_pdf(tmp_path / "doc.pdf", tmp_path)

    def test_file_not_found(self, indexed_store, monkeypatch):
        project, db_path = indexed_store
        monkeypatch.setattr("memboot.ingest.pdf.get_db_path", lambda p: db_path)

        mock_pdfplumber = MagicMock()
        with patch.dict("sys.modules", {"pdfplumber": mock_pdfplumber}):
            from importlib import reload

            import memboot.ingest.pdf

            reload(memboot.ingest.pdf)
            with pytest.raises(IngestError, match="File not found"):
                memboot.ingest.pdf.ingest_pdf(project / "nope.pdf", project)

    def test_success_with_mocked_pdf(self, indexed_store, monkeypatch):
        project, db_path = indexed_store
        monkeypatch.setattr("memboot.ingest.pdf.get_db_path", lambda p: db_path)

        # Create a fake PDF file
        pdf_file = project / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        # Mock pdfplumber
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is page one content with words."

        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]

        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        with patch.dict("sys.modules", {"pdfplumber": mock_pdfplumber}):
            from importlib import reload

            import memboot.ingest.pdf

            reload(memboot.ingest.pdf)
            chunks = memboot.ingest.pdf.ingest_pdf(pdf_file, project)
            assert len(chunks) >= 1
            assert all(c.embedding is not None for c in chunks)

    def test_empty_pdf(self, indexed_store, monkeypatch):
        project, db_path = indexed_store
        monkeypatch.setattr("memboot.ingest.pdf.get_db_path", lambda p: db_path)

        pdf_file = project / "empty.pdf"
        pdf_file.write_bytes(b"fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""

        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]

        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        with patch.dict("sys.modules", {"pdfplumber": mock_pdfplumber}):
            from importlib import reload

            import memboot.ingest.pdf

            reload(memboot.ingest.pdf)
            chunks = memboot.ingest.pdf.ingest_pdf(pdf_file, project)
            assert chunks == []


class TestIngestWeb:
    def test_import_error(self, tmp_path: Path):
        with (
            patch.dict("sys.modules", {"trafilatura": None}),
            pytest.raises(IngestError, match="trafilatura"),
        ):
            from importlib import reload

            import memboot.ingest.web

            reload(memboot.ingest.web)
            memboot.ingest.web.ingest_url("https://example.com", tmp_path)

    def test_success_with_mocked_web(self, indexed_store, monkeypatch):
        project, db_path = indexed_store
        monkeypatch.setattr("memboot.ingest.web.get_db_path", lambda p: db_path)

        mock_trafilatura = MagicMock()
        mock_trafilatura.fetch_url.return_value = "<html><body>Hello world content</body></html>"
        mock_trafilatura.extract.return_value = "Hello world content from the web page."

        with patch.dict("sys.modules", {"trafilatura": mock_trafilatura}):
            from importlib import reload

            import memboot.ingest.web

            reload(memboot.ingest.web)
            chunks = memboot.ingest.web.ingest_url("https://example.com", project)
            assert len(chunks) >= 1
            assert all(c.embedding is not None for c in chunks)

    def test_download_failure(self, indexed_store, monkeypatch):
        project, db_path = indexed_store

        mock_trafilatura = MagicMock()
        mock_trafilatura.fetch_url.return_value = None

        with patch.dict("sys.modules", {"trafilatura": mock_trafilatura}):
            from importlib import reload

            import memboot.ingest.web

            reload(memboot.ingest.web)
            with pytest.raises(IngestError, match="Could not download"):
                memboot.ingest.web.ingest_url("https://example.com", project)

    def test_no_extractable_content(self, indexed_store, monkeypatch):
        project, db_path = indexed_store

        mock_trafilatura = MagicMock()
        mock_trafilatura.fetch_url.return_value = "<html></html>"
        mock_trafilatura.extract.return_value = ""

        with patch.dict("sys.modules", {"trafilatura": mock_trafilatura}):
            from importlib import reload

            import memboot.ingest.web

            reload(memboot.ingest.web)
            with pytest.raises(IngestError, match="No extractable content"):
                memboot.ingest.web.ingest_url("https://example.com", project)
