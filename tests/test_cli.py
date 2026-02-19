"""Tests for memboot.cli."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from memboot import __version__
from memboot.cli import app
from memboot.exceptions import IndexingError, MembootError, QueryError
from memboot.models import Memory, MemoryType, ProjectInfo, SearchResult

runner = CliRunner()


class TestMain:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_no_command_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "memboot" in result.output.lower()


class TestStatus:
    def test_shows_tier(self, monkeypatch):
        monkeypatch.delenv("MEMBOOT_LICENSE", raising=False)
        monkeypatch.setattr("memboot.licensing._LICENSE_LOCATIONS", [])
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Free" in result.output

    def test_shows_features(self, monkeypatch):
        monkeypatch.delenv("MEMBOOT_LICENSE", raising=False)
        monkeypatch.setattr("memboot.licensing._LICENSE_LOCATIONS", [])
        result = runner.invoke(app, ["status"])
        assert "init" in result.output

    def test_pro_with_key(self, monkeypatch):
        from memboot.licensing import _compute_check_segment

        body = "TEST-ABCD"
        check = _compute_check_segment(body)
        key = f"MMBT-{body}-{check}"
        monkeypatch.setenv("MEMBOOT_LICENSE", key)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Pro" in result.output
        assert "valid" in result.output.lower()

    def test_invalid_key(self, monkeypatch):
        monkeypatch.setenv("MEMBOOT_LICENSE", "MMBT-ABCD-EFGH-ZZZZ")
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "invalid" in result.output.lower()


class TestInit:
    def test_success(self, tmp_path: Path):
        project = tmp_path / "proj"
        project.mkdir()
        (project / "test.py").write_text("x = 1\n")

        info = ProjectInfo(
            project_path=str(project),
            project_hash="abc",
            db_path=str(tmp_path / "test.db"),
            chunk_count=5,
            embedding_dim=512,
            embedding_backend="tfidf",
        )
        with patch("memboot.indexer.index_project", return_value=info):
            result = runner.invoke(app, ["init", str(project)])
            assert result.exit_code == 0
            assert "5 chunks" in result.output

    def test_error(self, tmp_path: Path):
        with patch("memboot.indexer.index_project", side_effect=IndexingError("bad dir")):
            result = runner.invoke(app, ["init", str(tmp_path)])
            assert result.exit_code == 1
            assert "bad dir" in result.output


class TestQuery:
    def test_table_output(self, tmp_path: Path):
        results = [
            SearchResult(
                content="def hello(): pass",
                source="main.py",
                score=0.95,
            )
        ]
        with patch("memboot.query.search", return_value=results):
            result = runner.invoke(app, ["query", "hello", "--project", str(tmp_path)])
            assert result.exit_code == 0
            assert "main.py" in result.output

    def test_json_output(self, tmp_path: Path):
        results = [SearchResult(content="test", source="f.py", score=0.8)]
        with patch("memboot.query.search", return_value=results):
            result = runner.invoke(app, ["query", "test", "--project", str(tmp_path), "--json"])
            assert result.exit_code == 0
            assert "f.py" in result.output

    def test_no_results(self, tmp_path: Path):
        with patch("memboot.query.search", return_value=[]):
            result = runner.invoke(app, ["query", "nothing", "--project", str(tmp_path)])
            assert result.exit_code == 0
            assert "No results" in result.output

    def test_error(self, tmp_path: Path):
        with patch("memboot.query.search", side_effect=QueryError("no index")):
            result = runner.invoke(app, ["query", "test", "--project", str(tmp_path)])
            assert result.exit_code == 1
            assert "no index" in result.output

    def test_with_start_line(self, tmp_path: Path):
        results = [
            SearchResult(
                content="def hello(): pass",
                source="main.py",
                score=0.95,
                start_line=10,
            )
        ]
        with patch("memboot.query.search", return_value=results):
            result = runner.invoke(app, ["query", "hello", "--project", str(tmp_path)])
            assert result.exit_code == 0
            assert "main.py:10" in result.output

    def test_long_content_truncated(self, tmp_path: Path):
        results = [
            SearchResult(
                content="x" * 200,
                source="main.py",
                score=0.95,
            )
        ]
        with patch("memboot.query.search", return_value=results):
            result = runner.invoke(app, ["query", "x", "--project", str(tmp_path)])
            assert result.exit_code == 0
            # Rich table may use unicode ellipsis (…) or ASCII (...)
            assert "..." in result.output or "…" in result.output


class TestRemember:
    def test_success(self, tmp_path: Path):
        mem = Memory(
            id="m1",
            content="Test note content",
            memory_type=MemoryType.NOTE,
        )
        with patch("memboot.memory.remember", return_value=mem):
            result = runner.invoke(
                app, ["remember", "Test note content", "--project", str(tmp_path)]
            )
            assert result.exit_code == 0
            assert "Remembered" in result.output

    def test_invalid_type(self, tmp_path: Path):
        result = runner.invoke(
            app, ["remember", "note", "--type", "invalid_type", "--project", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "Unknown memory type" in result.output

    def test_error(self, tmp_path: Path):
        with patch("memboot.memory.remember", side_effect=MembootError("fail")):
            result = runner.invoke(app, ["remember", "note", "--project", str(tmp_path)])
            assert result.exit_code == 1


class TestContext:
    def test_success(self, tmp_path: Path):
        with patch("memboot.context.build_context", return_value="## Context\nSome context here."):
            result = runner.invoke(app, ["context", "test query", "--project", str(tmp_path)])
            assert result.exit_code == 0
            assert "Context" in result.output

    def test_error(self, tmp_path: Path):
        with patch("memboot.context.build_context", side_effect=MembootError("fail")):
            result = runner.invoke(app, ["context", "test", "--project", str(tmp_path)])
            assert result.exit_code == 1


class TestReset:
    def test_with_yes_flag(self, tmp_path: Path):
        mock_store = MagicMock()
        db_path = tmp_path / "test.db"
        db_path.touch()
        with (
            patch("memboot.indexer.get_db_path", return_value=db_path),
            patch("memboot.store.MembootStore", return_value=mock_store),
        ):
            result = runner.invoke(app, ["reset", "--project", str(tmp_path), "--yes"])
            assert result.exit_code == 0
            assert "reset" in result.output.lower()
            mock_store.reset.assert_called_once()

    def test_no_index(self, tmp_path: Path):
        db_path = tmp_path / "nonexistent.db"
        with patch("memboot.indexer.get_db_path", return_value=db_path):
            result = runner.invoke(app, ["reset", "--project", str(tmp_path), "--yes"])
            assert result.exit_code == 0
            assert "No index" in result.output

    def test_abort(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        db_path.touch()
        with patch("memboot.indexer.get_db_path", return_value=db_path):
            result = runner.invoke(app, ["reset", "--project", str(tmp_path)], input="n\n")
            assert result.exit_code != 0


class TestIngest:
    def test_file_success(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("some data\n")
        with patch("memboot.ingest.files.ingest_file", return_value=[MagicMock()] * 3):
            result = runner.invoke(app, ["ingest", str(f), "--project", str(tmp_path)])
            assert result.exit_code == 0
            assert "3 chunks" in result.output

    def test_file_error(self, tmp_path: Path):
        from memboot.exceptions import IngestError

        with patch("memboot.ingest.files.ingest_file", side_effect=IngestError("bad")):
            result = runner.invoke(
                app, ["ingest", str(tmp_path / "f.py"), "--project", str(tmp_path)]
            )
            assert result.exit_code == 1

    def test_url_gated(self, monkeypatch):
        monkeypatch.delenv("MEMBOOT_LICENSE", raising=False)
        monkeypatch.setattr("memboot.licensing._LICENSE_LOCATIONS", [])
        result = runner.invoke(app, ["ingest", "https://example.com"])
        assert result.exit_code == 1

    def test_url_success_with_pro(self, tmp_path: Path, monkeypatch):
        from memboot.licensing import _compute_check_segment

        body = "TEST-ABCD"
        check = _compute_check_segment(body)
        monkeypatch.setenv("MEMBOOT_LICENSE", f"MMBT-{body}-{check}")
        with patch("memboot.ingest.web.ingest_url", return_value=[MagicMock()] * 2):
            result = runner.invoke(
                app, ["ingest", "https://example.com", "--project", str(tmp_path)]
            )
            assert result.exit_code == 0
            assert "2 chunks" in result.output

    def test_url_error_with_pro(self, tmp_path: Path, monkeypatch):
        from memboot.exceptions import IngestError
        from memboot.licensing import _compute_check_segment

        body = "TEST-ABCD"
        check = _compute_check_segment(body)
        monkeypatch.setenv("MEMBOOT_LICENSE", f"MMBT-{body}-{check}")
        with patch("memboot.ingest.web.ingest_url", side_effect=IngestError("fail")):
            result = runner.invoke(
                app, ["ingest", "https://example.com", "--project", str(tmp_path)]
            )
            assert result.exit_code == 1

    def test_pdf_gated(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv("MEMBOOT_LICENSE", raising=False)
        monkeypatch.setattr("memboot.licensing._LICENSE_LOCATIONS", [])
        result = runner.invoke(app, ["ingest", "doc.pdf", "--project", str(tmp_path)])
        assert result.exit_code == 1

    def test_pdf_success_with_pro(self, tmp_path: Path, monkeypatch):
        from memboot.licensing import _compute_check_segment

        body = "TEST-ABCD"
        check = _compute_check_segment(body)
        monkeypatch.setenv("MEMBOOT_LICENSE", f"MMBT-{body}-{check}")
        with patch("memboot.ingest.pdf.ingest_pdf", return_value=[MagicMock()] * 4):
            result = runner.invoke(app, ["ingest", "doc.pdf", "--project", str(tmp_path)])
            assert result.exit_code == 0
            assert "4 chunks" in result.output

    def test_pdf_error_with_pro(self, tmp_path: Path, monkeypatch):
        from memboot.exceptions import IngestError
        from memboot.licensing import _compute_check_segment

        body = "TEST-ABCD"
        check = _compute_check_segment(body)
        monkeypatch.setenv("MEMBOOT_LICENSE", f"MMBT-{body}-{check}")
        with patch("memboot.ingest.pdf.ingest_pdf", side_effect=IngestError("fail")):
            result = runner.invoke(app, ["ingest", "doc.pdf", "--project", str(tmp_path)])
            assert result.exit_code == 1


class TestServe:
    def test_gated(self, monkeypatch):
        monkeypatch.delenv("MEMBOOT_LICENSE", raising=False)
        monkeypatch.setattr("memboot.licensing._LICENSE_LOCATIONS", [])
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 1

    def test_serve_error_with_pro(self, monkeypatch):
        from memboot.licensing import _compute_check_segment

        body = "TEST-ABCD"
        check = _compute_check_segment(body)
        monkeypatch.setenv("MEMBOOT_LICENSE", f"MMBT-{body}-{check}")
        with patch("memboot.mcp_server.run_server", side_effect=MembootError("no mcp")):
            result = runner.invoke(app, ["serve"])
            assert result.exit_code == 1
