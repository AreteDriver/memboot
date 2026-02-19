"""Tests for memboot.context."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from memboot.context import build_context
from memboot.models import ChunkType, SearchResult


def _make_result(
    content: str = "def foo(): pass",
    source: str = "src/foo.py",
    score: float = 0.9,
    chunk_type: ChunkType | None = ChunkType.FUNCTION,
    start_line: int | None = 1,
    end_line: int | None = 5,
) -> SearchResult:
    return SearchResult(
        content=content,
        source=source,
        score=score,
        chunk_type=chunk_type,
        start_line=start_line,
        end_line=end_line,
    )


class TestBuildContext:
    def test_with_code_results(self, tmp_path: Path):
        results = [_make_result()]
        with patch("memboot.context.search", return_value=results):
            ctx = build_context("test", tmp_path)
            assert "src/foo.py:1-5" in ctx
            assert "function" in ctx
            assert "def foo(): pass" in ctx

    def test_with_memory_results(self, tmp_path: Path):
        results = [
            SearchResult(
                content="Always use fixtures.",
                source="memory:m1",
                score=0.85,
            )
        ]
        with patch("memboot.context.search", return_value=results):
            ctx = build_context("test", tmp_path)
            assert "Memory" in ctx
            assert "Always use fixtures." in ctx

    def test_no_results(self, tmp_path: Path):
        with patch("memboot.context.search", return_value=[]):
            ctx = build_context("test", tmp_path)
            assert "No relevant context found" in ctx

    def test_token_budget(self, tmp_path: Path):
        # Create results that exceed the budget
        results = [
            _make_result(content="x" * 500, source=f"f{i}.py", score=0.9 - i * 0.1)
            for i in range(20)
        ]
        with patch("memboot.context.search", return_value=results):
            ctx = build_context("test", tmp_path, max_tokens=200)
            # Should not include all 20 results
            assert ctx.count("```") < 40  # Less than 20 code blocks

    def test_line_attribution(self, tmp_path: Path):
        results = [_make_result(start_line=10, end_line=20, source="main.py")]
        with patch("memboot.context.search", return_value=results):
            ctx = build_context("test", tmp_path)
            assert "main.py:10-20" in ctx

    def test_source_without_lines(self, tmp_path: Path):
        results = [_make_result(start_line=None, end_line=None, source="data.txt")]
        with patch("memboot.context.search", return_value=results):
            ctx = build_context("test", tmp_path)
            assert "data.txt" in ctx

    def test_score_included(self, tmp_path: Path):
        results = [_make_result(score=0.876)]
        with patch("memboot.context.search", return_value=results):
            ctx = build_context("test", tmp_path)
            assert "0.876" in ctx

    def test_results_count_in_header(self, tmp_path: Path):
        results = [_make_result(source=f"f{i}.py") for i in range(3)]
        with patch("memboot.context.search", return_value=results):
            ctx = build_context("test", tmp_path)
            assert "3 results" in ctx
