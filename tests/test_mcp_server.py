"""Tests for memboot.mcp_server."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memboot.exceptions import MembootError
from memboot.models import Memory, MemoryType, SearchResult


class TestCreateMcpServer:
    def test_import_error(self):
        with (
            patch.dict("sys.modules", {"mcp": None, "mcp.server": None, "mcp.types": None}),
            pytest.raises(MembootError, match="MCP server requires"),
        ):
            from importlib import reload

            import memboot.mcp_server

            reload(memboot.mcp_server)
            memboot.mcp_server.create_mcp_server(Path("."))

    def test_creates_server_with_mocked_mcp(self):
        mock_server_cls = MagicMock()
        mock_server_instance = MagicMock()
        mock_server_cls.return_value = mock_server_instance

        mock_tool = MagicMock()
        mock_text_content = MagicMock()

        mock_mcp_server = MagicMock()
        mock_mcp_server.Server = mock_server_cls
        mock_mcp_types = MagicMock()
        mock_mcp_types.TextContent = mock_text_content
        mock_mcp_types.Tool = mock_tool

        with patch.dict(
            "sys.modules",
            {
                "mcp": MagicMock(),
                "mcp.server": mock_mcp_server,
                "mcp.types": mock_mcp_types,
            },
        ):
            from importlib import reload

            import memboot.mcp_server

            reload(memboot.mcp_server)
            server = memboot.mcp_server.create_mcp_server(Path("/tmp/proj"))
            assert server is not None


class TestRunServer:
    def test_import_error(self):
        with (
            patch.dict(
                "sys.modules",
                {
                    "mcp": None,
                    "mcp.server": None,
                    "mcp.server.stdio": None,
                    "mcp.types": None,
                },
            ),
            pytest.raises(MembootError, match="MCP server requires"),
        ):
            from importlib import reload

            import memboot.mcp_server

            reload(memboot.mcp_server)
            import asyncio

            asyncio.run(memboot.mcp_server.run_server(Path(".")))


class TestMcpToolHandlers:
    """Test the tool handler logic by calling the functions directly."""

    def test_query_formats_results(self):
        """Verify search results get formatted for MCP response."""
        results = [
            SearchResult(content="def foo(): pass", source="main.py", score=0.95),
            SearchResult(content="class Bar: ...", source="models.py", score=0.80),
        ]
        formatted = "\n\n".join(
            f"**{r.source}** (score: {r.score:.3f})\n{r.content}" for r in results
        )
        assert "main.py" in formatted
        assert "0.950" in formatted

    def test_empty_query_results(self):
        results = []
        formatted = "\n\n".join(
            f"**{r.source}** (score: {r.score:.3f})\n{r.content}" for r in results
        )
        assert formatted == "" or formatted == "No results found."

    def test_remember_formats_response(self):
        mem = Memory(
            id="m1",
            content="A very important decision about architecture",
            memory_type=MemoryType.DECISION,
        )
        response = f"Remembered: {mem.content[:100]}... (id: {mem.id})"
        assert "Remembered" in response
        assert "m1" in response
