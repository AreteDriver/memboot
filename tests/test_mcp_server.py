"""Tests for memboot.mcp_server."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memboot.exceptions import MembootError
from memboot.models import Memory, MemoryType, SearchResult


def _make_mock_mcp():
    """Create mock MCP modules that capture registered handlers."""
    captured = {}

    mock_server_instance = MagicMock()

    def mock_list_tools():
        def decorator(fn):
            captured["list_tools"] = fn
            return fn

        return decorator

    def mock_call_tool():
        def decorator(fn):
            captured["call_tool"] = fn
            return fn

        return decorator

    mock_server_instance.list_tools = mock_list_tools
    mock_server_instance.call_tool = mock_call_tool

    mock_server_cls = MagicMock(return_value=mock_server_instance)
    mock_tool = MagicMock(side_effect=lambda **kwargs: kwargs)
    mock_text_content = MagicMock(side_effect=lambda **kwargs: kwargs)

    mock_mcp_server = MagicMock()
    mock_mcp_server.Server = mock_server_cls
    mock_mcp_types = MagicMock()
    mock_mcp_types.TextContent = mock_text_content
    mock_mcp_types.Tool = mock_tool

    modules = {
        "mcp": MagicMock(),
        "mcp.server": mock_mcp_server,
        "mcp.types": mock_mcp_types,
    }
    return modules, captured, mock_server_instance


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
        modules, captured, _ = _make_mock_mcp()
        with patch.dict("sys.modules", modules):
            from importlib import reload

            import memboot.mcp_server

            reload(memboot.mcp_server)
            server = memboot.mcp_server.create_mcp_server(Path("/tmp/proj"))
            assert server is not None
            assert "list_tools" in captured
            assert "call_tool" in captured


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
            asyncio.run(memboot.mcp_server.run_server(Path(".")))

    def test_run_server_calls_stdio(self):
        modules, _, mock_server = _make_mock_mcp()
        mock_stdio = MagicMock()
        mock_read = MagicMock()
        mock_write = MagicMock()

        mock_stdio_ctx = MagicMock()

        async def aenter(*_):
            return mock_read, mock_write

        async def aexit(*_):
            return False

        mock_stdio_ctx.__aenter__ = aenter
        mock_stdio_ctx.__aexit__ = aexit
        mock_stdio.stdio_server = MagicMock(return_value=mock_stdio_ctx)

        async def fake_run(*args):
            pass

        mock_server.run = MagicMock(side_effect=fake_run)
        mock_server.create_initialization_options = MagicMock(return_value={})

        modules["mcp.server.stdio"] = mock_stdio
        with patch.dict("sys.modules", modules):
            from importlib import reload

            import memboot.mcp_server

            reload(memboot.mcp_server)
            asyncio.run(memboot.mcp_server.run_server(Path("/tmp")))
            mock_server.run.assert_called_once()


class TestMcpToolHandlers:
    """Test the registered MCP tool handlers end-to-end."""

    @pytest.fixture
    def _handlers(self):
        modules, captured, _ = _make_mock_mcp()
        with patch.dict("sys.modules", modules):
            from importlib import reload

            import memboot.mcp_server

            reload(memboot.mcp_server)
            memboot.mcp_server.create_mcp_server(Path("/tmp/proj"))
        # Return both captured handlers and the reloaded module for patching
        return captured, memboot.mcp_server

    def test_list_tools_returns_three(self, _handlers):
        captured, _ = _handlers
        tools = asyncio.run(captured["list_tools"]())
        assert len(tools) == 3
        names = {t["name"] for t in tools}
        assert names == {"query_memory", "remember", "get_context"}

    def test_call_query_memory(self, _handlers):
        captured, mod = _handlers
        mock_results = [
            SearchResult(content="def foo(): pass", source="main.py", score=0.95),
        ]
        with patch.object(mod, "search", return_value=mock_results):
            result = asyncio.run(
                captured["call_tool"]("query_memory", {"query": "foo", "top_k": 3})
            )
        assert len(result) == 1
        assert "main.py" in result[0]["text"]

    def test_call_query_memory_empty(self, _handlers):
        captured, mod = _handlers
        with patch.object(mod, "search", return_value=[]):
            result = asyncio.run(captured["call_tool"]("query_memory", {"query": "nothing"}))
        assert "No results found" in result[0]["text"]

    def test_call_remember(self, _handlers):
        captured, mod = _handlers
        mock_mem = Memory(id="m1", content="test note", memory_type=MemoryType.NOTE)
        with patch.object(mod, "remember_fn", return_value=mock_mem):
            result = asyncio.run(captured["call_tool"]("remember", {"content": "test note"}))
        assert "Remembered" in result[0]["text"]
        assert "m1" in result[0]["text"]

    def test_call_remember_with_type_and_tags(self, _handlers):
        captured, mod = _handlers
        mock_mem = Memory(id="m2", content="architecture decision", memory_type=MemoryType.DECISION)
        with patch.object(mod, "remember_fn", return_value=mock_mem) as mock_fn:
            asyncio.run(
                captured["call_tool"](
                    "remember",
                    {
                        "content": "architecture decision",
                        "memory_type": "decision",
                        "tags": ["arch"],
                    },
                )
            )
        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args
        assert call_kwargs.kwargs.get("tags") == ["arch"]

    def test_call_get_context(self, _handlers):
        captured, mod = _handlers
        with patch.object(mod, "build_context", return_value="## Context\nsome content"):
            result = asyncio.run(
                captured["call_tool"]("get_context", {"query": "architecture", "max_tokens": 2000})
            )
        assert "Context" in result[0]["text"]

    def test_call_unknown_tool(self, _handlers):
        captured, _ = _handlers
        result = asyncio.run(captured["call_tool"]("nonexistent", {}))
        assert "Unknown tool" in result[0]["text"]
