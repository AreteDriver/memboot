"""MCP stdio server for memboot (Pro feature)."""

from __future__ import annotations

from pathlib import Path

from memboot.context import build_context
from memboot.exceptions import MembootError
from memboot.memory import remember as remember_fn
from memboot.models import MemoryType
from memboot.query import search


def create_mcp_server(project_path: Path):
    """Create an MCP server with memboot tools."""
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as exc:
        raise MembootError(
            "MCP server requires the mcp SDK. Install with: pip install memboot[mcp]"
        ) from exc

    server = Server("memboot")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="query_memory",
                description="Search project memory for relevant code and notes",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="remember",
                description="Store a decision, note, or observation in project memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to remember"},
                        "memory_type": {
                            "type": "string",
                            "enum": ["decision", "note", "observation", "pattern"],
                            "default": "note",
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="get_context",
                description="Get formatted context block for a query",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Context query"},
                        "max_tokens": {"type": "integer", "default": 4000},
                    },
                    "required": ["query"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "query_memory":
            results = search(
                arguments["query"],
                project_path,
                top_k=arguments.get("top_k", 5),
            )
            formatted = "\n\n".join(
                f"**{r.source}** (score: {r.score:.3f})\n{r.content}" for r in results
            )
            return [TextContent(type="text", text=formatted or "No results found.")]

        elif name == "remember":
            mem_type = MemoryType(arguments.get("memory_type", "note"))
            memory = remember_fn(
                content=arguments["content"],
                memory_type=mem_type,
                project_path=project_path,
                tags=arguments.get("tags"),
            )
            return [
                TextContent(
                    type="text",
                    text=f"Remembered: {memory.content[:100]}... (id: {memory.id})",
                )
            ]

        elif name == "get_context":
            ctx = build_context(
                arguments["query"],
                project_path,
                max_tokens=arguments.get("max_tokens", 4000),
            )
            return [TextContent(type="text", text=ctx)]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def run_server(project_path: Path) -> None:
    """Run the MCP stdio server."""
    try:
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise MembootError(
            "MCP server requires the mcp SDK. Install with: pip install memboot[mcp]"
        ) from exc

    server = create_mcp_server(project_path)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
