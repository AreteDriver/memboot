# memboot

[![CI](https://github.com/AreteDriver/memboot/actions/workflows/ci.yml/badge.svg)](https://github.com/AreteDriver/memboot/actions/workflows/ci.yml)
[![CodeQL](https://github.com/AreteDriver/memboot/actions/workflows/codeql.yml/badge.svg)](https://github.com/AreteDriver/memboot/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Zero-infrastructure persistent memory for any LLM.**

Index your codebase, store decisions, and search everything with vector similarity — all local, all SQLite, no API keys needed.

## Features

- **Smart chunking** — AST-aware Python extraction, Markdown heading splits, YAML/JSON key-level, sliding window fallback
- **Local embeddings** — Built-in TF-IDF (zero deps), optional sentence-transformers for semantic search
- **Episodic memory** — Store decisions, patterns, observations alongside your code index
- **Context builder** — Token-budgeted markdown blocks ready for LLM prompts
- **MCP server** — Expose memory as tools for Claude Code, Cursor, and other MCP clients (Pro)
- **File ingestion** — Ingest external files, PDFs, and web pages into project memory

## Install

```bash
pip install memboot
```

Optional extras:

```bash
pip install memboot[embed]  # sentence-transformers for semantic embeddings
pip install memboot[mcp]    # MCP server support
pip install memboot[pdf]    # PDF ingestion
pip install memboot[watch]  # File watching for auto-reindex
pip install memboot[web]    # Web page ingestion
```

## Quick Start

```bash
# Index a project
memboot init /path/to/your/project

# Search for relevant code and memories
memboot query "authentication flow" --project /path/to/your/project

# Store a decision
memboot remember "Use JWT for API auth, sessions for web" --type decision --project /path/to/your/project

# Get formatted context for an LLM prompt
memboot context "database schema" --project /path/to/your/project --max-tokens 4000

# Check license status
memboot status
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `memboot init` | Scan, chunk, embed, and index a project |
| `memboot query` | Search project memory by similarity |
| `memboot remember` | Store an episodic memory (decision, note, observation, pattern) |
| `memboot context` | Export a formatted context block with token budget |
| `memboot status` | Show license tier and available features |
| `memboot reset` | Clear all indexed data and memories |
| `memboot ingest` | Add external files, PDFs, or URLs to memory |
| `memboot watch` | Watch project and auto-reindex on changes |
| `memboot serve` | Start MCP stdio server (Pro) |

## How It Works

```
Project Files ──→ Chunker ──→ Embedder ──→ SQLite Store
                   (AST)      (TF-IDF)     (~/.memboot/)
                                                │
Query Text ─────→ Embedder ──→ Cosine Sim ──→ Results
                                                │
Memories ───────→ Embedder ──→ Store ───────→ Searchable
```

1. **Index** — Recursively discover files, chunk by language (Python AST, Markdown headers, etc.), embed with TF-IDF, store in SQLite
2. **Query** — Embed your query, compute cosine similarity against all chunks and memories, return top-K
3. **Remember** — Store episodic memories (decisions, patterns, observations) with embeddings for later retrieval
4. **Context** — Build token-budgeted markdown blocks with source attribution for LLM consumption

Each project gets its own SQLite database at `~/.memboot/{hash}.db`. No servers, no API keys, no network calls.

## Architecture

```
src/memboot/
├── models.py        # Pydantic v2 data models
├── store.py         # SQLite WAL backend (numpy BLOB serialization)
├── chunker.py       # Language-aware chunking (Python/MD/YAML/JSON/window)
├── embedder.py      # TF-IDF (built-in) + sentence-transformers (optional)
├── indexer.py       # Discovery → chunk → embed → store pipeline
├── query.py         # Cosine similarity search
├── memory.py        # Episodic memory CRUD
├── context.py       # Token-budgeted context builder
├── licensing.py     # Free/Pro tier management
├── cli.py           # 8 Typer CLI commands
├── mcp_server.py    # MCP stdio server (3 tools)
└── ingest/          # External file/PDF/web ingestion
```

## License

MIT
