# memboot

Zero-infrastructure persistent memory for any LLM. SQLite + TF-IDF vector search with no external services.

## Quick Reference

- **Version**: 0.1.0
- **Python**: >=3.11
- **Package layout**: `src/memboot/` (setuptools, `src` layout)
- **Tests**: `tests/` (pytest, 273 tests, 94% coverage, fail_under=90)
- **License**: MIT

## Build & Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# CLI
memboot --version
memboot init <project-path>
memboot query "search terms" --project <path>
memboot remember "important fact" --type note --project <path>
memboot context "query" --project <path>
memboot status
memboot reset --project <path> --yes
memboot ingest <file-or-url> --project <path>
memboot serve  # Pro only — MCP server
```

## Testing

```bash
pytest tests/ -v                    # Full suite with coverage
pytest tests/test_store.py -v       # Single module
```

Coverage gate: 90% (enforced in pyproject.toml via `--cov-fail-under=90`).

## Linting

```bash
ruff check src/ tests/              # Lint
ruff format src/ tests/             # Format (auto-fix)
ruff check src/ tests/ && ruff format --check src/ tests/  # CI check
```

Rules: E, F, W, I, N, UP, B, A, SIM. B008 ignored for `cli.py` (typer defaults).

## Architecture

```
src/memboot/
├── __init__.py          # Version
├── __main__.py          # Entry point
├── models.py            # Pydantic models (Chunk, Memory, SearchResult, ProjectInfo, MembootConfig)
├── exceptions.py        # MembootError hierarchy (7 subclasses)
├── store.py             # SQLite WAL store (chunks/memories/meta, numpy BLOB serialization)
├── embedder.py          # TfidfEmbedder (numpy-only), SentenceTransformerEmbedder (optional)
├── chunker.py           # AST-based Python chunking + markdown/yaml/json/window strategies
├── indexer.py           # Full pipeline: discover → chunk → embed → store
├── query.py             # Cosine similarity search over chunks + memories
├── memory.py            # CRUD for persistent memories
├── context.py           # Formatted markdown context builder with token budget
├── licensing.py         # MMBT license keys, HMAC checksum, Free/Pro tiers
├── gates.py             # @require_pro decorator
├── mcp_server.py        # MCP server (3 tools: query_memory, remember, get_context)
└── ingest/
    ├── files.py         # File ingestion (chunk + embed + store)
    ├── pdf.py           # PDF ingestion (pdfplumber, optional)
    └── web.py           # Web ingestion (trafilatura, optional)
```

## Key Patterns

- **Lazy imports**: CLI commands import at function level. Optional deps (mcp, pdfplumber, trafilatura, sentence-transformers) use try/except ImportError.
- **Patch targets**: When testing CLI, patch at source modules (`memboot.indexer.index_project`), not `memboot.cli.xxx`.
- **Store**: SQLite WAL mode, numpy float32 BLOB serialization, INSERT OR REPLACE for upserts.
- **Embeddings**: TF-IDF state saved to store meta as JSON, restored on query/ingest.
- **Licensing**: MMBT prefix, `memboot-v1` salt, SHA256 checksum. Key via `MEMBOOT_LICENSE` env var or `~/.memboot-license` file.
- **Per-project DB**: `~/.memboot/{sha256(project_path)[:12]}.db` — one DB per project, auto-created on `init`.

## Optional Dependencies

```bash
pip install memboot[mcp]    # MCP server support
pip install memboot[pdf]    # PDF ingestion (pdfplumber)
pip install memboot[web]    # Web ingestion (trafilatura)
pip install memboot[embed]  # Sentence-transformer embeddings
```

## Conventions

- Type hints throughout (mypy strict)
- `from __future__ import annotations` in test files
- Pydantic v2 models with StrEnum
- Ruff for all linting/formatting
- Conventional commits
