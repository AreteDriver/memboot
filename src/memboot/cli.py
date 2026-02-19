"""CLI entry point for memboot."""

from __future__ import annotations

import json as json_mod
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from memboot import __version__
from memboot.exceptions import MembootError
from memboot.licensing import (
    TIER_DEFINITIONS,
    get_license_info,
    get_upgrade_message,
    has_feature,
)

app = typer.Typer(
    name="memboot",
    help="Zero-infrastructure persistent memory for any LLM.",
)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Zero-infrastructure persistent memory for any LLM."""
    if version:
        console.print(f"memboot {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def status() -> None:
    """Show license status and available features."""
    info = get_license_info()
    tier_config = TIER_DEFINITIONS[info.tier]

    console.print(f"\n[bold]memboot {__version__}[/bold]")
    console.print(f"[bold]Tier:[/bold] {tier_config.name} ({tier_config.price_label})")

    if info.license_key:
        masked = info.license_key[:9] + "****-****"
        console.print(f"[bold]Key:[/bold] {masked}")
        valid_str = "[green]valid[/green]" if info.valid else "[red]invalid[/red]"
        console.print(f"[bold]Valid:[/bold] {valid_str}")

    console.print(f"\n[bold]Features:[/bold] {', '.join(tier_config.features)}")
    console.print()


@app.command(name="init")
def init_cmd(
    project_dir: Path = typer.Argument(".", help="Project directory to index."),
    force: bool = typer.Option(False, "--force", "-f", help="Force full reindex."),
    backend: str = typer.Option("tfidf", "--backend", "-b", help="Embedding backend."),
) -> None:
    """Scan, chunk, embed, and index a project."""
    from memboot.indexer import index_project
    from memboot.models import MembootConfig

    try:
        config = MembootConfig(embedding_backend=backend)
        info = index_project(project_dir.resolve(), config=config, force=force)
        console.print(f"\n[green]Indexed {info.chunk_count} chunks[/green]")
        console.print(f"[dim]DB: {info.db_path}[/dim]")
        console.print(f"[dim]Backend: {info.embedding_backend}, dim: {info.embedding_dim}[/dim]")
        console.print()
    except MembootError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def query(
    text: str = typer.Argument(..., help="Search query."),
    project_dir: Path = typer.Option(".", "--project", "-p", help="Project directory."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results."),
    json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Search project memory."""
    from memboot.query import search

    try:
        results = search(text, project_dir.resolve(), top_k=top_k)
    except MembootError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if json:
        data = [r.model_dump(exclude_none=True) for r in results]
        console.print_json(json_mod.dumps(data, indent=2))
        return

    if not results:
        console.print("[dim]No results found.[/dim]")
        return

    table = Table(title=f"Results for: {text}")
    table.add_column("Source", style="cyan")
    table.add_column("Score", style="green", justify="right")
    table.add_column("Content", max_width=60)

    for r in results:
        source = r.source
        if r.start_line is not None:
            source += f":{r.start_line}"
        content_preview = r.content[:100].replace("\n", " ")
        if len(r.content) > 100:
            content_preview += "..."
        table.add_row(source, f"{r.score:.3f}", content_preview)

    console.print(table)


@app.command()
def remember(
    content: str = typer.Argument(..., help="Content to remember."),
    memory_type: str = typer.Option("note", "--type", "-t", help="Memory type."),
    project_dir: Path = typer.Option(".", "--project", "-p", help="Project directory."),
    tags: list[str] | None = typer.Option(None, "--tag", help="Tags (repeatable)."),
) -> None:
    """Store an episodic memory."""
    from memboot.memory import remember as remember_fn
    from memboot.models import MemoryType

    try:
        mem_type = MemoryType(memory_type)
    except ValueError:
        console.print(f"[red]Unknown memory type:[/red] {memory_type}")
        console.print(f"[dim]Valid types: {', '.join(t.value for t in MemoryType)}[/dim]")
        raise typer.Exit(1) from None

    try:
        memory = remember_fn(content, mem_type, project_dir.resolve(), tags=tags)
        console.print(f"[green]Remembered:[/green] {memory.content[:80]}")
        console.print(f"[dim]ID: {memory.id}[/dim]")
    except MembootError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def context(
    query_text: str = typer.Argument(..., help="Query for context."),
    project_dir: Path = typer.Option(".", "--project", "-p", help="Project directory."),
    max_tokens: int = typer.Option(4000, "--max-tokens", help="Token budget."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Max results to consider."),
) -> None:
    """Export formatted context block."""
    from memboot.context import build_context

    try:
        ctx = build_context(query_text, project_dir.resolve(), max_tokens=max_tokens, top_k=top_k)
        console.print(ctx)
    except MembootError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def reset(
    project_dir: Path = typer.Option(".", "--project", "-p", help="Project directory."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Clear all indexed data and memories."""
    from memboot.indexer import get_db_path

    db_path = get_db_path(project_dir.resolve())
    if not db_path.exists():
        console.print("[dim]No index found. Nothing to reset.[/dim]")
        return

    if not yes:
        confirm = typer.confirm("This will delete all indexed data and memories. Continue?")
        if not confirm:
            raise typer.Abort()

    from memboot.store import MembootStore

    store = MembootStore(db_path)
    store.reset()
    store.close()
    console.print("[green]Memory reset.[/green]")


@app.command()
def ingest(
    source: str = typer.Argument(..., help="File path or URL to ingest."),
    project_dir: Path = typer.Option(".", "--project", "-p", help="Project directory."),
) -> None:
    """Ingest an external file into project memory."""
    if source.startswith(("http://", "https://")):
        if not has_feature("ingest_web"):
            console.print(f"[yellow]{get_upgrade_message('ingest_web')}[/yellow]")
            raise typer.Exit(1)
        from memboot.ingest.web import ingest_url

        try:
            chunks = ingest_url(source, project_dir.resolve())
            console.print(f"[green]Ingested {len(chunks)} chunks from URL.[/green]")
        except MembootError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
    elif source.lower().endswith(".pdf"):
        if not has_feature("ingest_pdf"):
            console.print(f"[yellow]{get_upgrade_message('ingest_pdf')}[/yellow]")
            raise typer.Exit(1)
        from memboot.ingest.pdf import ingest_pdf

        try:
            chunks = ingest_pdf(Path(source), project_dir.resolve())
            console.print(f"[green]Ingested {len(chunks)} chunks from PDF.[/green]")
        except MembootError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
    else:
        from memboot.ingest.files import ingest_file

        try:
            chunks = ingest_file(Path(source), project_dir.resolve())
            console.print(f"[green]Ingested {len(chunks)} chunks from {source}.[/green]")
        except MembootError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc


@app.command()
def serve(
    project_dir: Path = typer.Option(".", "--project", "-p", help="Project directory."),
) -> None:
    """Start MCP stdio server (Pro feature)."""
    if not has_feature("serve"):
        console.print(f"[yellow]{get_upgrade_message('serve')}[/yellow]")
        raise typer.Exit(1)

    import asyncio

    from memboot.mcp_server import run_server

    try:
        asyncio.run(run_server(project_dir.resolve()))
    except MembootError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
