"""Export formatted context blocks."""

from __future__ import annotations

from pathlib import Path

from memboot.query import search


def build_context(
    query_text: str,
    project_path: Path,
    max_tokens: int = 4000,
    top_k: int = 10,
) -> str:
    """Build a formatted markdown context block with source attribution."""
    results = search(query_text, project_path, top_k=top_k)

    if not results:
        return "## No relevant context found.\n"

    sections: list[str] = []
    total_tokens = 0

    for r in results:
        entry_tokens = len(r.content) // 4 + 20
        if total_tokens + entry_tokens > max_tokens:
            break

        if r.source.startswith("memory:"):
            sections.append(f"### Memory\n{r.content}\n*Score: {r.score:.3f}*")
        else:
            loc = r.source
            if r.start_line is not None:
                loc += f":{r.start_line}"
                if r.end_line is not None:
                    loc += f"-{r.end_line}"
            chunk_label = r.chunk_type.value if r.chunk_type else "text"
            sections.append(
                f"### {loc} ({chunk_label})\n```\n{r.content}\n```\n*Score: {r.score:.3f}*"
            )
        total_tokens += entry_tokens

    header = f"## Relevant Context ({len(sections)} results)\n\n"
    return header + "\n\n---\n\n".join(sections) + "\n"
