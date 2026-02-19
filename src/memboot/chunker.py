"""Smart chunking for different file types."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

from memboot.exceptions import ChunkError
from memboot.models import ChunkType, MembootConfig


class ChunkResult:
    """Result from chunking a file."""

    __slots__ = ("content", "chunk_type", "start_line", "end_line", "metadata")

    def __init__(
        self,
        content: str,
        chunk_type: ChunkType,
        start_line: int,
        end_line: int,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.content = content
        self.chunk_type = chunk_type
        self.start_line = start_line
        self.end_line = end_line
        self.metadata = metadata or {}


def chunk_file(file_path: Path, config: MembootConfig | None = None) -> list[ChunkResult]:
    """Dispatch to the right chunker based on file extension."""
    config = config or MembootConfig()
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ChunkError(f"Cannot read {file_path}: {exc}") from exc

    if not content.strip():
        return []

    ext = file_path.suffix.lower()
    if ext == ".py":
        return _chunk_python(content, config)
    elif ext == ".md":
        return _chunk_markdown(content, config)
    elif ext in (".yaml", ".yml"):
        return _chunk_yaml(content, config)
    elif ext == ".json":
        return _chunk_json(content, config)
    else:
        return _chunk_window(content, config)


def _chunk_python(content: str, config: MembootConfig) -> list[ChunkResult]:
    """AST-based chunking for Python files."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _chunk_window(content, config)

    lines = content.splitlines(keepends=True)
    chunks: list[ChunkResult] = []
    covered_lines: set[int] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = node.end_lineno or node.lineno
            chunk_content = "".join(lines[start - 1 : end])
            chunks.append(
                ChunkResult(
                    content=chunk_content.rstrip(),
                    chunk_type=ChunkType.FUNCTION,
                    start_line=start,
                    end_line=end,
                    metadata={"name": node.name},
                )
            )
            covered_lines.update(range(start, end + 1))

        elif isinstance(node, ast.ClassDef):
            start = node.lineno
            end = node.end_lineno or node.lineno
            # Check if class has methods that should be split
            methods = [
                n
                for n in ast.iter_child_nodes(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            est_tokens = (end - start + 1) * 4  # rough estimate
            if methods and est_tokens > config.max_chunk_tokens * 4:
                # Split class into method-level chunks
                for method in methods:
                    m_start = method.lineno
                    m_end = method.end_lineno or method.lineno
                    m_content = "".join(lines[m_start - 1 : m_end])
                    chunks.append(
                        ChunkResult(
                            content=m_content.rstrip(),
                            chunk_type=ChunkType.METHOD,
                            start_line=m_start,
                            end_line=m_end,
                            metadata={"class": node.name, "name": method.name},
                        )
                    )
                    covered_lines.update(range(m_start, m_end + 1))
                # Also capture class header (docstring, class vars)
                first_method_line = min(m.lineno for m in methods)
                if first_method_line > start + 1:
                    header = "".join(lines[start - 1 : first_method_line - 1])
                    if header.strip():
                        chunks.append(
                            ChunkResult(
                                content=header.rstrip(),
                                chunk_type=ChunkType.CLASS,
                                start_line=start,
                                end_line=first_method_line - 1,
                                metadata={"name": node.name},
                            )
                        )
                covered_lines.update(range(start, end + 1))
            else:
                chunk_content = "".join(lines[start - 1 : end])
                chunks.append(
                    ChunkResult(
                        content=chunk_content.rstrip(),
                        chunk_type=ChunkType.CLASS,
                        start_line=start,
                        end_line=end,
                        metadata={"name": node.name},
                    )
                )
                covered_lines.update(range(start, end + 1))

    # Capture module-level code not covered by functions/classes
    module_lines: list[str] = []
    module_start: int | None = None
    for i, line in enumerate(lines, 1):
        if i not in covered_lines and line.strip() and not line.strip().startswith("#"):
            if module_start is None:
                module_start = i
            module_lines.append(line)

    if module_lines and module_start is not None:
        module_content = "".join(module_lines)
        if module_content.strip():
            chunks.append(
                ChunkResult(
                    content=module_content.rstrip(),
                    chunk_type=ChunkType.MODULE,
                    start_line=module_start,
                    end_line=module_start + len(module_lines) - 1,
                )
            )

    return chunks if chunks else _chunk_window(content, config)


def _chunk_markdown(content: str, config: MembootConfig) -> list[ChunkResult]:
    """Split markdown on headers."""
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    chunks: list[ChunkResult] = []
    lines = content.split("\n")

    matches = list(header_pattern.finditer(content))
    if not matches:
        return _chunk_window(content, config)

    # Find line numbers for each match
    positions: list[tuple[int, str]] = []
    for match in matches:
        line_num = content[: match.start()].count("\n") + 1
        positions.append((line_num, match.group(0)))

    for i, (line_num, header) in enumerate(positions):
        end_line = positions[i + 1][0] - 1 if i + 1 < len(positions) else len(lines)

        section = "\n".join(lines[line_num - 1 : end_line]).strip()
        if section:
            chunks.append(
                ChunkResult(
                    content=section,
                    chunk_type=ChunkType.MARKDOWN_SECTION,
                    start_line=line_num,
                    end_line=end_line,
                    metadata={"header": header.lstrip("#").strip()},
                )
            )

    # Capture content before first header
    if positions and positions[0][0] > 1:
        preamble = "\n".join(lines[: positions[0][0] - 1]).strip()
        if preamble:
            chunks.insert(
                0,
                ChunkResult(
                    content=preamble,
                    chunk_type=ChunkType.MARKDOWN_SECTION,
                    start_line=1,
                    end_line=positions[0][0] - 1,
                    metadata={"header": "preamble"},
                ),
            )

    return chunks


def _chunk_yaml(content: str, config: MembootConfig) -> list[ChunkResult]:
    """Split YAML on top-level keys."""
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError:
        return _chunk_window(content, config)

    if not isinstance(data, dict):
        return _chunk_window(content, config)

    chunks: list[ChunkResult] = []
    lines = content.split("\n")

    # Find top-level key positions (lines starting with non-whitespace + colon)
    key_positions: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        match = re.match(r"^(\S+)\s*:", line)
        if match:
            key_positions.append((i, match.group(1)))

    for i, (line_num, key) in enumerate(key_positions):
        end_line = key_positions[i + 1][0] - 1 if i + 1 < len(key_positions) else len(lines)

        section = "\n".join(lines[line_num - 1 : end_line]).strip()
        if section:
            chunks.append(
                ChunkResult(
                    content=section,
                    chunk_type=ChunkType.YAML_KEY,
                    start_line=line_num,
                    end_line=end_line,
                    metadata={"key": key},
                )
            )

    return chunks if chunks else _chunk_window(content, config)


def _chunk_json(content: str, config: MembootConfig) -> list[ChunkResult]:
    """Split JSON on top-level keys."""
    import json as json_mod

    try:
        data = json_mod.loads(content)
    except (json_mod.JSONDecodeError, ValueError):
        return _chunk_window(content, config)

    if isinstance(data, dict):
        chunks: list[ChunkResult] = []
        for key, value in data.items():
            serialized = json_mod.dumps({key: value}, indent=2)
            chunks.append(
                ChunkResult(
                    content=serialized,
                    chunk_type=ChunkType.JSON_KEY,
                    start_line=1,
                    end_line=1,
                    metadata={"key": key},
                )
            )
        return chunks if chunks else _chunk_window(content, config)

    return _chunk_window(content, config)


def _chunk_window(content: str, config: MembootConfig) -> list[ChunkResult]:
    """Sliding window chunking for arbitrary text."""
    lines = content.split("\n")
    if not lines:
        return []

    # Estimate chars per chunk (~4 chars per token)
    chars_per_chunk = config.max_chunk_tokens * 4
    overlap_chars = config.overlap_tokens * 4

    chunks: list[ChunkResult] = []
    current_start = 0
    total_chars = len(content)

    while current_start < total_chars:
        end = min(current_start + chars_per_chunk, total_chars)
        chunk_text = content[current_start:end]

        # Find line numbers
        start_line = content[:current_start].count("\n") + 1
        end_line = content[:end].count("\n") + 1

        if chunk_text.strip():
            chunks.append(
                ChunkResult(
                    content=chunk_text.strip(),
                    chunk_type=ChunkType.WINDOW,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        if end >= total_chars:
            break
        current_start = end - overlap_chars

    return chunks
