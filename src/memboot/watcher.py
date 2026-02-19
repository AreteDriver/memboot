"""File system watcher for auto-reindexing."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from memboot.exceptions import MembootError
from memboot.models import MembootConfig


def watch_project(
    project_path: Path,
    config: MembootConfig | None = None,
    debounce: float = 2.0,
    on_reindex: callable | None = None,
) -> None:
    """Watch a project directory and auto-reindex on changes.

    Args:
        project_path: Project root to watch.
        config: Indexing configuration.
        debounce: Seconds to wait after last change before reindexing.
        on_reindex: Optional callback(ProjectInfo) after each reindex.

    Raises:
        MembootError: If watchdog is not installed.
    """
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError as exc:
        raise MembootError(
            "Watch mode requires watchdog. Install with: pip install memboot[watch]"
        ) from exc

    from memboot.indexer import index_project

    config = config or MembootConfig()
    project_path = project_path.resolve()
    extensions = set(config.file_extensions)

    timer: threading.Timer | None = None
    lock = threading.Lock()

    def _do_reindex():
        nonlocal timer
        try:
            info = index_project(project_path, config=config)
            if on_reindex:
                on_reindex(info)
        except Exception:
            pass  # Callback handles display; don't crash the watcher

    def _schedule_reindex():
        nonlocal timer
        with lock:
            if timer is not None:
                timer.cancel()
            timer = threading.Timer(debounce, _do_reindex)
            timer.daemon = True
            timer.start()

    class _Handler(FileSystemEventHandler):
        def _is_relevant(self, path: str) -> bool:
            p = Path(path)
            if p.suffix.lower() not in extensions:
                return False
            try:
                rel = p.relative_to(project_path)
            except ValueError:
                return False
            parts = rel.parts
            ignore = set(config.ignore_patterns)
            return not any(part in ignore for part in parts)

        def on_created(self, event):
            if not event.is_directory and self._is_relevant(event.src_path):
                _schedule_reindex()

        def on_modified(self, event):
            if not event.is_directory and self._is_relevant(event.src_path):
                _schedule_reindex()

        def on_deleted(self, event):
            if not event.is_directory and self._is_relevant(event.src_path):
                _schedule_reindex()

    observer = Observer()
    observer.schedule(_Handler(), str(project_path), recursive=True)
    observer.start()

    try:
        while observer.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        with lock:
            if timer is not None:
                timer.cancel()
