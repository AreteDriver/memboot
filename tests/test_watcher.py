"""Tests for memboot.watcher."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from memboot.exceptions import MembootError


class TestWatchProjectImportError:
    def test_missing_watchdog(self):
        mods = {
            "watchdog": None,
            "watchdog.events": None,
            "watchdog.observers": None,
        }
        with (
            patch.dict("sys.modules", mods),
            pytest.raises(MembootError, match="watchdog"),
        ):
            from importlib import reload

            import memboot.watcher

            reload(memboot.watcher)
            memboot.watcher.watch_project(Path("."))


class TestWatcherHandler:
    """Test the file event handler logic."""

    def test_relevant_file_triggers_reindex(self, tmp_path: Path):
        from memboot.watcher import watch_project

        project = tmp_path / "proj"
        project.mkdir()
        (project / "main.py").write_text('def hello(): return "hi"\n')

        db_dir = tmp_path / ".memboot"
        reindex_called = threading.Event()
        reindex_info = {}

        def on_reindex(info):
            reindex_info["info"] = info
            reindex_called.set()

        def run_watcher():
            try:
                with patch(
                    "memboot.indexer.Path.expanduser",
                    lambda self: db_dir if str(self) == "~/.memboot" else Path.home(),
                ):
                    watch_project(
                        project,
                        debounce=0.3,
                        on_reindex=on_reindex,
                    )
            except Exception:
                pass

        t = threading.Thread(target=run_watcher, daemon=True)
        t.start()

        # Give observer time to start (CI runners are slower)
        time.sleep(1.0)

        # Modify a file
        (project / "main.py").write_text('def hello(): return "modified"\n')

        # Wait for debounced reindex (generous timeout for CI)
        got_it = reindex_called.wait(timeout=10.0)
        assert got_it, "Reindex callback was not called"
        assert "info" in reindex_info

    def test_irrelevant_file_ignored(self, tmp_path: Path):
        """Changes to non-indexed files (e.g., .log) don't trigger reindex."""
        from memboot.watcher import watch_project

        project = tmp_path / "proj"
        project.mkdir()
        (project / "main.py").write_text("x = 1\n")

        reindex_called = threading.Event()

        def on_reindex(info):
            reindex_called.set()

        t = threading.Thread(
            target=lambda: watch_project(project, debounce=0.2, on_reindex=on_reindex),
            daemon=True,
        )
        t.start()
        time.sleep(1.0)

        # Create a non-indexed file type
        (project / "debug.log").write_text("log entry\n")
        time.sleep(1.5)

        assert not reindex_called.is_set()

    def test_debounce_coalesces_changes(self, tmp_path: Path):
        """Multiple rapid changes should result in a single reindex."""
        from memboot.watcher import watch_project

        project = tmp_path / "proj"
        project.mkdir()
        (project / "main.py").write_text("x = 1\n")

        reindex_count = {"n": 0}
        lock = threading.Lock()

        def on_reindex(info):
            with lock:
                reindex_count["n"] += 1

        t = threading.Thread(
            target=lambda: watch_project(
                project,
                debounce=0.5,
                on_reindex=on_reindex,
            ),
            daemon=True,
        )
        t.start()
        time.sleep(1.0)

        # Rapid-fire changes
        for i in range(5):
            (project / "main.py").write_text(f"x = {i}\n")
            time.sleep(0.05)

        # Wait for debounce + reindex
        time.sleep(3.0)

        # Should have coalesced into 1-2 reindexes, not 5
        assert reindex_count["n"] <= 2


class TestWatchCLI:
    def test_watch_command_exists(self):
        from typer.testing import CliRunner

        from memboot.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "auto-reindex" in result.output.lower()

    def test_watch_missing_watchdog(self):
        from typer.testing import CliRunner

        from memboot.cli import app

        runner = CliRunner()
        with patch(
            "memboot.watcher.watch_project",
            side_effect=MembootError("watchdog not installed"),
        ):
            result = runner.invoke(app, ["watch", "."])
        assert result.exit_code == 1
        assert "watchdog" in result.output.lower() or "Error" in result.output
