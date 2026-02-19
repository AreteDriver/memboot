"""Tests for memboot.gates."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import typer

from memboot.gates import require_pro


def _dummy_func():
    """A dummy function for testing."""
    return "success"


class TestRequirePro:
    def test_allowed_with_feature(self):
        with patch("memboot.gates.has_feature", return_value=True):
            decorated = require_pro("serve")(_dummy_func)
            result = decorated()
            assert result == "success"

    def test_blocked_without_feature(self):
        with patch("memboot.gates.has_feature", return_value=False):
            decorated = require_pro("serve")(_dummy_func)
            with pytest.raises(typer.Exit):
                decorated()

    def test_preserves_function_name(self):
        with patch("memboot.gates.has_feature", return_value=True):
            decorated = require_pro("serve")(_dummy_func)
            assert decorated.__name__ == "_dummy_func"

    def test_passes_args_through(self):
        def func_with_args(x, y=10):
            return x + y

        with patch("memboot.gates.has_feature", return_value=True):
            decorated = require_pro("serve")(func_with_args)
            result = decorated(5, y=20)
            assert result == 25
