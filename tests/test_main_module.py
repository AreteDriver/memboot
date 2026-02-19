"""Tests for memboot.__main__."""

from __future__ import annotations

import runpy

import pytest


class TestMainModule:
    def test_run_module(self):
        with pytest.raises(SystemExit):
            runpy.run_module("memboot", run_name="__main__")
