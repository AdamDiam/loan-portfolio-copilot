from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.data.generate import generate, write_sqlite  # noqa: E402


@pytest.fixture(scope="session")
def small_db(tmp_path_factory) -> Path:
    """A small, deterministic portfolio shared by all tests."""
    path = tmp_path_factory.mktemp("data") / "test_portfolio.db"
    write_sqlite(generate(n_loans=3000, seed=7), path)
    return path
