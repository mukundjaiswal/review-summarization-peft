"""Shared fixtures. Offline: no GPU, no weights, no download."""

from __future__ import annotations

import pytest

from review_peft.data import Record
from review_peft.eval_metrics import register_task_metrics
from review_peft.settings import Settings


@pytest.fixture(autouse=True)
def _metrics_registered():
    """Register this task's metrics before each test.

    The registry is shared with sibling projects, so tests register explicitly
    rather than relying on an import having done it.
    """
    register_task_metrics(replace=True)


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings that ignore any .env on the developer's machine."""
    return Settings(_env_file=None, seed=42, eval_size=10)


@pytest.fixture
def records() -> list[Record]:
    """Thirty synthetic records."""
    return [
        Record(review=f"review {i} " * 10, summary=f"summary {i}", category="a")
        for i in range(30)
    ]
