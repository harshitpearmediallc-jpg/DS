"""Shared fixtures. A small synthetic dataset generated once per test session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# make the repo root importable (so `import src...` works under pytest)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.synthetic import generate, GenConfig


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def small_dataset():
    """A modest, fully-known synthetic dataset for fast statistical assertions."""
    gc = GenConfig(n_campaigns=120, seed=7)
    campaigns, events, truth = generate(gc)
    return campaigns, events, truth
