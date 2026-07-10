"""Config loading. One YAML file is the single source of truth for tunables.

We wrap the raw dict in a tiny dot-accessible object so downstream code reads
``cfg.backtest.decision_conversion_points`` instead of stringly-typed dict keys,
which makes typos fail loudly at import sites rather than silently returning None.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Cfg:
    """Recursive dot-access wrapper over a plain dict.

    Kept deliberately minimal — it is a *view* onto the YAML, not a schema. We
    do not validate here; validation lives where a value is used, so error
    messages point at the statistical choice that cares about the value.
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        try:
            val = self._data[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"config key '{key}' not found. Available: {list(self._data)}"
            ) from exc
        return Cfg(val) if isinstance(val, dict) else val

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        return Cfg(val) if isinstance(val, dict) else val

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:  # pragma: no cover
        return f"Cfg({self._data!r})"


def _project_root() -> Path:
    # src/config.py -> project root is one directory up from src/
    return Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> Cfg:
    """Load config.yaml. Defaults to the repo-root config.yaml.

    Paths inside the config that are relative (e.g. data/synthetic) are resolved
    relative to the project root so the tool works regardless of the current
    working directory.
    """
    root = _project_root()
    cfg_path = Path(path) if path else root / "config.yaml"
    with open(cfg_path, "r") as fh:
        raw = yaml.safe_load(fh)
    raw["_project_root"] = str(root)
    return Cfg(raw)


def resolve_path(cfg: Cfg, relative: str) -> Path:
    """Resolve a possibly-relative path against the project root."""
    root = Path(cfg.to_dict()["_project_root"])
    p = Path(relative)
    return p if p.is_absolute() else root / p
