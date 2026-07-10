"""DataSource factory + canonical schema exports."""

from __future__ import annotations

from ..config import Cfg
from .base import (
    DataSource,
    CampaignCols,
    EventCols,
    DIMENSION_COLS,
    ADDITIVE_METRICS,
    SchemaValidationError,
)
from .file_source import FileDataSource
from .sql_source import SqlDataSource


def make_datasource(cfg: Cfg) -> DataSource:
    """Instantiate the DataSource named by config.data_source.kind."""
    kind = cfg.data_source.kind
    if kind == "file":
        return FileDataSource(cfg)
    if kind == "sql":
        return SqlDataSource(cfg)
    raise ValueError(f"unknown data_source.kind '{kind}' (expected 'file' or 'sql')")


__all__ = [
    "DataSource",
    "FileDataSource",
    "SqlDataSource",
    "make_datasource",
    "CampaignCols",
    "EventCols",
    "DIMENSION_COLS",
    "ADDITIVE_METRICS",
    "SchemaValidationError",
]
