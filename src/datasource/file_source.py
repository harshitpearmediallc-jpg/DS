"""File-backed DataSource: CSV or parquet on disk.

This is the offline / synthetic-data implementation used for all development and
testing. It expects two files in ``data_source.file.path``:

    campaigns.(csv|parquet)
    events.(csv|parquet)

with columns named per ``data_source.column_map`` (for synthetic data these
already match the canonical names, so the map is an identity).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Cfg, resolve_path
from .base import DataSource, EventCols


class FileDataSource(DataSource):
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self._dir = resolve_path(cfg, cfg.data_source.file.path)
        self._fmt = cfg.data_source.file.format
        self._colmap = cfg.data_source.column_map

    def _read(self, stem: str) -> pd.DataFrame:
        path = self._dir / f"{stem}.{self._fmt}"
        if not path.exists():
            raise FileNotFoundError(
                f"expected {path} — run the synthetic generator or point "
                f"data_source.file.path at real exports."
            )
        if self._fmt == "parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def load_campaigns(self) -> pd.DataFrame:
        df = self._read("campaigns")
        df = self._rename_to_canonical(df, self._colmap.campaigns.to_dict())
        return self._validate_campaigns(df)

    def load_events(self, campaign_ids: list | None = None) -> pd.DataFrame:
        df = self._read("events")
        df = self._rename_to_canonical(df, self._colmap.events.to_dict())
        df = self._validate_events(df)
        if campaign_ids is not None:
            df = df[df[EventCols.ID].isin(campaign_ids)]
        return df.sort_values([EventCols.ID, EventCols.TS]).reset_index(drop=True)
