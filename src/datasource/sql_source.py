"""SQL DataSource: read-only SQLAlchemy access to a tracker replica.

Deliberately conservative:
  * builds SELECTs from the column_map so no tracker column name is hardcoded;
  * opens the connection in a read-only fashion and never issues writes/DDL;
  * this is the ONLY module that talks SQL. Everything else sees canonical frames.

The exact SQL dialect quirks (schema-qualified names, quoting) depend on the real
tracker; those get finalised once we have the real schema (see SCHEMA_ASSUMPTIONS.md).
Until then this is exercised against a SQLite fixture in the tests to prove the
mapping + read-only contract hold.
"""

from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine, text

from ..config import Cfg
from .base import DataSource, EventCols


def resolve_url(raw: str) -> str:
    """Resolve a config URL that may reference an environment variable.

    Secrets should NEVER be written into config.yaml. Set the URL in config to
    ``env:TRACKER_DB_URL`` and export the real connection string in the
    environment; this reads it back at connect time so the credential is never
    persisted to disk in the repo.
    """
    if raw and raw.startswith("env:"):
        var = raw[len("env:"):]
        val = os.environ.get(var)
        if not val:
            raise RuntimeError(
                f"config points at env:{var} but ${var} is not set. "
                f"Export the read-only connection string into that variable."
            )
        return val
    return raw


class SqlDataSource(DataSource):
    def __init__(self, cfg: Cfg, engine=None):
        self.cfg = cfg
        self._colmap = cfg.data_source.column_map
        self._campaigns_table = cfg.data_source.sql.campaigns_table
        self._events_table = cfg.data_source.sql.events_table
        # engine injectable for testing (e.g. a SQLite fixture).
        self._engine = engine or create_engine(
            resolve_url(cfg.data_source.sql.url),
            # execution_options isolation ensures we never accidentally commit.
            execution_options={"readonly": True},
        )

    def _select_columns(self, canonical_to_real: dict) -> str:
        # emit  real_name AS canonical_name  so the frame comes back canonical.
        parts = [f'{real} AS {canon}' for canon, real in canonical_to_real.items()]
        return ", ".join(parts)

    def _read_sql(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        # Read-only: we only ever run SELECTs and never call commit().
        with self._engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})

    def load_campaigns(self) -> pd.DataFrame:
        cols = self._select_columns(self._colmap.campaigns.to_dict())
        df = self._read_sql(f"SELECT {cols} FROM {self._campaigns_table}")
        return self._validate_campaigns(df)

    def load_events(self, campaign_ids: list | None = None) -> pd.DataFrame:
        cols = self._select_columns(self._colmap.events.to_dict())
        sql = f"SELECT {cols} FROM {self._events_table}"
        params: dict = {}
        if campaign_ids is not None:
            id_real = self._colmap.events.to_dict()["campaign_id"]
            # named-parameter IN clause; kept simple + parameterised (no f-string ids).
            placeholders = ", ".join(f":id{i}" for i in range(len(campaign_ids)))
            params = {f"id{i}": cid for i, cid in enumerate(campaign_ids)}
            sql += f" WHERE {id_real} IN ({placeholders})"
        df = self._read_sql(sql, params)
        df = self._validate_events(df)
        return df.sort_values([EventCols.ID, EventCols.TS]).reset_index(drop=True)
