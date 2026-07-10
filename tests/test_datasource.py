"""DataSource layer: file roundtrip + SQL mapping/read-only against SQLite.

The important property: no tracker column name is hardcoded — a non-identity
column_map still yields canonical frames downstream.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.config import Cfg
from src.datasource import EventCols, CampaignCols
from src.datasource.sql_source import SqlDataSource
from src.datasource.base import SchemaValidationError


def _sql_cfg():
    # column_map uses REAL names (left=canonical, right=real) that differ from canonical
    return Cfg({
        "data_source": {
            "kind": "sql",
            "sql": {"url": "sqlite://", "campaigns_table": "camp_tbl",
                    "events_table": "stats_tbl"},
            "column_map": {
                "campaigns": {"campaign_id": "cid", "name": "cname", "offer": "off",
                              "start_time": "st", "end_time": "et", "status": "stat"},
                "events": {"campaign_id": "cid", "timestamp": "spend_ts",
                           "postback_time": "pb_ts", "spend": "cost",
                           "clicks": "clk", "conversions": "cv", "revenue": "rev",
                           "sub_id": "subid", "creative": "cr", "lander": "lp",
                           "offer": "off", "audience": "aud", "region": "reg",
                           "city": "cty"},
            },
        }
    })


@pytest.fixture
def sqlite_engine():
    eng = create_engine("sqlite://")
    camp = pd.DataFrame({
        "cid": ["A", "B"], "cname": ["camp A", "camp B"], "off": ["o1", "o2"],
        "st": ["2025-01-01", "2025-01-01"], "et": [None, None], "stat": ["active", "active"],
    })
    stats = pd.DataFrame({
        "cid": ["A", "A", "B"],
        "spend_ts": ["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 00:00"],
        "pb_ts": [None, "2025-01-01 03:00", "2025-01-01 02:00"],
        "cost": [10.0, 0.0, 5.0], "clk": [100, 0, 50],
        "cv": [0, 1, 1], "rev": [0.0, 25.0, 8.0],
        "subid": ["s1", "s1", "s2"], "cr": ["c", "c", "c"], "lp": ["l", "l", "l"],
        "off": ["o1", "o1", "o2"], "aud": ["a", "a", "a"],
        "reg": ["r", "r", "r"], "cty": ["x", "x", "y"],
    })
    camp.to_sql("camp_tbl", eng, index=False)
    stats.to_sql("stats_tbl", eng, index=False)
    return eng


def test_sql_source_returns_canonical_columns(sqlite_engine):
    ds = SqlDataSource(_sql_cfg(), engine=sqlite_engine)
    camps = ds.load_campaigns()
    assert CampaignCols.ID in camps.columns
    assert set(camps[CampaignCols.ID]) == {"A", "B"}

    events = ds.load_events()
    for col in (EventCols.ID, EventCols.TS, EventCols.SPEND,
                EventCols.CONVERSIONS, EventCols.REVENUE):
        assert col in events.columns
    # metrics coerced numeric, timestamps parsed
    assert events[EventCols.SPEND].dtype.kind == "f"
    assert str(events[EventCols.TS].dtype).startswith("datetime")


def test_sql_source_filters_by_campaign(sqlite_engine):
    ds = SqlDataSource(_sql_cfg(), engine=sqlite_engine)
    events = ds.load_events(campaign_ids=["B"])
    assert set(events[EventCols.ID]) == {"B"}


def test_missing_required_column_raises(sqlite_engine):
    cfg = _sql_cfg()
    # break the map: point 'conversions' at a column that doesn't exist -> AS alias
    # still succeeds in SQL but we simulate a real missing mapping by dropping it
    bad = cfg.to_dict()
    del bad["data_source"]["column_map"]["events"]["conversions"]
    ds = SqlDataSource(Cfg(bad), engine=sqlite_engine)
    with pytest.raises(SchemaValidationError):
        ds.load_events()


def test_file_source_roundtrip(tmp_path):
    from src.datasource.file_source import FileDataSource
    from src.synthetic import generate, GenConfig

    campaigns, events, _ = generate(GenConfig(n_campaigns=10, seed=1))
    (tmp_path / "d").mkdir()
    campaigns.to_parquet(tmp_path / "d" / "campaigns.parquet", index=False)
    events.to_parquet(tmp_path / "d" / "events.parquet", index=False)

    cfg = Cfg({
        "_project_root": str(tmp_path),
        "data_source": {
            "kind": "file",
            "file": {"path": "d", "format": "parquet"},
            "column_map": {  # identity map (synthetic already canonical)
                "campaigns": {c: c for c in ["campaign_id", "name", "offer",
                                             "start_time", "end_time", "status"]},
                "events": {c: c for c in ["campaign_id", "timestamp", "postback_time",
                                          "spend", "clicks", "conversions", "revenue",
                                          "sub_id", "creative", "lander", "offer",
                                          "audience", "region", "city"]},
            },
        },
    })
    ds = FileDataSource(cfg)
    ev = ds.load_events()
    assert len(ev) == len(events)
    assert EventCols.CONVERSIONS in ev.columns
