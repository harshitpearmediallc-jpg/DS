"""Adapt the exported spend_revenue CSV into the tool's canonical parquet files.

The real tracker schema differs from the tool's canonical schema in three ways
(see the conversation / SCHEMA_ASSUMPTIONS.md):
  * time is split across `date` + `hour`  -> we synthesise a single `timestamp`
  * rows are aggregated (spend+clicks+conversions+revenue per row), no postback
  * dimensions use tracker names (redtrack_sub3, source, vertical_code, ...)

Output: data/real/{campaigns,events}.parquet with canonical column names, ready
for the FileDataSource (identity column_map). Read-only w.r.t. the source; this
only transforms an already-exported CSV.

USAGE
    python prepare_real_data.py path/to/spend_revenue_mature.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "data" / "real"

# real CSV column -> canonical event column
EVENT_MAP = {
    "campaign_id": "campaign_id",
    "spend": "spend",
    "revenue": "revenue",
    "conversions": "conversions",
    "clicks": "clicks",
    "redtrack_sub3": "sub_id",
    "source": "source",
    "vertical_code": "offer",       # closest analogue; dimension only
    "platform_code": "audience",    # dimension only
    "ad_account_id": "ad_account",  # extra dimension, carried through
}


def prepare(csv_path: str) -> Path:
    df = pd.read_csv(csv_path, low_memory=False,
                     dtype={"campaign_id": "string", "redtrack_sub3": "string"})
    df.columns = [c.strip().lower() for c in df.columns]
    # drop rows with no campaign id (can't attribute) and normalise id to str
    df = df[df["campaign_id"].notna() & (df["campaign_id"].astype(str).str.len() > 0)]

    required = ["date", "hour", "campaign_id", "spend", "revenue", "conversions"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing expected columns {missing}. Got: {list(df.columns)}")

    # --- build a single timestamp from date + hour ---------------------------
    date = pd.to_datetime(df["date"], errors="coerce")
    hour = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int).clip(0, 23)
    df["timestamp"] = date + pd.to_timedelta(hour, unit="h")

    # --- coerce numeric metrics ----------------------------------------------
    for col in ["spend", "revenue", "conversions", "clicks", "impressions"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["campaign_id"] = df["campaign_id"].astype(str)

    # --- canonical events frame ----------------------------------------------
    events = pd.DataFrame({"campaign_id": df["campaign_id"], "timestamp": df["timestamp"]})
    for real, canon in EVENT_MAP.items():
        if real in df.columns and real != "campaign_id":
            events[canon] = df[real]
    # no postback_time in this source -> omit the column entirely; the
    # reconstruction treats its absence as "already click-attributed" (observed
    # ROI == aligned ROI, no separable lag).
    events = events.dropna(subset=["timestamp"])
    events = events.sort_values(["campaign_id", "timestamp"]).reset_index(drop=True)

    # --- canonical campaigns frame (derive from the stats rows) --------------
    name_col = "campaign_name" if "campaign_name" in df.columns else None
    status_col = "campaign_status" if "campaign_status" in df.columns else None
    agg = {}
    if name_col:
        agg["name"] = (name_col, "first")
    if status_col:
        agg["status"] = (status_col, "first")
    agg["start_time"] = ("timestamp", "min")
    agg["end_time"] = ("timestamp", "max")
    campaigns = df.groupby("campaign_id").agg(**agg).reset_index()
    if "name" not in campaigns.columns:
        campaigns["name"] = campaigns["campaign_id"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(OUT_DIR / "events.parquet", index=False)
    campaigns.to_parquet(OUT_DIR / "campaigns.parquet", index=False)

    # --- report --------------------------------------------------------------
    total_conv = int(events["conversions"].sum())
    print(f"rows: {len(events):,}  campaigns: {events['campaign_id'].nunique():,}")
    print(f"date span: {events['timestamp'].min()} -> {events['timestamp'].max()}")
    print(f"total conversions: {total_conv:,}  total spend: ${df['spend'].sum():,.0f}  "
          f"total revenue: ${df['revenue'].sum():,.0f}")
    # grain hint
    per_day = df.groupby(["campaign_id", date.dt.date]).size()
    print(f"median rows per campaign-day: {per_day.median():.1f} "
          f"(1 ~ daily grain, ~24 ~ hourly)")
    print(f"wrote {OUT_DIR}/events.parquet and campaigns.parquet")
    return OUT_DIR


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python prepare_real_data.py path/to/spend_revenue_mature.csv")
    prepare(sys.argv[1])
