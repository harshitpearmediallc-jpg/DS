#!/usr/bin/env python3
"""One-shot tracker exporter — run this on ANY machine that can reach the DB.

The Claude cloud sandbox cannot open a Postgres connection (its network policy
blocks port 5432), so we export the two tables to files there and load them into
the tool as a FileDataSource. This script:

  1. connects to the DB (read-only SELECTs only),
  2. introspects every table/column and writes `schema_report.json`,
  3. auto-detects the `campaigns` and `events` tables by column names,
  4. exports those two tables to parquet (falls back to CSV),
  5. prints what it chose so you can sanity-check.

Then upload the produced files (schema_report.json + the two data files) back to
the chat. If the auto-detection picks the wrong tables, just send me
schema_report.json and I'll tell you exactly which tables to export.

USAGE
    pip install "sqlalchemy>=2" psycopg2-binary pandas pyarrow
    export TRACKER_DB_URL='postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME'
    python export_tracker.py                 # auto-detect + export
    python export_tracker.py --schema-only   # just the schema report, no data
    python export_tracker.py --table public.my_stats --table public.my_campaigns
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, inspect, text

# canonical fields + the tracker synonyms we match against (lowercased, alnum only)
SYNONYMS = {
    "campaign_id": ["campaignid", "campaign", "campid", "cid", "campaignuuid"],
    "name": ["name", "campaignname", "title", "label"],
    "offer": ["offer", "offername", "offerid", "product"],
    "start_time": ["starttime", "startdate", "createdat", "created", "launchdate"],
    "end_time": ["endtime", "enddate", "stoppedat", "archivedat", "closedat"],
    "status": ["status", "state", "campaignstatus", "active"],
    "timestamp": ["timestamp", "time", "datetime", "hour", "date", "eventtime",
                  "clicktime", "spendtime", "statdate", "day"],
    "postback_time": ["postbacktime", "postback", "conversiontime", "convtime",
                      "conversiondate", "payouttime", "posttime"],
    "spend": ["spend", "cost", "adspend", "mediaspend", "costusd"],
    "clicks": ["clicks", "click", "clickcount", "visits"],
    "conversions": ["conversions", "conversion", "convs", "conv", "leads",
                    "sales", "actions"],
    "revenue": ["revenue", "payout", "rev", "income", "earnings", "payoutusd"],
    "sub_id": ["subid", "sub", "subid1", "sourceid", "sub1"],
    "creative": ["creative", "ad", "adid", "creativeid", "banner"],
    "lander": ["lander", "landingpage", "landerid", "lp", "page"],
    "audience": ["audience", "segment", "targeting", "audienceid"],
    "region": ["region", "state", "geo", "province", "regionname"],
    "city": ["city", "cityname", "town", "dma"],
}
CAMPAIGN_KEY = ["campaign_id", "name", "status", "offer"]
EVENT_KEY = ["campaign_id", "conversions", "spend", "revenue"]


def norm(s): return "".join(c for c in s.lower() if c.isalnum())


def best_col(field, columns):
    best, score = None, 0.0
    for col in columns:
        n = norm(col)
        for syn in SYNONYMS.get(field, [field]):
            if n == syn: s = 1.0
            elif syn in n or n in syn: s = 0.85
            else: s = difflib.SequenceMatcher(None, n, syn).ratio()
            if s > score: best, score = col, s
    return best, round(score, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema-only", action="store_true")
    ap.add_argument("--table", action="append", default=[],
                    help="explicit table(s) to export, e.g. public.campaign_stats")
    ap.add_argument("--limit", type=int, default=None,
                    help="row cap per table (omit to export everything)")
    args = ap.parse_args()

    url = os.environ.get("TRACKER_DB_URL")
    if not url:
        sys.exit("Set TRACKER_DB_URL (postgresql+psycopg2://user:pw@host:port/db)")
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)

    eng = create_engine(url)
    insp = inspect(eng)
    with eng.connect() as c:
        print("connected:", c.execute(text("select version()")).scalar()[:60])

    # 1-2. introspect
    report = {"tables": []}
    schemas = [s for s in insp.get_schema_names()
               if s not in ("information_schema", "pg_catalog")]
    all_tables = []
    for sch in schemas:
        names = set(insp.get_table_names(schema=sch)) | set(insp.get_view_names(schema=sch))
        for t in sorted(names):
            cols = insp.get_columns(t, schema=sch)
            fq = f"{sch}.{t}"
            all_tables.append((fq, [col["name"] for col in cols]))
            try:
                with eng.connect() as c:
                    n = c.execute(text(f"SELECT count(*) FROM {fq}")).scalar()
                    sample = pd.read_sql(text(f"SELECT * FROM {fq} LIMIT 3"), c)
            except Exception as e:
                n, sample = None, None
            report["tables"].append({
                "table": fq,
                "columns": {col["name"]: str(col["type"]) for col in cols},
                "row_count": int(n) if n is not None else None,
                "sample": sample.astype(str).to_dict("records") if sample is not None else [],
            })
    with open("schema_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote schema_report.json ({len(all_tables)} tables)")

    # 3. auto-detect campaigns + events tables
    def coverage(keyfields, cols):
        return sum(best_col(f, cols)[1] for f in keyfields) / len(keyfields)

    if args.table:
        chosen = {"explicit": args.table}
    else:
        camp = max(all_tables, key=lambda tc: coverage(CAMPAIGN_KEY, tc[1]))
        evt = max(all_tables, key=lambda tc: coverage(EVENT_KEY, tc[1]))
        chosen = {"campaigns": camp[0], "events": evt[0]}
        print(f"auto-detected campaigns table: {camp[0]} "
              f"(coverage {coverage(CAMPAIGN_KEY, camp[1]):.2f})")
        print(f"auto-detected events table:    {evt[0]} "
              f"(coverage {coverage(EVENT_KEY, evt[1]):.2f})")
        # proposed column map for the events table (for your review)
        proposed = {f: best_col(f, evt[1]) for f in SYNONYMS}
        print("proposed events column map (field -> (column, confidence)):")
        for f, (col, sc) in proposed.items():
            print(f"    {f:14s} -> {col} ({sc})")

    if args.schema_only:
        print("schema-only mode: done. Upload schema_report.json.")
        return

    # 4. export chosen tables
    targets = args.table if args.table else [chosen["campaigns"], chosen["events"]]
    for fq in dict.fromkeys(targets):  # dedupe, keep order
        q = f"SELECT * FROM {fq}" + (f" LIMIT {args.limit}" if args.limit else "")
        with eng.connect() as c:
            df = pd.read_sql(text(q), c)
        safe = fq.replace(".", "__")
        try:
            out = f"export_{safe}.parquet"
            df.to_parquet(out, index=False)
        except Exception:
            out = f"export_{safe}.csv"
            df.to_csv(out, index=False)
        print(f"exported {fq}: {len(df):,} rows, {len(df.columns)} cols -> {out}")

    print("\nDONE. Upload schema_report.json and the export_*.{parquet,csv} files "
          "back to the chat.")


if __name__ == "__main__":
    main()
