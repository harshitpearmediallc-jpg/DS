"""Live schema introspection + heuristic auto-mapping.

When we connect to a real tracker we do NOT want to guess the schema from a doc.
Instead we read the actual tables/columns and propose a ``column_map`` that the
user confirms. Everything here is READ-ONLY: it queries the catalog and pulls a
handful of sample rows, nothing more.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from .base import CampaignCols, EventCols
from .sql_source import resolve_url


# canonical fields we want to fill, grouped by logical table
CAMPAIGN_FIELDS = ["campaign_id", "name", "offer", "start_time", "end_time", "status"]
EVENT_FIELDS = ["campaign_id", "timestamp", "postback_time", "spend", "clicks",
                "conversions", "revenue", "sub_id", "creative", "lander", "offer",
                "audience", "region", "city"]

# synonyms the tracker might use for each canonical field (lowercased, no separators)
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


@dataclass
class TableInfo:
    schema: str
    name: str
    columns: list[str]
    column_types: dict
    approx_rows: int | None = None
    sample: pd.DataFrame | None = None

    @property
    def fqname(self) -> str:
        return f"{self.schema}.{self.name}" if self.schema else self.name


def probe_connection(url: str) -> dict:
    """Try to connect. Returns {ok, version, error}. Never raises."""
    try:
        eng = create_engine(resolve_url(url))
        with eng.connect() as conn:
            ver = conn.execute(text("SELECT version()")).scalar()
        return {"ok": True, "version": str(ver), "error": None, "engine": eng}
    except Exception as exc:  # broad: we want the reason surfaced to the user
        return {"ok": False, "version": None, "error": f"{type(exc).__name__}: {exc}",
                "engine": None}


def list_tables(engine, schemas: list[str] | None = None) -> list[TableInfo]:
    """List tables (and views) with their columns across the given schemas."""
    insp = inspect(engine)
    schemas = schemas or [s for s in insp.get_schema_names()
                          if s not in ("information_schema", "pg_catalog")]
    out: list[TableInfo] = []
    for sch in schemas:
        names = set(insp.get_table_names(schema=sch)) | set(insp.get_view_names(schema=sch))
        for tbl in sorted(names):
            cols = insp.get_columns(tbl, schema=sch)
            out.append(TableInfo(
                schema=sch, name=tbl,
                columns=[c["name"] for c in cols],
                column_types={c["name"]: str(c["type"]) for c in cols},
            ))
    return out


def sample_table(engine, info: TableInfo, n: int = 5) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(f'SELECT * FROM {info.fqname} LIMIT :n'),
                           conn, params={"n": n})


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _best_column(field: str, columns: list[str]) -> tuple[str | None, float]:
    """Return (best_column, score) matching a canonical field to real columns."""
    cand_syn = SYNONYMS.get(field, [field])
    norm_cols = {c: _norm(c) for c in columns}
    best, best_score = None, 0.0
    for col, ncol in norm_cols.items():
        # exact/synonym containment gets top score
        score = 0.0
        for syn in cand_syn:
            if ncol == syn:
                score = max(score, 1.0)
            elif syn in ncol or ncol in syn:
                score = max(score, 0.85)
            else:
                score = max(score, difflib.SequenceMatcher(None, ncol, syn).ratio())
        if score > best_score:
            best, best_score = col, score
    return best, best_score


def guess_table(fields: list[str], tables: list[TableInfo]) -> TableInfo | None:
    """Pick the table whose columns best cover the required canonical fields."""
    best, best_cov = None, -1.0
    for t in tables:
        cov = sum(_best_column(f, t.columns)[1] for f in fields) / len(fields)
        if cov > best_cov:
            best, best_cov = t, cov
    return best


def propose_column_map(engine, threshold: float = 0.6) -> dict:
    """Introspect and return a proposed column_map + the chosen tables + scores.

    Returns a dict ready to drop under config.data_source.column_map, plus
    metadata (chosen table names, per-field confidence) for the user to confirm.
    """
    tables = list_tables(engine)
    camp_tbl = guess_table(["campaign_id", "name", "status", "offer"], tables)
    evt_tbl = guess_table(["campaign_id", "conversions", "spend", "revenue"], tables)

    def _map_for(tbl: TableInfo, fields: list[str]) -> tuple[dict, dict]:
        mapping, scores = {}, {}
        for f in fields:
            col, sc = _best_column(f, tbl.columns) if tbl else (None, 0.0)
            scores[f] = round(sc, 2)
            # only auto-fill confident matches; leave weak ones for the user
            mapping[f] = col if sc >= threshold else None
        return mapping, scores

    camp_map, camp_scores = _map_for(camp_tbl, CAMPAIGN_FIELDS)
    evt_map, evt_scores = _map_for(evt_tbl, EVENT_FIELDS)

    return {
        "campaigns_table": camp_tbl.fqname if camp_tbl else None,
        "events_table": evt_tbl.fqname if evt_tbl else None,
        "column_map": {"campaigns": camp_map, "events": evt_map},
        "confidence": {"campaigns": camp_scores, "events": evt_scores},
        "all_tables": [t.fqname for t in tables],
    }
