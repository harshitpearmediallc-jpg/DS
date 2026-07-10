"""DataSource interface + canonical schema.

Everything downstream (backtest, confidence) speaks ONLY the canonical column
names defined here. Each concrete DataSource is responsible for translating the
real backend into these names using ``config.data_source.column_map``. This is
the seam that lets us swap CSV <-> SQL <-> a real tracker without touching the
statistics.

Read-only by contract: there are no write methods on this interface, and the
SQL implementation opens read-only transactions.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import pandas as pd


# ---- canonical column names (the ONLY names downstream code may use) --------

class CampaignCols:
    ID = "campaign_id"
    NAME = "name"
    OFFER = "offer"
    START = "start_time"
    END = "end_time"
    STATUS = "status"


class EventCols:
    ID = "campaign_id"
    TS = "timestamp"            # time the spend/click occurred
    POSTBACK_TS = "postback_time"  # time the conversion/revenue landed (optional)
    SPEND = "spend"
    CLICKS = "clicks"
    CONVERSIONS = "conversions"
    REVENUE = "revenue"
    # dimensions (features / corroborating evidence only)
    SUB_ID = "sub_id"
    CREATIVE = "creative"
    LANDER = "lander"
    OFFER = "offer"
    AUDIENCE = "audience"
    REGION = "region"
    CITY = "city"


DIMENSION_COLS = [
    EventCols.SUB_ID,
    EventCols.CREATIVE,
    EventCols.LANDER,
    EventCols.OFFER,
    EventCols.AUDIENCE,
    EventCols.REGION,
    EventCols.CITY,
]

# Metrics that must be numeric & additive.
ADDITIVE_METRICS = [
    EventCols.SPEND,
    EventCols.CLICKS,
    EventCols.CONVERSIONS,
    EventCols.REVENUE,
]


@dataclass
class SchemaValidationError(Exception):
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


class DataSource(abc.ABC):
    """Thin read-only interface. Two loads, both returning canonical dataframes."""

    @abc.abstractmethod
    def load_campaigns(self) -> pd.DataFrame:
        """Return campaigns with canonical CampaignCols columns (at least ID)."""

    @abc.abstractmethod
    def load_events(self, campaign_ids: list | None = None) -> pd.DataFrame:
        """Return event/stats time series with canonical EventCols columns.

        If ``campaign_ids`` is given, restrict to those campaigns (lets callers
        stream campaign-by-campaign instead of loading everything).
        """

    # -- shared helpers -------------------------------------------------------

    @staticmethod
    def _rename_to_canonical(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
        """Rename real columns -> canonical using {canonical: real} map.

        Only columns present in the frame are renamed; missing optional columns
        are simply absent downstream (callers guard for that).
        """
        # column_map is {canonical_name: real_name}; we invert to {real: canonical}
        inverse = {real: canon for canon, real in column_map.items() if real in df.columns}
        return df.rename(columns=inverse)

    @staticmethod
    def _validate_events(df: pd.DataFrame) -> pd.DataFrame:
        required = [EventCols.ID, EventCols.TS, EventCols.CONVERSIONS,
                    EventCols.SPEND, EventCols.REVENUE]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SchemaValidationError(
                f"events missing required canonical columns {missing}. "
                f"Fix config.data_source.column_map.events. Got: {list(df.columns)}"
            )
        # coerce metric types; non-additive/text metrics are a mapping bug.
        for col in ADDITIVE_METRICS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[EventCols.TS] = pd.to_datetime(df[EventCols.TS], errors="coerce")
        if EventCols.POSTBACK_TS in df.columns:
            df[EventCols.POSTBACK_TS] = pd.to_datetime(
                df[EventCols.POSTBACK_TS], errors="coerce"
            )
        return df

    @staticmethod
    def _validate_campaigns(df: pd.DataFrame) -> pd.DataFrame:
        if CampaignCols.ID not in df.columns:
            raise SchemaValidationError(
                f"campaigns missing required '{CampaignCols.ID}'. "
                f"Fix config.data_source.column_map.campaigns. Got: {list(df.columns)}"
            )
        for tcol in (CampaignCols.START, CampaignCols.END):
            if tcol in df.columns:
                df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        return df
