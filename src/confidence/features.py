"""Shared helpers to turn a campaign's events into the quantities Phase 2 needs.

Works on the same canonical event frame the backtest uses, and — crucially —
supports TRUNCATING a campaign to the state it was in at its N-th conversion, so
the exact same code path serves a live "decide now" call and a historical
backtest/calibration point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..datasource import EventCols


@dataclass
class CampaignState:
    campaign_id: str
    n_conversions: int
    spend: float
    revenue: float
    roi: float
    payouts: np.ndarray          # per-conversion payout sample (length ~ n)
    events: pd.DataFrame         # the (possibly truncated) events used


def _conv_time(conv: pd.DataFrame) -> pd.Series:
    if EventCols.POSTBACK_TS in conv.columns:
        return conv[EventCols.POSTBACK_TS].fillna(conv[EventCols.TS])
    return conv[EventCols.TS]


def truncate_to_n(events: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """Return the events a buyer would have seen by the campaign's N-th conversion.

    Spend rows up to the decision time and conversion rows up to N conversions.
    If n is None, returns the full history (a live "decide now" call).
    """
    if n is None:
        return events
    conv = events[events[EventCols.CONVERSIONS] > 0].copy()
    if conv.empty:
        return events.iloc[0:0]
    conv["_ct"] = _conv_time(conv)
    conv = conv.sort_values("_ct")
    cum = np.cumsum(conv[EventCols.CONVERSIONS].to_numpy())
    if n > cum[-1]:
        return events.iloc[0:0]           # never reached N conversions
    idx = int(np.searchsorted(cum, n, side="left"))
    t_n = conv["_ct"].to_numpy()[idx]
    # everything at or before the decision instant (spend accrues on the TS clock)
    return events[events[EventCols.TS] <= t_n]


def per_conversion_payouts(events: pd.DataFrame, max_sample: int = 20000,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """Reconstruct an empirical per-conversion payout sample from aggregated rows.

    Each row carries `conversions` conversions sharing `revenue`; we treat each as
    `conversions` draws of revenue/conversions. This is the distribution whose
    variance drives how uncertain the ROI estimate is at a given sample size.
    Down-samples very large campaigns for speed (unbiased random subsample).
    """
    conv = events[events[EventCols.CONVERSIONS] > 0]
    if conv.empty:
        return np.array([])
    counts = conv[EventCols.CONVERSIONS].to_numpy().astype(int)
    per = (conv[EventCols.REVENUE].to_numpy() / counts)
    payouts = np.repeat(per, counts)
    if len(payouts) > max_sample:
        rng = rng or np.random.default_rng(0)
        payouts = rng.choice(payouts, size=max_sample, replace=False)
    return payouts


def campaign_state(events: pd.DataFrame, n: int | None = None,
                   rng: np.random.Generator | None = None) -> CampaignState | None:
    """Build a CampaignState from (optionally truncated) events."""
    ev = truncate_to_n(events, n)
    conv = ev[ev[EventCols.CONVERSIONS] > 0]
    spend = float(ev[ev[EventCols.SPEND] > 0][EventCols.SPEND].sum())
    if conv.empty or spend <= 0:
        return None
    revenue = float(conv[EventCols.REVENUE].sum())
    n_conv = int(conv[EventCols.CONVERSIONS].sum())
    roi = (revenue - spend) / spend
    payouts = per_conversion_payouts(ev, rng=rng)
    cid = str(ev[EventCols.ID].iloc[0])
    return CampaignState(campaign_id=cid, n_conversions=n_conv, spend=spend,
                         revenue=revenue, roi=roi, payouts=payouts, events=ev)
