"""Data-quality guard: flag campaigns whose spend looks under-tracked.

Extreme ROIs in the backtest trace back to a small set of campaigns where spend
is zero or implausibly low relative to clicks (implied CPC below a floor). These
inflate ROI and, if killed early, inflate the false-kill dollar figure. The tool
should surface them rather than silently trust or silently drop them.

By default we KEEP them (the headline is robust to their exclusion) but always
report them. Set `data_quality.exclude_untracked_from_headline: true` in config
to score the backtest without them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Cfg
from ..datasource import EventCols


def flag_untracked(events: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    """Return per-campaign totals with an `untracked_spend` flag + reason."""
    dq = cfg.get("data_quality")
    min_cpc = dq.get("flag_min_cpc", 0.05) if dq else 0.05
    # only treat low CPC as "under-tracked" when it ALSO produces an implausible
    # ROI — otherwise we'd wrongly flag legitimately cheap, high-volume traffic.
    min_roi = dq.get("flag_min_roi_for_lowcpc", 3.0) if dq else 3.0

    g = events.groupby(EventCols.ID).agg(
        total_spend=(EventCols.SPEND, "sum"),
        total_revenue=(EventCols.REVENUE, "sum"),
        total_conversions=(EventCols.CONVERSIONS, "sum"),
        total_clicks=(EventCols.CLICKS, "sum") if EventCols.CLICKS in events.columns
        else (EventCols.SPEND, "size"),
    ).reset_index()

    g["implied_cpc"] = np.where(g["total_clicks"] > 0,
                                g["total_spend"] / g["total_clicks"], np.nan)
    g["roi"] = np.where(g["total_spend"] > 0,
                        (g["total_revenue"] - g["total_spend"]) / g["total_spend"], np.nan)

    zero_spend = g["total_spend"] <= 0
    low_cpc_high_roi = (g["implied_cpc"] < min_cpc) & (g["roi"] > min_roi)
    g["untracked_spend"] = zero_spend | low_cpc_high_roi
    g["reason"] = np.where(zero_spend, "zero_or_negative_spend",
                           np.where(low_cpc_high_roi,
                                    f"cpc<{min_cpc}_and_roi>{min_roi}", ""))
    return g


def untracked_ids(events: pd.DataFrame, cfg: Cfg) -> set:
    g = flag_untracked(events, cfg)
    return set(g.loc[g["untracked_spend"], EventCols.ID])


def quality_summary(events: pd.DataFrame, cfg: Cfg) -> dict:
    g = flag_untracked(events, cfg)
    flagged = g[g["untracked_spend"]]
    return {
        "n_campaigns": int(len(g)),
        "n_untracked": int(len(flagged)),
        "n_zero_spend": int((g["total_spend"] <= 0).sum()),
        "revenue_on_untracked": float(flagged["total_revenue"].sum()),
        "flagged_ids_sample": flagged.sort_values("roi", ascending=False)
        [EventCols.ID].head(20).tolist(),
    }
