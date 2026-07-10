"""Lag vs. genuine-recovery decomposition.

Guard against the trap: a campaign that looks like an early LOSS and later shows
PROFIT may simply be delayed conversions posting (lag), not a real turnaround.
If we credit the naive rule with "catching recoveries" we would overstate the
value of waiting. So we decompose, for each early-loss-but-eventually-winner
campaign, the journey from the observed early ROI to the final ROI into:

    lag_component     = aligned_roi(N) - observed_roi(N)
        revenue already EARNED by spend<=T_N but not yet POSTED. Pure lag.

    genuine_component = final_roi - aligned_roi(N)
        improvement BEYOND what the pending-conversion pipeline explains. Real
        recovery (or real further decline).

Reported as an aggregate share so the buyer knows how much of "loss turns to
profit" is an artefact of postback timing.
"""

from __future__ import annotations

import pandas as pd

from ..datasource import EventCols


def decompose(states: pd.DataFrame, ground_truth: pd.DataFrame,
              decision_point: int, early_loss_roi: float = 0.0) -> pd.DataFrame:
    """For winners that showed an early loss at `decision_point`, split the gap
    to final ROI into lag vs genuine components.
    """
    winners = ground_truth[ground_truth["verdict"] == "winner"][[EventCols.ID, "roi"]]
    winners = winners.rename(columns={EventCols.ID: "campaign_id", "roi": "final_roi"})

    st = states[states["n_conversions"] == decision_point].merge(
        winners, on="campaign_id", how="inner")
    # only campaigns that actually looked like a loss early
    st = st[st["observed_roi"] <= early_loss_roi].copy()
    if st.empty:
        return st

    st["lag_component"] = st["aligned_roi"] - st["observed_roi"]
    st["genuine_component"] = st["final_roi"] - st["aligned_roi"]
    st["total_gap"] = st["final_roi"] - st["observed_roi"]
    return st


def decomposition_summary(decomp: pd.DataFrame) -> dict:
    """Aggregate share of early-loss reversal attributable to lag vs genuine."""
    if decomp.empty:
        return {"n_campaigns": 0, "lag_share": None, "genuine_share": None,
                "mean_lag_component": None, "mean_genuine_component": None}
    lag = decomp["lag_component"].clip(lower=0).sum()
    genuine = decomp["genuine_component"].clip(lower=0).sum()
    denom = lag + genuine
    return {
        "n_campaigns": int(len(decomp)),
        "lag_share": float(lag / denom) if denom else None,
        "genuine_share": float(genuine / denom) if denom else None,
        "mean_lag_component": float(decomp["lag_component"].mean()),
        "mean_genuine_component": float(decomp["genuine_component"].mean()),
    }
