"""Ground-truth verdicts, maturity gate, borderline exclusion, survivor cohort.

A campaign earns a trustworthy verdict only if it is MATURE (enough conversions
and/or spend that per-conversion payout variance has washed out). The verdict is
the sign of lifetime profit from ACTUAL payouts. Borderline campaigns (finishing
within a small band around break-even) are genuinely too-close-to-call and are
excluded from headline accuracy on the first pass.
"""

from __future__ import annotations

import pandas as pd

from ..config import Cfg
from ..datasource import EventCols, CampaignCols


def campaign_totals(events: pd.DataFrame) -> pd.DataFrame:
    """Per-campaign lifetime totals from actual payouts."""
    g = events.groupby(EventCols.ID)
    tot = g.agg(
        total_spend=(EventCols.SPEND, "sum"),
        total_revenue=(EventCols.REVENUE, "sum"),
        total_conversions=(EventCols.CONVERSIONS, "sum"),
    ).reset_index()
    tot["profit"] = tot["total_revenue"] - tot["total_spend"]
    tot["roi"] = tot.apply(
        lambda r: r["profit"] / r["total_spend"] if r["total_spend"] > 0 else 0.0, axis=1
    )
    return tot


def assign_ground_truth(events: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    """Return per-campaign totals plus `mature`, `verdict`, `borderline` flags.

    verdict in {'winner','loser'} for mature+non-borderline campaigns, else None.
    """
    gt = cfg.ground_truth
    tot = campaign_totals(events)

    conv_gate = tot["total_conversions"] >= gt.maturity_min_conversions
    if gt.maturity_min_spend is not None:
        spend_gate = tot["total_spend"] >= gt.maturity_min_spend
        tot["mature"] = (conv_gate | spend_gate) if gt.maturity_logic == "or" else (conv_gate & spend_gate)
    else:
        tot["mature"] = conv_gate

    band = gt.borderline_roi_band
    tot["borderline"] = tot["roi"].abs() <= band

    def _verdict(r):
        if not r["mature"] or r["borderline"]:
            return None
        return "winner" if r["profit"] > 0 else "loser"

    tot["verdict"] = tot.apply(_verdict, axis=1)
    return tot


def survivor_cohort(states: pd.DataFrame, ground_truth: pd.DataFrame,
                    early_loss_roi: float = -0.15, early_point: int | None = None) -> pd.DataFrame:
    """Campaigns that looked like a LOSS early but were kept running and turned
    out to be ground-truth WINNERS.

    This is the direct, self-contained proof that early-loss != doomed, and the
    reason our false-kill cost is only a lower bound (the buyers who *did* kill
    such campaigns are invisible to us).

    early_point: which decision point to judge "early" on; defaults to the
    smallest available.
    """
    winners = set(ground_truth.loc[ground_truth["verdict"] == "winner", EventCols.ID])
    df = states[states["campaign_id"].isin(winners)].copy()
    if df.empty:
        return df
    if early_point is None:
        early_point = int(df["n_conversions"].min())
    early = df[df["n_conversions"] == early_point]
    cohort_ids = early.loc[early["observed_roi"] <= early_loss_roi, "campaign_id"]
    cohort = ground_truth[ground_truth[EventCols.ID].isin(cohort_ids)].copy()
    # attach the early observed roi for storytelling
    early_map = early.set_index("campaign_id")["observed_roi"]
    cohort["early_observed_roi"] = cohort[EventCols.ID].map(early_map)
    cohort["early_point"] = early_point
    return cohort.sort_values("early_observed_roi")
