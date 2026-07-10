"""Simulate the naive rule at each decision point and score it vs ground truth.

Naive rule (the buyers' current mental model):
    observed cumulative ROI > threshold  -> "scale"  (predicts winner)
    otherwise                            -> "kill"   (predicts loser)

We score each (campaign, decision point) against the mature ground-truth verdict
and tally the confusion matrix and the two costly error types.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Cfg
from ..datasource import EventCols


def naive_call(observed_roi: float, threshold: float) -> str:
    return "scale" if observed_roi > threshold else "kill"


def simulate(states: pd.DataFrame, ground_truth: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    """Join reconstructed states to verdicts and label each call.

    Returns one row per (campaign, decision point) that has a ground-truth
    verdict, with columns: call, verdict, correct, error_type, post_profit, ...
    """
    thr = cfg.backtest.naive_scale_threshold_roi
    verdicts = ground_truth[[EventCols.ID, "verdict"]].rename(columns={EventCols.ID: "campaign_id"})
    df = states.merge(verdicts, on="campaign_id", how="inner")
    df = df[df["verdict"].notna()].copy()

    df["call"] = df["observed_roi"].apply(lambda r: naive_call(r, thr))
    df["predicts"] = df["call"].map({"scale": "winner", "kill": "loser"})
    df["correct"] = df["predicts"] == df["verdict"]

    def _etype(r):
        if r["correct"]:
            return "correct"
        # kill call on a true winner
        if r["call"] == "kill" and r["verdict"] == "winner":
            return "false_kill"
        # scale call on a true loser
        if r["call"] == "scale" and r["verdict"] == "loser":
            return "false_scale"
        return "correct"  # unreachable given binary labels

    df["error_type"] = df.apply(_etype, axis=1)
    return df


def confusion_by_point(sim: pd.DataFrame) -> pd.DataFrame:
    """Per decision point: counts, misclassification rate, and error breakdown."""
    rows = []
    for n, grp in sim.groupby("n_conversions"):
        total = len(grp)
        wrong = (~grp["correct"]).sum()
        rows.append({
            "n_conversions": int(n),
            "n_campaigns_scored": total,
            "false_kill": int((grp["error_type"] == "false_kill").sum()),
            "false_scale": int((grp["error_type"] == "false_scale").sum()),
            "misclassification_rate": wrong / total if total else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n_conversions").reset_index(drop=True)
