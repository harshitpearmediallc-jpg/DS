"""Phase 1 orchestrator: run the whole backtest and export the v1 threshold.

Pipeline:
  1. load events + assign mature ground-truth verdicts
  2. reconstruct each campaign's observed/aligned state at every decision point
  3. simulate the naive rule and score it
  4. tally misclassification + dollar cost by decision point
  5. read off N* = the smallest decision point whose error <= tolerable rate
  6. price the headline at the point buyers actually act (typical_decision)
  7. decompose early-loss reversal into lag vs genuine recovery
  8. find the survivor cohort (early loss, kept running, true winners)
  9. export thresholds/config for Phase 2

All headline numbers are framed defensively (lower bounds) because of
survivorship censoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import Cfg, resolve_path
from ..datasource import DataSource, EventCols
from . import cohorts, reconstruct, simulate, cost, lag, quality


@dataclass
class Phase1Result:
    ground_truth: pd.DataFrame
    states: pd.DataFrame
    sim: pd.DataFrame
    confusion: pd.DataFrame          # misclassification by decision point
    costs: pd.DataFrame              # dollar cost by decision point
    trust_threshold_n: int | None    # N* read off the error curve
    typical_point: int
    headline: dict
    lag_summary: dict
    survivor_cohort: pd.DataFrame
    meta: dict = field(default_factory=dict)


def run_phase1(ds: DataSource, cfg: Cfg) -> Phase1Result:
    points = list(cfg.backtest.decision_conversion_points)
    typical = int(cfg.backtest.typical_decision_conversions)

    events = ds.load_events()

    # 0. data-quality guard: flag (and optionally exclude) under-tracked spend
    dq = quality.quality_summary(events, cfg)
    dq_cfg = cfg.get("data_quality")
    if dq_cfg and dq_cfg.get("exclude_untracked_from_headline", False):
        bad = quality.untracked_ids(events, cfg)
        events = events[~events[EventCols.ID].isin(bad)]

    # 1. ground truth
    gt = cohorts.assign_ground_truth(events, cfg)

    # 2. reconstruct decision states
    states = reconstruct.reconstruct_all(events, points)

    # merge post_profit etc. is already on states; simulate needs verdicts
    # 3. simulate naive rule (only scored on campaigns with a verdict)
    sim = simulate.simulate(states, gt, cfg)

    # 4. tallies
    confusion = simulate.confusion_by_point(sim)
    costs = cost.cost_by_point(sim)
    curve = confusion.merge(costs, on="n_conversions", how="outer").sort_values("n_conversions")

    # 5. read off trust threshold N*
    tol = cfg.backtest.tolerable_error_rate
    ok = curve[curve["misclassification_rate"] <= tol]
    trust_n = int(ok["n_conversions"].min()) if not ok.empty else None

    # 6. headline at the typical decision point
    typ_row = curve[curve["n_conversions"] == typical]
    headline = {}
    if not typ_row.empty:
        r = typ_row.iloc[0]
        headline = {
            "typical_decision_conversions": typical,
            "misclassification_rate": float(r["misclassification_rate"]),
            "n_campaigns_scored": int(r["n_campaigns_scored"]),
            "false_kill_cost_lower_bound": float(r["false_kill_cost_lower_bound"]),
            "false_scale_cost": float(r["false_scale_cost"]),
            "total_cost_lower_bound": float(r["total_cost_lower_bound"]),
            "n_false_kill": int(r["n_false_kill"]),
            "n_false_scale": int(r["n_false_scale"]),
        }

    # 7. lag vs genuine recovery decomposition (at the typical point)
    decomp = lag.decompose(states, gt, decision_point=typical)
    lag_summary = lag.decomposition_summary(decomp)

    # 8. survivor cohort (early loss, kept running, true winners)
    survivors = cohorts.survivor_cohort(states, gt, early_point=min(points))

    meta = {
        "n_campaigns_total": int(gt.shape[0]),
        "n_mature": int(gt["mature"].sum()),
        "n_with_verdict": int(gt["verdict"].notna().sum()),
        "n_winners": int((gt["verdict"] == "winner").sum()),
        "n_losers": int((gt["verdict"] == "loser").sum()),
        "n_borderline_excluded": int((gt["mature"] & gt["borderline"]).sum()),
        "decision_points": points,
        "tolerable_error_rate": tol,
        "data_quality": dq,
    }

    return Phase1Result(
        ground_truth=gt, states=states, sim=sim, confusion=confusion, costs=costs,
        trust_threshold_n=trust_n, typical_point=typical, headline=headline,
        lag_summary=lag_summary, survivor_cohort=survivors, meta=meta,
    )


def export_thresholds(result: Phase1Result, cfg: Cfg) -> Path:
    """Write the learned threshold + summary for Phase 2 to consume."""
    out = resolve_path(cfg, cfg.io.thresholds_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "min_conversions_to_trust": result.trust_threshold_n,
        "typical_decision_conversions": result.typical_point,
        "tolerable_error_rate": cfg.backtest.tolerable_error_rate,
        "headline_lower_bound": result.headline,
        "lag_summary": result.lag_summary,
        "backtest_meta": result.meta,
        "note": ("false-kill costs are LOWER BOUNDS due to survivorship censoring; "
                 "killed winners are unobserved."),
    }
    with open(out, "w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return out
