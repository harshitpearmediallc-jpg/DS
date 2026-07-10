"""The backtest must recover the KNOWN structure from synthetic ground truth:
error decays with sample size, both error types appear, and costs are lower bounds.
"""

from __future__ import annotations

import pandas as pd

from src.backtest import cohorts, reconstruct, simulate, cost, lag


def _run(cfg, events):
    points = list(cfg.backtest.decision_conversion_points)
    gt = cohorts.assign_ground_truth(events, cfg)
    states = reconstruct.reconstruct_all(events, points)
    sim = simulate.simulate(states, gt, cfg)
    conf = simulate.confusion_by_point(sim)
    costs = cost.cost_by_point(sim)
    return gt, states, sim, conf, costs


def test_error_decays_with_sample_size(cfg, small_dataset):
    _, events, _ = small_dataset
    _, _, _, conf, _ = _run(cfg, events)
    conf = conf.sort_values("n_conversions")
    err = conf["misclassification_rate"].to_numpy()
    # error at the smallest point must be much higher than at the largest
    assert err[0] > err[-1] + 0.15
    # broadly monotone decreasing (allow tiny wobble)
    diffs = err[1:] - err[:-1]
    assert (diffs <= 0.05).all()


def test_both_error_types_present_early(cfg, small_dataset):
    _, events, _ = small_dataset
    _, _, sim, _, _ = _run(cfg, events)
    early = sim[sim["n_conversions"] == sim["n_conversions"].min()]
    types = set(early["error_type"])
    assert "false_kill" in types
    assert "false_scale" in types  # small-sample noise makes losers look like winners too


def test_false_kill_dominates_cost(cfg, small_dataset):
    """Because winners run long and losers get killed, forgone winner profit is
    the larger money leak — and it is explicitly a lower bound."""
    _, events, _ = small_dataset
    _, _, _, _, costs = _run(cfg, events)
    typ = costs[costs["n_conversions"] == 10].iloc[0]
    assert typ["false_kill_cost_lower_bound"] > 0
    assert typ["total_cost_lower_bound"] >= typ["false_kill_cost_lower_bound"]


def test_recovered_error_matches_ground_truth_labels(cfg, small_dataset):
    """A sanity cross-check: campaigns the backtest marks false_kill really are
    ground-truth winners that the naive rule killed."""
    _, events, _ = small_dataset
    gt, _, sim, _, _ = _run(cfg, events)
    fk = sim[sim["error_type"] == "false_kill"]
    assert (fk["verdict"] == "winner").all()
    assert (fk["call"] == "kill").all()
    fs = sim[sim["error_type"] == "false_scale"]
    assert (fs["verdict"] == "loser").all()
    assert (fs["call"] == "scale").all()


def test_lag_decomposition_reports_a_share(cfg, small_dataset):
    _, events, _ = small_dataset
    gt = cohorts.assign_ground_truth(events, cfg)
    states = reconstruct.reconstruct_all(events, list(cfg.backtest.decision_conversion_points))
    decomp = lag.decompose(states, gt, decision_point=10)
    summ = lag.decomposition_summary(decomp)
    assert summ["n_campaigns"] > 0
    # shares are well-formed and sum to ~1
    assert 0.0 <= summ["lag_share"] <= 1.0
    assert abs(summ["lag_share"] + summ["genuine_share"] - 1.0) < 1e-6


def test_survivor_cohort_are_winners_with_bad_early_stretch(cfg, small_dataset):
    _, events, _ = small_dataset
    gt = cohorts.assign_ground_truth(events, cfg)
    points = list(cfg.backtest.decision_conversion_points)
    states = reconstruct.reconstruct_all(events, points)
    cohort = cohorts.survivor_cohort(states, gt, early_point=min(points))
    assert len(cohort) > 0
    assert (cohort["verdict"] == "winner").all()
    assert (cohort["early_observed_roi"] <= 0).all()
