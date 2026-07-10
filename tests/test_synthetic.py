"""The synthetic harness must produce data whose TRUE verdict we can recover."""

from __future__ import annotations

import numpy as np

from src.datasource import EventCols


def test_both_labels_present(small_dataset):
    _, _, truth = small_dataset
    assert set(truth["true_label"]) == {"winner", "loser"}
    # all five archetypes should appear
    assert truth["archetype"].nunique() == 5


def test_mature_verdict_matches_truth(small_dataset):
    """On mature campaigns, realised ROI sign must match the injected true label
    the vast majority of the time (marginal archetype near zero is the exception)."""
    _, _, truth = small_dataset
    mature = truth[truth["total_conversions"] >= 100]
    assert len(mature) > 50
    agree = ((mature["realised_roi"] > 0) == (mature["true_label"] == "winner")).mean()
    assert agree > 0.90


def test_realised_tracks_true_roi(small_dataset):
    _, _, truth = small_dataset
    # per-archetype mean realised ROI should be within a small band of the target
    for arch, grp in truth.groupby("archetype"):
        if arch == "marginal":
            continue
        assert abs(grp["true_roi"].mean() - grp["realised_roi"].mean()) < 0.06


def test_conversion_lag_present(small_dataset):
    """Postbacks must land at or after the spend that generated them."""
    _, events, _ = small_dataset
    conv = events[events[EventCols.CONVERSIONS] > 0]
    lag_hours = (conv[EventCols.POSTBACK_TS] - conv[EventCols.TS]).dt.total_seconds() / 3600
    assert (lag_hours >= -1e-9).all()          # never negative
    assert lag_hours.mean() > 0.5              # real, non-trivial lag


def test_early_stretch_injected(small_dataset):
    """unlucky_winner campaigns must actually look bad early despite being winners."""
    campaigns, events, truth = small_dataset
    unlucky = truth[truth["archetype"] == "unlucky_winner"]["campaign_id"]
    assert len(unlucky) > 0
    # first few conversions (by spend-time) should be far below the campaign's mean payout
    for cid in unlucky.head(3):
        conv = events[(events[EventCols.ID] == cid) & (events[EventCols.CONVERSIONS] > 0)]
        conv = conv.sort_values(EventCols.TS)
        early_mean = conv[EventCols.REVENUE].head(10).mean()
        overall_mean = conv[EventCols.REVENUE].mean()
        assert early_mean < overall_mean
