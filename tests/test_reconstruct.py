"""Deterministic reconstruction test on a hand-built campaign.

We construct spend and conversions with known times so the two-clock arithmetic
(observed vs lag-aligned ROI) can be checked by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.synthetic import EPOCH
from src.backtest.reconstruct import reconstruct_campaign


def _row(cid, ts, pb, spend, clicks, conv, rev):
    return {
        "campaign_id": cid, "timestamp": ts, "postback_time": pb,
        "spend": spend, "clicks": clicks, "conversions": conv, "revenue": rev,
        "sub_id": "s", "creative": "c", "lander": "l", "offer": "o",
        "audience": "a", "region": "r", "city": "x",
    }


def _hour(h):
    return EPOCH + pd.Timedelta(hours=h)


@pytest.fixture
def hand_campaign():
    rows = []
    # spend: $10 at each of hours 0..4 -> total spend 50
    for h in range(5):
        rows.append(_row("C1", _hour(h), pd.NaT, 10.0, 100, 0, 0.0))
    # conversions: (spend-time, postback-time, revenue)
    rows.append(_row("C1", _hour(0), _hour(2), 0.0, 0, 1, 20.0))  # A posts @2
    rows.append(_row("C1", _hour(1), _hour(4), 0.0, 0, 1, 30.0))  # B posts @4
    rows.append(_row("C1", _hour(2), _hour(6), 0.0, 0, 1, 40.0))  # C posts @6
    return pd.DataFrame(rows)


def test_observed_vs_aligned_at_n1(hand_campaign):
    states = reconstruct_campaign(hand_campaign, [1, 2])
    s1 = next(s for s in states if s.n_conversions == 1)
    # T_1 = hour 2 (A posts). spend accrued through hour2 = hours 0,1,2 = 30.
    assert s1.observed_spend == pytest.approx(30.0)
    assert s1.observed_revenue == pytest.approx(20.0)
    assert s1.observed_roi == pytest.approx((20 - 30) / 30)
    # aligned: conversions with spend-time <= hour2 = A,B,C = 90 revenue
    assert s1.aligned_revenue == pytest.approx(90.0)
    assert s1.aligned_roi == pytest.approx((90 - 30) / 30)


def test_observed_vs_aligned_at_n2(hand_campaign):
    states = reconstruct_campaign(hand_campaign, [1, 2])
    s2 = next(s for s in states if s.n_conversions == 2)
    # T_2 = hour 4 (B posts). spend accrued through hour4 = all 5 rows = 50.
    assert s2.observed_spend == pytest.approx(50.0)
    assert s2.observed_revenue == pytest.approx(50.0)
    assert s2.observed_roi == pytest.approx(0.0)
    assert s2.aligned_roi == pytest.approx((90 - 50) / 50)


def test_lag_biases_observed_below_aligned(hand_campaign):
    """The whole point: observed ROI is <= aligned ROI because revenue lags spend."""
    states = reconstruct_campaign(hand_campaign, [1, 2])
    for s in states:
        assert s.observed_roi <= s.aligned_roi + 1e-9


def test_decision_point_beyond_data_skipped(hand_campaign):
    # only 3 conversions exist; asking for N=5 must be skipped, not error
    states = reconstruct_campaign(hand_campaign, [1, 5])
    ns = {s.n_conversions for s in states}
    assert ns == {1}
