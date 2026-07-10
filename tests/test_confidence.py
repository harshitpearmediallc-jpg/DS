"""Phase 2 tests: the SCALE/HOLD/KILL layer must flag noise as HOLD, catch real
degradation as KILL, confirm clear winners as SCALE, and calibrate monotonically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.synthetic import generate, GenConfig, EPOCH
from src.datasource import EventCols
from src.confidence import (
    recommend, SCALE, HOLD, KILL, detect_regime, fit_calibrator, reliability,
    bootstrap_roi,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def synth():
    return generate(GenConfig(n_campaigns=120, seed=7))


def _row(cid, h, spend, conv, rev):
    return {
        "campaign_id": cid, "timestamp": EPOCH + pd.Timedelta(hours=h),
        "spend": spend, "clicks": 10, "conversions": conv, "revenue": rev,
        "sub_id": "s", "creative": "c", "lander": "l", "offer": "o",
        "audience": "a", "region": "r", "city": "x",
    }


# ---- sample-size gate -------------------------------------------------------

def test_gate_holds_below_threshold(cfg, synth):
    _, events, truth = synth
    cid = truth[truth.archetype == "clear_winner"].campaign_id.iloc[0]
    ev = events[events[EventCols.ID] == cid]
    r = recommend(ev, cfg, trust_n=100, n_cap=5)  # far below threshold
    assert r.recommendation == HOLD
    assert r.conversions_needed_to_trust > 0
    assert "need" in r.reason.lower()


# ---- clear winner / loser at maturity --------------------------------------

def test_clear_winner_scales(cfg, synth):
    _, events, truth = synth
    cid = truth[truth.archetype == "clear_winner"].campaign_id.iloc[0]
    ev = events[events[EventCols.ID] == cid]
    r = recommend(ev, cfg, trust_n=30, n_cap=300)
    assert r.recommendation == SCALE
    assert r.ci_low > 0                      # interval clears break-even
    assert r.confidence >= 0.9


def test_clear_loser_kills_with_honest_reason(cfg, synth):
    _, events, truth = synth
    cid = truth[truth.archetype == "clear_loser"].campaign_id.iloc[0]
    ev = events[events[EventCols.ID] == cid]
    r = recommend(ev, cfg, trust_n=30, n_cap=300)
    assert r.recommendation == KILL
    assert r.ci_high < 0
    # a consistent loser is NOT reported as a regime change
    assert "loss is real" in r.reason.lower()


# ---- noise -> HOLD ----------------------------------------------------------

def test_noise_holds_when_interval_straddles_break_even(cfg):
    # a campaign right at break-even with few, noisy payouts: CI should straddle 0
    rng = np.random.default_rng(1)
    rows = []
    for h in range(40):
        # payout ~ mean 1.0 (break-even vs spend 1.0) with high variance
        rev = float(max(0.0, rng.normal(1.0, 1.5)))
        rows.append(_row("N1", h, 1.0, 1, rev))
    ev = pd.DataFrame(rows)
    r = recommend(ev, cfg, trust_n=30, n_cap=None)
    assert r.recommendation == HOLD
    assert r.ci_low < 0 < r.ci_high


# ---- regime change: profitable then degrades -> KILL -----------------------

def _degrading_campaign():
    rows = []
    h = 0
    for _ in range(100):                     # prior: strong winner ($3 on $1)
        rows.append(_row("D1", h, 1.0, 1, 3.0)); h += 1
    for _ in range(40):                      # recent: collapses ($0.2 on $1)
        rows.append(_row("D1", h, 1.0, 1, 0.2)); h += 1
    return pd.DataFrame(rows)


def test_regime_detects_real_degradation(cfg):
    ev = _degrading_campaign()
    reg = detect_regime(ev, recent_window=30, cusum_k_sigma=4.0)
    assert reg.detected
    assert reg.direction == "down"
    assert reg.recent_roi < 0 < reg.prior_roi


def test_regime_downturn_yields_kill(cfg):
    ev = _degrading_campaign()
    r = recommend(ev, cfg, trust_n=30, n_cap=None)
    assert r.recommendation == KILL
    assert "adverse change" in r.reason.lower() or "trend" in r.reason.lower()


def test_stable_winner_not_flagged_as_regime(cfg):
    rows = [_row("W1", h, 1.0, 1, 2.0) for h in range(140)]  # steady winner
    reg = detect_regime(pd.DataFrame(rows), recent_window=30)
    assert not reg.detected


# ---- calibration ------------------------------------------------------------

def test_isotonic_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 500)
    # true prob increases with raw; draw noisy binary outcomes
    outcomes = (rng.uniform(0, 1, 500) < raw).astype(int)
    cal = fit_calibrator(raw, outcomes)
    grid = np.linspace(0, 1, 50)
    preds = np.array([cal.predict(g) for g in grid])
    assert np.all(np.diff(preds) >= -1e-9)          # nondecreasing
    assert preds.min() >= 0 and preds.max() <= 1


def test_calibration_reduces_ece_when_raw_is_miscalibrated():
    rng = np.random.default_rng(2)
    n = 2000
    true_p = rng.uniform(0, 1, n)
    outcomes = (rng.uniform(0, 1, n) < true_p).astype(int)
    raw = np.clip(true_p ** 2, 0, 1)                 # overconfident-low, miscalibrated
    _, ece_raw = reliability(raw, outcomes)
    cal = fit_calibrator(raw, outcomes)
    cal_conf = np.array([cal.predict(r) for r in raw])
    _, ece_cal = reliability(cal_conf, outcomes)
    assert ece_cal <= ece_raw + 1e-9
    assert ece_cal < 0.1


def test_bootstrap_interval_tightens_with_more_data():
    rng = np.random.default_rng(3)
    payouts_small = rng.lognormal(0.3, 0.8, 15)
    payouts_big = rng.lognormal(0.3, 0.8, 600)
    s_small = bootstrap_roi(payouts_small, spend=len(payouts_small), n_conversions=15, rng=rng)
    s_big = bootstrap_roi(payouts_big, spend=len(payouts_big), n_conversions=600, rng=rng)
    assert (s_big.ci_high - s_big.ci_low) < (s_small.ci_high - s_small.ci_low)
