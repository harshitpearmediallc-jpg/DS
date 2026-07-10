"""recommend(): the SCALE / HOLD / KILL decision, with an explainable reason.

Order of checks (matches the Phase-2 spec):
  1. Sample-size gate FIRST. Below the Phase-1 trust threshold N -> HOLD, with how
     many more conversions are needed. No amount of apparent ROI overrides this.
  2. Regime check. If a genuine adverse level shift is detected (not variance),
     KILL/WARNING with the evidence — even if the campaign looks profitable overall.
  3. Significance. With enough data and no adverse regime: if the ROI interval
     clears break-even -> SCALE; if it is clearly below -> KILL; otherwise HOLD.

Every path returns a short, buyer-checkable reason and a calibrated confidence.
Trust is the product: a skeptical media buyer must be able to sanity-check the call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import Cfg, resolve_path
from ..logging_.schema import RecommendationRecord
from . import features, significance, regime as regime_mod
from .calibration import Calibrator

SCALE, HOLD, KILL = "SCALE", "HOLD", "KILL"


@dataclass
class Recommendation:
    campaign_id: str
    recommendation: str
    confidence: float | None          # calibrated P(profitable) where meaningful
    raw_confidence: float | None
    n_conversions: int
    roi: float
    ci_low: float | None
    ci_high: float | None
    conversions_needed_to_trust: int
    reason: str
    regime: dict = field(default_factory=dict)
    spend: float = 0.0
    revenue: float = 0.0


def load_trust_threshold(cfg: Cfg) -> int:
    """Prefer the exported Phase-1 threshold; fall back to config."""
    for key in ("thresholds_out",):
        p = resolve_path(cfg, cfg.io.to_dict().get(key, "outputs/phase1_thresholds.yaml"))
        if Path(p).exists():
            data = yaml.safe_load(Path(p).read_text()) or {}
            n = data.get("min_conversions_to_trust")
            if n:
                return int(n)
    return int(cfg.confidence.min_conversions_to_trust)


def recommend(events_campaign: pd.DataFrame, cfg: Cfg, trust_n: int | None = None,
              n_cap: int | None = None, calibrator: Calibrator | None = None,
              rng: np.random.Generator | None = None) -> Recommendation:
    """Produce a SCALE/HOLD/KILL recommendation for ONE campaign.

    n_cap: evaluate at the state after this many conversions (for backtest /
    calibration). None => use the full observed history ("decide now").
    """
    rng = rng or np.random.default_rng(cfg.random_seed)
    if trust_n is None:
        trust_n = load_trust_threshold(cfg)
    alpha = float(cfg.confidence.significance_level)
    B = int(cfg.confidence.bootstrap_iterations)
    reg_cfg = cfg.confidence.regime

    st = features.campaign_state(events_campaign, n=n_cap, rng=rng)
    if st is None:
        return Recommendation("unknown", HOLD, None, None, 0, 0.0, None, None,
                              trust_n, "Held: no spend/conversions observed yet.")

    n, roi = st.n_conversions, st.roi

    # ---- 1. sample-size gate ------------------------------------------------
    if n < trust_n:
        needed = trust_n - n
        return Recommendation(
            st.campaign_id, HOLD, None, None, n, roi, None, None, needed,
            f"Held: only {n} conversions. ROI {roi:+.0%} is within expected noise at "
            f"this sample size — need ~{needed} more (≈{trust_n} total) before a "
            f"scale/kill call is trustworthy.", spend=st.spend, revenue=st.revenue)

    # significance interval (used by both the regime and the final call)
    sig = significance.bootstrap_roi(st.payouts, st.spend, n, iterations=B,
                                     alpha=alpha, rng=rng)
    raw_conf = sig.p_profitable
    conf = float(calibrator.predict(raw_conf)) if calibrator is not None else raw_conf

    # ---- 2. regime check ----------------------------------------------------
    reg = regime_mod.detect_regime(
        st.events, recent_window=int(reg_cfg.recent_window_conversions),
        cusum_k_sigma=float(reg_cfg.cusum_threshold_sigma), rng=rng)
    regd = dict(detected=reg.detected, direction=reg.direction,
                recent_roi=reg.recent_roi, prior_roi=reg.prior_roi,
                degraded_dimensions=reg.degraded_dimensions, evidence=reg.evidence)

    if reg.detected and reg.direction == "down":
        return Recommendation(
            st.campaign_id, KILL, conf, raw_conf, n, roi, sig.ci_low, sig.ci_high,
            0,
            f"Kill / warning: genuine adverse change, not variance — {reg.evidence}. "
            f"Overall ROI {roi:+.0%} but the recent trend has turned.", regd,
            spend=st.spend, revenue=st.revenue)

    # ---- 3. significance-driven call ---------------------------------------
    ci = f"95% CI {sig.ci_low:+.0%}..{sig.ci_high:+.0%}"
    if sig.direction == "profitable":
        rec, reason = SCALE, (
            f"Scale: {n} conversions, ROI {roi:+.0%}, {conf:.0%} likely profitable "
            f"({ci} excludes break-even). Signal is real and holding.")
    elif sig.direction == "unprofitable":
        rec, reason = KILL, (
            f"Kill: {n} conversions, ROI {roi:+.0%}, only {conf:.0%} likely profitable "
            f"({ci} is entirely below break-even). The loss is real, not noise.")
    else:
        rec, reason = HOLD, (
            f"Hold: {n} conversions but ROI {roi:+.0%} is not distinguishable from "
            f"break-even ({ci} straddles zero). Let it run before acting.")

    return Recommendation(st.campaign_id, rec, conf, raw_conf, n, roi,
                          sig.ci_low, sig.ci_high, 0, reason, regd,
                          spend=st.spend, revenue=st.revenue)


def to_log_record(rec: Recommendation, decided_at: str,
                  dimension_snapshot: dict | None = None) -> RecommendationRecord:
    """Build the append-only log record (outcome backfilled later)."""
    return RecommendationRecord(
        campaign_id=rec.campaign_id, decided_at=decided_at,
        conversions_so_far=rec.n_conversions, spend_so_far=rec.spend,
        revenue_so_far=rec.revenue, observed_roi=rec.roi,
        recommendation=rec.recommendation, confidence=rec.confidence,
        roi_ci_low=rec.ci_low, roi_ci_high=rec.ci_high,
        conversions_needed_to_trust=(rec.conversions_needed_to_trust or None),
        reason=rec.reason, dimension_snapshot=dimension_snapshot or {},
    )
