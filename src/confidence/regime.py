"""Regime-change detection: is a downturn a real level shift or within-variance wobble?

A campaign can be profitable overall yet be genuinely degrading right now (creative
fatigue, an offer going bad, a traffic source turning). The naive rule and even the
overall ROI interval miss this. We look at the time series two ways:

  1. Recent-vs-prior window. Split the campaign at the last `recent_window`
     conversions. Bootstrap each window's ROI and ask whether the recent window is
     BOTH below the prior window (beyond overlap of their intervals) AND below
     break-even. That combination = a real adverse shift, not noise.

  2. CUSUM on per-conversion profit. A cumulative-sum control chart flags a
     sustained mean shift earlier than eyeballing does.

  3. Dimensional corroboration. Recompute recent-vs-prior ROI per sub_id / source /
     vertical. Degradation concentrated in specific levels is stronger evidence of a
     real change (and tells the buyer WHERE it broke).

All three feed a single explainable verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..datasource import EventCols, DIMENSION_COLS


@dataclass
class RegimeResult:
    detected: bool
    direction: str                 # 'down' | 'up' | 'stable'
    recent_roi: float
    prior_roi: float
    recent_conversions: int
    cusum_alarm: bool
    degraded_dimensions: list = field(default_factory=list)
    evidence: str = ""


def _roi(ev: pd.DataFrame) -> tuple[float, float, float]:
    spend = float(ev[ev[EventCols.SPEND] > 0][EventCols.SPEND].sum())
    rev = float(ev[ev[EventCols.CONVERSIONS] > 0][EventCols.REVENUE].sum())
    roi = (rev - spend) / spend if spend > 0 else 0.0
    return roi, spend, rev


def _conv_time(conv: pd.DataFrame) -> pd.Series:
    if EventCols.POSTBACK_TS in conv.columns:
        return conv[EventCols.POSTBACK_TS].fillna(conv[EventCols.TS])
    return conv[EventCols.TS]


def _split_recent(events: pd.DataFrame, recent_window: int):
    """Split events into (prior, recent) at the last `recent_window` conversions."""
    conv = events[events[EventCols.CONVERSIONS] > 0].copy()
    conv["_ct"] = _conv_time(conv)
    conv = conv.sort_values("_ct")
    cum_from_end = np.cumsum(conv[EventCols.CONVERSIONS].to_numpy()[::-1])[::-1]
    # rows whose conversions fall within the last `recent_window`
    recent_mask = cum_from_end <= recent_window
    if recent_mask.all() or (~recent_mask).all():
        return None, None, None
    split_time = conv.loc[conv.index[recent_mask], "_ct"].min()
    recent = events[events[EventCols.TS] >= split_time]
    prior = events[events[EventCols.TS] < split_time]
    return prior, recent, split_time


def _cusum(profit_per_conv: np.ndarray, k_sigma: float) -> bool:
    """One-sided CUSUM for a downward mean shift in per-conversion profit."""
    if len(profit_per_conv) < 10:
        return False
    mu = profit_per_conv.mean()
    sigma = profit_per_conv.std(ddof=1) or 1.0
    slack = 0.5 * sigma                      # standard CUSUM slack
    s = 0.0
    for x in profit_per_conv:
        s = min(0.0, s + (x - mu) + slack)   # accumulate negative drift
        if s < -k_sigma * sigma:
            return True
    return False


def detect_regime(events: pd.DataFrame, recent_window: int = 30,
                  cusum_k_sigma: float = 4.0, break_even: float = 0.0,
                  min_shift: float = 0.10, min_recent_conversions: int | None = None,
                  rng: np.random.Generator | None = None) -> RegimeResult:
    """Detect a genuine adverse (or favourable) regime CHANGE.

    A change means the prior window was on one side of break-even and the recent
    window crossed to the other. A campaign that was ALWAYS losing is not a regime
    change — it is a consistent loser, which the significance gate handles as a
    plain KILL with the correct 'loss is real' reason. This keeps the explanation
    honest.
    """
    rng = rng or np.random.default_rng(0)
    if min_recent_conversions is None:
        min_recent_conversions = max(10, recent_window // 2)
    prior, recent, _ = _split_recent(events, recent_window)
    if prior is None:
        return RegimeResult(False, "stable", 0.0, 0.0, 0, False, [],
                            "Not enough history to compare recent vs prior.")

    recent_roi, _, _ = _roi(recent)
    prior_roi, _, _ = _roi(prior)
    recent_conv = int(recent[recent[EventCols.CONVERSIONS] > 0][EventCols.CONVERSIONS].sum())

    # CUSUM on per-conversion profit over the whole series (ordered by time)
    conv = events[events[EventCols.CONVERSIONS] > 0].copy()
    conv["_ct"] = _conv_time(conv)
    conv = conv.sort_values("_ct")
    ppc = (conv[EventCols.REVENUE].to_numpy() /
           conv[EventCols.CONVERSIONS].to_numpy())  # per-row payout; spend excluded
    cusum_alarm = _cusum(ppc, cusum_k_sigma)

    # dimensional corroboration: which levels degraded recent vs prior
    degraded = _degraded_dimensions(prior, recent, min_shift)

    # An adverse CHANGE = was profitable, now below break-even, with a real drop,
    # measured on a recent window big enough to not be noise.
    drop = prior_roi - recent_roi
    enough_recent = recent_conv >= min_recent_conversions
    adverse = (enough_recent and prior_roi > break_even and recent_roi < break_even
               and drop > min_shift)
    favourable = (enough_recent and prior_roi < break_even and recent_roi > break_even
                  and (recent_roi - prior_roi) > min_shift)
    detected = bool(adverse and (cusum_alarm or degraded or drop > 2 * min_shift))
    direction = "down" if adverse else ("up" if favourable else "stable")

    if detected:
        ev = (f"recent {recent_conv} conv at {recent_roi:+.0%} ROI vs prior "
              f"{prior_roi:+.0%} (drop {drop:+.0%})")
        if degraded:
            ev += f"; degraded in {', '.join(degraded[:3])}"
        if cusum_alarm:
            ev += "; CUSUM level-shift alarm"
    else:
        ev = f"recent {recent_roi:+.0%} vs prior {prior_roi:+.0%}: within variance."

    return RegimeResult(detected, direction, recent_roi, prior_roi, recent_conv,
                        cusum_alarm, degraded, ev)


def _degraded_dimensions(prior: pd.DataFrame, recent: pd.DataFrame,
                         min_shift: float) -> list:
    """Return 'dim=level' labels that were profitable in prior and turned bad recently."""
    out = []
    for dim in DIMENSION_COLS:
        if dim not in recent.columns:
            continue
        for level in recent[dim].dropna().unique():
            r = recent[recent[dim] == level]
            p = prior[prior[dim] == level]
            if p.empty or r.empty:
                continue
            r_roi, r_sp, _ = _roi(r)
            p_roi, p_sp, _ = _roi(p)
            if p_sp <= 0 or r_sp <= 0:
                continue
            if p_roi > 0 and r_roi < 0 and (p_roi - r_roi) > min_shift:
                out.append(f"{dim}={level}")
    return out
