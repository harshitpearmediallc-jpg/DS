"""Significance / sample-size gate: is observed ROI distinguishable from break-even?

We hold the OBSERVED spend fixed (it is known) and treat the uncertainty as coming
from the per-conversion payout distribution at the current sample size. A bootstrap
over the empirical payouts gives:
  * a confidence interval on ROI, and
  * P(profitable) = P(ROI > 0),
both of which shrink toward certainty as conversions accumulate. This is exactly
the "is the swing real or just small-sample noise?" question, answered with a
proper interval rather than a point estimate.

Why bootstrap (not a t-interval): affiliate payouts are heavy-tailed and often
zero-inflated, so a normal/t interval understates uncertainty at small n. The
bootstrap makes no distributional assumption. For large n it agrees with the CLT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SignificanceResult:
    n_conversions: int
    spend: float
    roi_point: float
    ci_low: float
    ci_high: float
    p_profitable: float          # bootstrap P(ROI > break_even)
    distinguishable: bool        # CI excludes break-even (either side)
    direction: str               # 'profitable' | 'unprofitable' | 'inconclusive'


def bootstrap_roi(payouts: np.ndarray, spend: float, n_conversions: int,
                  iterations: int = 2000, alpha: float = 0.05,
                  break_even: float = 0.0,
                  rng: np.random.Generator | None = None) -> SignificanceResult:
    """Bootstrap the ROI estimate at fixed observed spend.

    payouts: empirical per-conversion payout sample.
    spend:   observed spend so far (fixed).
    """
    rng = rng or np.random.default_rng(0)
    n = len(payouts)
    if n == 0 or spend <= 0:
        return SignificanceResult(n_conversions, spend, 0.0, 0.0, 0.0, 0.5,
                                  False, "inconclusive")

    # resample n payouts -> revenue -> roi, `iterations` times (vectorised)
    idx = rng.integers(0, n, size=(iterations, n))
    boot_rev = payouts[idx].sum(axis=1)
    boot_roi = (boot_rev - spend) / spend

    roi_point = (payouts.sum() - spend) / spend
    lo, hi = np.percentile(boot_roi, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_prof = float(np.mean(boot_roi > break_even))

    if lo > break_even:
        distinguishable, direction = True, "profitable"
    elif hi < break_even:
        distinguishable, direction = True, "unprofitable"
    else:
        distinguishable, direction = False, "inconclusive"

    return SignificanceResult(
        n_conversions=n_conversions, spend=spend, roi_point=float(roi_point),
        ci_low=float(lo), ci_high=float(hi), p_profitable=p_prof,
        distinguishable=distinguishable, direction=direction,
    )
