"""Calibration: make the stated confidence mean what it says.

If the tool says "80% likely profitable," then among all past campaigns it called
80%, about 80% must have actually matured profitable. We enforce this by fitting an
isotonic (monotone) map from the raw bootstrap P(profitable) to the empirical
probability, using the Phase-1 ground truth as labels.

Isotonic regression via Pool-Adjacent-Violators (PAV) — implemented here so we keep
the dependency list lean (no scikit-learn). It is monotone-nondecreasing, which is
the right shape: higher raw confidence should never map to lower true probability.

We also report the reliability curve and Expected Calibration Error (ECE) so the
quality of calibration is visible, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Calibrator:
    """Monotone step function fit by PAV; call .predict(p_raw)."""
    x_thresholds: np.ndarray     # sorted raw scores
    y_values: np.ndarray         # calibrated probs (nondecreasing)

    def predict(self, p_raw) -> np.ndarray:
        p_raw = np.atleast_1d(np.asarray(p_raw, dtype=float))
        # right-continuous step interpolation over the fitted thresholds
        idx = np.searchsorted(self.x_thresholds, p_raw, side="right") - 1
        idx = np.clip(idx, 0, len(self.y_values) - 1)
        out = self.y_values[idx]
        return out if out.size > 1 else float(out[0])

    def to_dict(self) -> dict:
        return {"x_thresholds": [float(x) for x in self.x_thresholds],
                "y_values": [float(y) for y in self.y_values]}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(np.asarray(d["x_thresholds"], dtype=float),
                   np.asarray(d["y_values"], dtype=float))


def _pav(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Pool-Adjacent-Violators isotonic fit. Returns (x_sorted, y_fitted)."""
    order = np.argsort(x, kind="mergesort")
    x, y, w = x[order], y[order].astype(float), w[order].astype(float)
    # blocks of (value, weight)
    vals = list(y)
    wts = list(w)
    xs = list(x)
    i = 0
    # iterative pooling
    blocks = [[y[j], w[j], [x[j]]] for j in range(len(y))]
    merged = True
    while merged:
        merged = False
        out = []
        for b in blocks:
            if out and out[-1][0] > b[0]:  # violation -> pool
                v0, w0, xs0 = out.pop()
                nw = w0 + b[1]
                nv = (v0 * w0 + b[0] * b[1]) / nw
                out.append([nv, nw, xs0 + b[2]])
                merged = True
            else:
                out.append(b)
        blocks = out
    # expand blocks into thresholds
    thr, yval = [], []
    for v, _, xs_block in blocks:
        thr.append(min(xs_block))
        yval.append(v)
    return np.array(thr), np.array(yval)


def fit_calibrator(raw_conf: np.ndarray, outcomes: np.ndarray) -> Calibrator:
    """Fit isotonic calibration. raw_conf in [0,1], outcomes in {0,1}."""
    raw_conf = np.asarray(raw_conf, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    w = np.ones_like(raw_conf)
    thr, yval = _pav(raw_conf, outcomes, w)
    return Calibrator(x_thresholds=thr, y_values=yval)


def reliability(conf: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """Return per-bin (mean_conf, empirical_rate, count) and the ECE."""
    conf = np.asarray(conf, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows, ece, total = [], 0.0, len(conf)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        m = (conf >= lo) & (conf < hi if b < n_bins - 1 else conf <= hi)
        if not m.any():
            continue
        mc = conf[m].mean()
        er = outcomes[m].mean()
        cnt = int(m.sum())
        rows.append({"bin_lo": lo, "bin_hi": hi, "mean_conf": mc,
                     "empirical_rate": er, "count": cnt})
        ece += (cnt / total) * abs(mc - er)
    return rows, float(ece)
