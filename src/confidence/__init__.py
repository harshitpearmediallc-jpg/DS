"""Phase 2 — the SCALE / HOLD / KILL confidence layer."""

from .recommend import (
    recommend,
    Recommendation,
    load_trust_threshold,
    to_log_record,
    SCALE,
    HOLD,
    KILL,
)
from .significance import bootstrap_roi, SignificanceResult
from .regime import detect_regime, RegimeResult
from .calibration import fit_calibrator, reliability, Calibrator
from . import features

__all__ = [
    "recommend", "Recommendation", "load_trust_threshold", "to_log_record",
    "SCALE", "HOLD", "KILL",
    "bootstrap_roi", "SignificanceResult",
    "detect_regime", "RegimeResult",
    "fit_calibrator", "reliability", "Calibrator",
    "features",
]
