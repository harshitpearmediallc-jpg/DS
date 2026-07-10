"""Recommendation + outcome logging schema.

Phase 2 emits one record every time the tool makes a SCALE/HOLD/KILL call. The
schema deliberately anticipates a FUTURE trajectory predictor (explicitly out of
scope now): it captures the full observable state at decision time plus a slot to
backfill the realised outcome later, so the logs accumulate into labelled
training data without a schema migration.

Nothing here makes predictions or trains anything today — it only defines the
record and an append-only JSONL writer. Included now so Phase 2 has a stable
contract to write against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class RecommendationRecord:
    # --- identity / timing (timestamp passed in; never generated internally) --
    campaign_id: str
    decided_at: str                     # ISO8601, supplied by caller
    schema_version: int = 1

    # --- observable state at decision time (the future model's features) ------
    conversions_so_far: int = 0
    spend_so_far: float = 0.0
    revenue_so_far: float = 0.0
    observed_roi: float = 0.0
    # per-dimension health snapshot (sub_id/region/creative), opaque to schema
    dimension_snapshot: dict[str, Any] = field(default_factory=dict)

    # --- the recommendation + its justification -------------------------------
    recommendation: str = "HOLD"        # SCALE | HOLD | KILL
    confidence: Optional[float] = None  # calibrated P(profitable), 0..1
    roi_ci_low: Optional[float] = None  # confidence interval on ROI estimate
    roi_ci_high: Optional[float] = None
    conversions_needed_to_trust: Optional[int] = None
    reason: str = ""                    # human-readable, buyer-checkable

    # --- outcome (backfilled later; the label for the future predictor) -------
    outcome_verdict: Optional[str] = None   # winner | loser | None (unknown yet)
    outcome_final_roi: Optional[float] = None
    outcome_resolved_at: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class RecommendationLog:
    """Append-only JSONL log. Read-many, write-append; never mutates rows in place
    (outcome backfill is done by rewriting during a periodic reconciliation job in
    Phase 2, kept out of the hot path)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RecommendationRecord) -> None:
        with open(self.path, "a") as fh:
            fh.write(record.to_json() + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as fh:
            return [json.loads(line) for line in fh if line.strip()]
