"""Recommendation logging schema: append + read roundtrip, outcome backfill slot."""

from __future__ import annotations

from src.logging_.schema import RecommendationRecord, RecommendationLog


def test_record_roundtrip(tmp_path):
    log = RecommendationLog(tmp_path / "rec.jsonl")
    rec = RecommendationRecord(
        campaign_id="C1", decided_at="2025-01-01T00:00:00Z",
        conversions_so_far=9, spend_so_far=100.0, revenue_so_far=130.0,
        observed_roi=0.30, recommendation="HOLD", confidence=0.62,
        conversions_needed_to_trust=25,
        reason="Held: only 9 conversions; +30% ROI is within noise, need ~25 to trust.",
    )
    log.append(rec)
    rows = log.read_all()
    assert len(rows) == 1
    assert rows[0]["recommendation"] == "HOLD"
    # outcome slot exists and is empty, ready for the future trajectory predictor
    assert rows[0]["outcome_verdict"] is None
    assert rows[0]["schema_version"] == 1


def test_append_is_additive(tmp_path):
    log = RecommendationLog(tmp_path / "rec.jsonl")
    for i in range(3):
        log.append(RecommendationRecord(campaign_id=f"C{i}", decided_at="t"))
    assert len(log.read_all()) == 3
