"""
src/feedback.py

Layer 6 — Feedback Surface.
Captures analyst signal on every output element.
Signals feed the Layer 2 retrieval ranking and Layer 3 intuition models
once enough volume has accumulated.

Feedback types (from Zenith blueprint §6.1):
  evidence_relevance   — thumbs on a specific claim's evidence
  claim_quality        — analyst corrects or confirms a claim decomposition
  contradiction_accuracy — confirms/rejects a flagged contradiction
  overall              — 1-5 star rating on the full report output
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from .db import collections, is_connected
from .models import FeedbackSignal

logger = logging.getLogger(__name__)

VALID_TYPES = {
    "evidence_relevance",
    "claim_quality",
    "contradiction_accuracy",
    "overall",
}
VALID_VALUES = {"positive", "negative", "1", "2", "3", "4", "5"}


async def save_feedback(
    validation_id: str,
    feedback_type: str,
    feedback_value: str,
    claim_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Persist one feedback signal. Returns the feedback_id.
    Raises ValueError for invalid type/value so the endpoint can return 400.
    Silently skips DB write if MongoDB is unavailable.
    """
    if feedback_type not in VALID_TYPES:
        raise ValueError(
            f"feedback_type must be one of {sorted(VALID_TYPES)}, got {feedback_type!r}"
        )
    if feedback_value not in VALID_VALUES:
        raise ValueError(
            f"feedback_value must be one of {sorted(VALID_VALUES)}, got {feedback_value!r}"
        )

    feedback_id = str(uuid.uuid4())
    signal = FeedbackSignal(
        feedback_id=feedback_id,
        validation_id=validation_id,
        claim_id=claim_id,
        feedback_type=feedback_type,
        feedback_value=feedback_value,
        notes=notes,
        created_at=datetime.utcnow(),
    )

    try:
        if not is_connected():
            logger.info("[feedback] MongoDB not connected — signal %s not persisted", feedback_id)
            return feedback_id
        await collections.feedback_signals.insert_one(
            {**signal.model_dump(), "_id": feedback_id}
        )
        logger.info("[feedback] %s on validation=%s claim=%s value=%s",
                    feedback_type, validation_id, claim_id, feedback_value)
    except Exception as exc:
        logger.warning("[feedback] failed to save signal: %s", exc)

    return feedback_id


async def get_feedback_for_validation(validation_id: str) -> list[dict]:
    """All feedback signals for a given validation run."""
    try:
        if not is_connected():
            return []
        docs = await collections.feedback_signals.find(
            {"validation_id": validation_id},
            sort=[("created_at", 1)],
        ).to_list(length=500)
        return [{k: v for k, v in d.items() if k != "_id"} for d in docs]
    except Exception as exc:
        logger.warning("[feedback] get failed for %s: %s", validation_id, exc)
        return []


async def get_feedback_stats(ticker: Optional[str] = None) -> dict:
    """
    Aggregate feedback stats — used by the governance layer to decide
    when to trigger a retraining run.

    Returns counts per feedback_type and per feedback_value, plus
    total signal volume. Optionally scoped to a ticker by joining
    with audit_logs.
    """
    try:
        if not is_connected():
            return {"total": 0, "by_type": {}, "by_value": {}, "ticker": ticker}

        if ticker:
            # Find all validation_ids for this ticker via audit_logs
            audit_ids = await collections.audit_logs.distinct(
                "validation_id", {"ticker": ticker.upper()}
            )
            match_stage = {"$match": {"validation_id": {"$in": audit_ids}}}
        else:
            match_stage = {"$match": {}}

        pipeline = [
            match_stage,
            {"$facet": {
                "by_type": [
                    {"$group": {"_id": "$feedback_type", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                ],
                "by_value": [
                    {"$group": {"_id": "$feedback_value", "count": {"$sum": 1}}},
                    {"$sort": {"_id": 1}},
                ],
                "total": [{"$count": "n"}],
            }},
        ]

        result = await collections.feedback_signals.aggregate(pipeline).to_list(length=1)
        if not result:
            return {"total": 0, "by_type": {}, "by_value": {}, "ticker": ticker}

        facet = result[0]
        return {
            "total": facet["total"][0]["n"] if facet["total"] else 0,
            "by_type": {r["_id"]: r["count"] for r in facet["by_type"]},
            "by_value": {r["_id"]: r["count"] for r in facet["by_value"]},
            "ticker": ticker,
        }
    except Exception as exc:
        logger.warning("[feedback] stats failed: %s", exc)
        return {"total": 0, "by_type": {}, "by_value": {}, "ticker": ticker}
