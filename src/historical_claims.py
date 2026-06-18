"""
src/historical_claims.py

Reads and writes CredibilitySnapshot objects to MongoDB.
Used by the trend engine to build historical context per ticker.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

from .db import collections, is_connected
from .models import CredibilitySnapshot, ValidationStatus

logger = logging.getLogger(__name__)


async def save_snapshot(
    ticker: str,
    score_result: dict,
    claims: list,
    validations: list[dict],
    report_type: str = "report",
    source_label: str = "",
) -> str:
    """
    Persist a CredibilitySnapshot after a successful pipeline run.
    Returns the run_id. Silently skips if MongoDB is unavailable.
    """
    run_id = str(uuid.uuid4())

    verified = sum(
        1 for v in validations
        if v.get("status") == ValidationStatus.VERIFIED.value
    )
    contradicted = sum(
        1 for v in validations
        if v.get("status") == ValidationStatus.CONTRADICTED.value
    )
    partially = sum(
        1 for v in validations
        if v.get("status") == ValidationStatus.PARTIALLY_VERIFIED.value
    )
    unverifiable = sum(
        1 for v in validations
        if v.get("status") == ValidationStatus.UNVERIFIABLE.value
    )

    high_flags = [
        f["message"] for f in score_result.get("red_flags", [])
        if f.get("severity") == "HIGH"
    ]

    snapshot = CredibilitySnapshot(
        ticker=ticker.upper(),
        run_id=run_id,
        timestamp=datetime.utcnow(),
        overall_score=score_result.get("overall_score", 0.0),
        credibility_rating=score_result.get("credibility_rating", "UNKNOWN"),
        analyst_bias=score_result.get("analyst_bias", "BALANCED"),
        report_type=report_type,
        total_claims=len(claims),
        verified_count=verified,
        contradicted_count=contradicted,
        partially_verified_count=partially,
        unverifiable_count=unverifiable,
        red_flag_count=len(score_result.get("red_flags", [])),
        high_severity_flags=high_flags,
        source_label=source_label or f"Report {datetime.utcnow().strftime('%Y-%m-%d')}",
    )

    try:
        if not is_connected():
            logger.info("[history] MongoDB not connected — skipping snapshot for %s", ticker)
            return run_id
        await collections.credibility_history.insert_one(
            {**snapshot.model_dump(), "_id": run_id}
        )
        logger.info("[history] saved snapshot %s for %s (score=%.1f)",
                    run_id, ticker, snapshot.overall_score)
    except Exception as exc:
        logger.warning("[history] failed to save snapshot for %s: %s", ticker, exc)

    return run_id


async def get_ticker_history(ticker: str, limit: int = 10) -> list[CredibilitySnapshot]:
    """
    Retrieve the last `limit` CredibilitySnapshots for a ticker, newest first.
    Returns empty list if none exist or MongoDB is unavailable.
    """
    try:
        if not is_connected():
            return []
        docs = await collections.credibility_history.find(
            {"ticker": ticker.upper()},
            sort=[("timestamp", -1)],
        ).to_list(length=limit)
        # Exclude MongoDB's _id field before constructing Pydantic models
        return [
            CredibilitySnapshot(**{k: v for k, v in d.items() if k != "_id"})
            for d in docs
        ]
    except Exception as exc:
        logger.warning("[history] failed to fetch history for %s: %s", ticker, exc)
        return []


async def get_all_tickers_with_history() -> list[str]:
    """Returns all tickers that have at least one snapshot."""
    try:
        if not is_connected():
            return []
        return await collections.credibility_history.distinct("ticker")
    except Exception as exc:
        logger.warning("[history] failed to list tickers: %s", exc)
        return []
