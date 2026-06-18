"""
src/audit.py

Layer 6 — Governance & Audit.
Saves and retrieves immutable execution traces for every validation run.
Records are never updated after insert — append-only by convention.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from .db import collections, is_connected
from .embeddings import EMBEDDING_MODEL
from .models import AuditLog, ValidationStatus

logger = logging.getLogger(__name__)


async def save_audit_log(
    validation_id: str,
    input_text: str,
    input_type: str,
    ticker: Optional[str],
    claims: list,
    validations: list[dict],
    scoring: dict,
    timings: dict,
) -> None:
    """
    Persist an immutable audit record for one pipeline run.
    Silently skips if MongoDB is unavailable.
    """
    from .groq_client import ACTIVE_MODEL, ACTIVE_BACKEND

    verified = sum(1 for v in validations if v.get("status") == ValidationStatus.VERIFIED.value)
    contradicted = sum(1 for v in validations if v.get("status") == ValidationStatus.CONTRADICTED.value)
    partially = sum(1 for v in validations if v.get("status") == ValidationStatus.PARTIALLY_VERIFIED.value)
    unverifiable = sum(1 for v in validations if v.get("status") == ValidationStatus.UNVERIFIABLE.value)

    red_flags = scoring.get("red_flags", [])

    record = AuditLog(
        validation_id=validation_id,
        created_at=datetime.utcnow(),
        input_summary=input_text[:400],
        input_type=input_type,
        ticker=ticker.upper() if ticker else None,
        llm_model=ACTIVE_MODEL,
        llm_backend=ACTIVE_BACKEND,
        embedding_model=EMBEDDING_MODEL,
        claims_count=len(claims),
        verified_count=verified,
        contradicted_count=contradicted,
        partially_verified_count=partially,
        unverifiable_count=unverifiable,
        red_flag_count=len(red_flags),
        overall_score=scoring.get("overall_score", 0.0),
        credibility_rating=scoring.get("credibility_rating", "UNKNOWN"),
        analyst_bias=scoring.get("analyst_bias", "BALANCED"),
        extraction_seconds=timings.get("extraction_seconds", 0.0),
        validation_seconds=timings.get("validation_seconds", 0.0),
        scoring_seconds=timings.get("scoring_seconds", 0.0),
        total_seconds=timings.get("total_seconds", 0.0),
        claims=[c.model_dump() if hasattr(c, "model_dump") else c for c in claims],
        validations=validations,
        red_flags=red_flags,
    )

    try:
        if not is_connected():
            logger.info("[audit] MongoDB not connected — skipping audit log %s", validation_id)
            return
        await collections.audit_logs.insert_one(
            {**record.model_dump(), "_id": validation_id}
        )
        logger.info("[audit] saved run %s (score=%.1f, claims=%d, %.1fs)",
                    validation_id, record.overall_score, record.claims_count, record.total_seconds)
    except Exception as exc:
        logger.warning("[audit] failed to save audit log %s: %s", validation_id, exc)


async def get_audit_log(validation_id: str) -> dict | None:
    """Retrieve a single audit record by validation_id."""
    try:
        if not is_connected():
            return None
        doc = await collections.audit_logs.find_one({"validation_id": validation_id})
        if not doc:
            return None
        return {k: v for k, v in doc.items() if k != "_id"}
    except Exception as exc:
        logger.warning("[audit] get failed for %s: %s", validation_id, exc)
        return None


async def list_audit_logs(
    ticker: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """
    Return the most recent audit records, newest first.
    Optionally filter by ticker.
    """
    try:
        if not is_connected():
            return []
        query = {"ticker": ticker.upper()} if ticker else {}
        docs = await collections.audit_logs.find(
            query,
            sort=[("created_at", -1)],
            projection={"claims": 0, "validations": 0},  # exclude heavy fields in list view
        ).to_list(length=limit)
        return [{k: v for k, v in d.items() if k != "_id"} for d in docs]
    except Exception as exc:
        logger.warning("[audit] list failed: %s", exc)
        return []
