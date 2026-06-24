from __future__ import annotations

"""
Async MongoDB connection layer (motor).

Usage:
    from .db import get_db, collections
    col = collections.xbrl_facts

Call `init_db()` once at app startup (in the FastAPI lifespan).
Call `close_db()` at shutdown.
"""

import logging
import os
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


class _Collections:
    """Typed accessors for every collection we use."""

    @property
    def ticker_lookup(self) -> AsyncIOMotorCollection:
        return _db["ticker_lookup"]

    @property
    def xbrl_facts(self) -> AsyncIOMotorCollection:
        return _db["xbrl_facts"]

    @property
    def derivation_log(self) -> AsyncIOMotorCollection:
        return _db["derivation_log"]

    @property
    def formula_definitions(self) -> AsyncIOMotorCollection:
        return _db["formula_definitions"]

    @property
    def text_chunks(self) -> AsyncIOMotorCollection:
        return _db["text_chunks"]

    @property
    def filing_index(self) -> AsyncIOMotorCollection:
        return _db["filing_index"]

    @property
    def non_gaap_metrics(self) -> AsyncIOMotorCollection:
        return _db["non_gaap_metrics"]

    @property
    def segment_facts(self) -> AsyncIOMotorCollection:
        return _db["segment_facts"]

    @property
    def credibility_history(self) -> AsyncIOMotorCollection:
        return _db["credibility_history"]

    @property
    def trend_insights(self) -> AsyncIOMotorCollection:
        return _db["trend_insights"]

    @property
    def audit_logs(self) -> AsyncIOMotorCollection:
        return _db["audit_logs"]

    @property
    def feedback_signals(self) -> AsyncIOMotorCollection:
        return _db["feedback_signals"]


collections = _Collections()


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("DB not initialised — call init_db() first")
    return _db


async def init_db() -> None:
    global _client, _db
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "finvalidator")

    kwargs: dict = {"serverSelectionTimeoutMS": 5000}

    is_atlas = "mongodb+srv" in uri or ".mongodb.net" in uri
    if is_atlas:
        # Windows Python 3.12 + OpenSSL fails the Atlas certificate chain handshake
        # with TLSV1_ALERT_INTERNAL_ERROR even with certifi. Bypass cert validation
        # for the dev environment; set MONGODB_TLS_VERIFY=true to re-enable.
        if os.getenv("MONGODB_TLS_VERIFY", "false").lower() in ("true", "1"):
            try:
                import certifi
                kwargs["tlsCAFile"] = certifi.where()
            except ImportError:
                pass
        else:
            kwargs["tlsAllowInvalidCertificates"] = True
    else:
        # Local / non-Atlas — honour explicit TLS env var, default off
        if os.getenv("MONGODB_TLS", "").lower() in ("true", "1"):
            kwargs["tls"] = True

    _client = AsyncIOMotorClient(uri, **kwargs)
    _db = _client[db_name]
    try:
        await _ensure_indexes()
        logger.info("MongoDB connected — db=%s uri=%s", db_name, uri.split("@")[-1])
    except Exception:
        _client.close()
        _client = None
        _db = None
        raise


async def close_db() -> None:
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
    logger.info("MongoDB connection closed")


async def _ensure_indexes() -> None:
    from pymongo import ASCENDING, DESCENDING, IndexModel

    # ticker_lookup
    await collections.ticker_lookup.create_indexes([
        IndexModel([("cik", ASCENDING)]),
        IndexModel([("aliases", ASCENDING)]),
    ])

    # xbrl_facts — compound covers (cik, period_end desc, metric_alias)
    await collections.xbrl_facts.create_indexes([
        IndexModel([("cik", ASCENDING), ("period_end", DESCENDING), ("metric_alias", ASCENDING)]),
    ])

    # derivation_log — compound covers (cik, period_end desc, concept)
    await collections.derivation_log.create_indexes([
        IndexModel([("cik", ASCENDING), ("period_end", DESCENDING), ("concept", ASCENDING)]),
    ])

    # filing_index
    await collections.filing_index.create_indexes([
        IndexModel([("cik", ASCENDING), ("form", ASCENDING), ("period_end", DESCENDING)]),
    ])

    # text_chunks — standard compound; vector index must be created in Atlas UI
    await collections.text_chunks.create_indexes([
        IndexModel([("cik", ASCENDING), ("form", ASCENDING), ("section", ASCENDING), ("period_end", DESCENDING)]),
    ])

    # non_gaap_metrics — 8-K earnings release non-GAAP reconciliation data
    await collections.non_gaap_metrics.create_indexes([
        IndexModel([("cik", ASCENDING), ("period_end", DESCENDING), ("metric_name", ASCENDING)]),
        IndexModel([("cik", ASCENDING), ("metric_name", ASCENDING)]),
    ])

    # segment_facts — ASC 280 segment-level revenue/income from 10-K notes
    await collections.segment_facts.create_indexes([
        IndexModel([("cik", ASCENDING), ("period_end", DESCENDING), ("segment_name", ASCENDING), ("metric", ASCENDING)]),
        IndexModel([("cik", ASCENDING), ("metric", ASCENDING)]),
    ])

    # credibility_history — one snapshot per pipeline run per ticker
    await collections.credibility_history.create_indexes([
        IndexModel([("ticker", ASCENDING), ("timestamp", DESCENDING)], name="ticker_timestamp_desc"),
        IndexModel([("run_id", ASCENDING)], unique=True),
    ])

    # trend_insights — cached trend outputs per ticker
    await collections.trend_insights.create_indexes([
        IndexModel([("ticker", ASCENDING), ("generated_at", DESCENDING)], name="ticker_generated_desc"),
    ])

    # audit_logs — immutable execution trace per validation run (Layer 6)
    await collections.audit_logs.create_indexes([
        IndexModel([("validation_id", ASCENDING)], unique=True, name="validation_id_unique"),
        IndexModel([("created_at", DESCENDING)], name="audit_created_desc"),
        IndexModel([("ticker", ASCENDING), ("created_at", DESCENDING)], name="audit_ticker_created"),
    ])

    # feedback_signals — analyst feedback on validation outputs (Layer 6)
    await collections.feedback_signals.create_indexes([
        IndexModel([("validation_id", ASCENDING), ("claim_id", ASCENDING)], name="feedback_validation_claim"),
        IndexModel([("created_at", DESCENDING)], name="feedback_created_desc"),
        IndexModel([("feedback_type", ASCENDING), ("feedback_value", ASCENDING)], name="feedback_type_value"),
    ])

    logger.info("MongoDB indexes ensured")


def is_connected() -> bool:
    return _db is not None
