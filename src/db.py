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


collections = _Collections()


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("DB not initialised — call init_db() first")
    return _db


async def init_db() -> None:
    global _client, _db
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "finvalidator")
    try:
        import certifi
        tls_ca = certifi.where()
    except ImportError:
        tls_ca = None
    kwargs: dict = {"serverSelectionTimeoutMS": 5000, "tls": True}
    if tls_ca:
        kwargs["tlsCAFile"] = tls_ca
    _client = AsyncIOMotorClient(uri, **kwargs)
    _db = _client[db_name]
    await _ensure_indexes()
    logger.info("MongoDB connected — db=%s uri=%s", db_name, uri.split("@")[-1])


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

    logger.info("MongoDB indexes ensured")


def is_connected() -> bool:
    return _db is not None
