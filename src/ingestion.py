from __future__ import annotations

"""
XBRL ingestion pipeline.

Entry point: `ingest_ticker(ticker)`

Steps:
  1. Resolve ticker → CIK (EDGAR API, cache result in ticker_lookup)
  2. Check filing_index — skip if already ingested for this CIK + period
  3. Fetch all XBRL company facts from EDGAR
  4. Parse each known metric concept → upsert into xbrl_facts
  5. On new filing: invalidate derivation_log for this CIK
  6. Re-compute derived metrics via DEFAULT_GRAPH → store in derivation_log
  7. Mark filing_index.derived_computed = True
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from .db import collections, is_connected
from .formula_graph import DEFAULT_GRAPH, MultiResolution
from .xbrl_lookup import (
    CONCEPT_MAP,
    EDGAR_BASE,
    HEADERS,
    TICKER_URL,
    _normalize_ticker,
    _find_period_value,
    get_company_facts,
)

logger = logging.getLogger(__name__)

# All metric aliases we want to store in xbrl_facts
_ALL_ALIASES: list[str] = list(CONCEPT_MAP.keys())


async def ingest_ticker(ticker: str) -> dict:
    """
    Fetch and store all known XBRL facts for `ticker`.

    Returns a summary dict: {cik, periods_processed, facts_written, derived_written, skipped}.
    Raises RuntimeError if DB is not connected.
    """
    if not is_connected():
        raise RuntimeError("DB not initialised — cannot ingest")

    ticker = _normalize_ticker(ticker)
    cik = await _resolve_and_cache_cik(ticker)
    if not cik:
        raise ValueError(f"Ticker {ticker!r} not found in SEC EDGAR")

    facts_json = await get_company_facts(cik)
    gaap = facts_json.get("facts", {}).get("us-gaap", {})

    # Collect (period_end, concept, value) tuples across all metrics
    rows: list[dict] = []
    for alias, concepts in CONCEPT_MAP.items():
        for concept in concepts:
            if concept not in gaap:
                continue
            units = gaap[concept].get("units", {})
            found = _find_period_value(units, period=None)
            if found:
                val, period_end, accn, filed = found
                rows.append({
                    "alias": alias,
                    "concept": concept,
                    "value": val,
                    "period_end": period_end,
                    "accn": accn,
                    "filed": filed,
                })
            # Also collect all annual entries for this concept (historical)
            for entry in _all_annual_entries(units):
                if entry["end"] != (found[1] if found else ""):
                    rows.append({
                        "alias": alias,
                        "concept": concept,
                        "value": entry["val"],
                        "period_end": entry["end"],
                        "accn": entry.get("accn", ""),
                        "filed": entry.get("filed", ""),
                    })

    # Upsert xbrl_facts — grouped by (period_end, concept)
    facts_written = 0
    periods_seen: set[str] = set()
    for row in rows:
        doc_id = f"{cik}__{row['period_end']}__{row['concept']}"
        cik_bare = cik.lstrip("0")
        accn_clean = row["accn"].replace("-", "")
        edgar_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/{accn_clean}/"
            if accn_clean else None
        )
        doc = {
            "_id": doc_id,
            "cik": cik,
            "ticker": ticker,
            "period_end": row["period_end"],
            "period_type": "annual",
            "xbrl_concept": row["concept"],
            "metric_alias": row["alias"],
            "value": row["value"],
            "unit": "USD",
            "form": "10-K",
            "accession_number": row["accn"] or None,
            "filing_date": row["filed"] or None,
            "edgar_url": edgar_url,
            "ingested_at": _now(),
        }
        await collections.xbrl_facts.update_one(
            {"_id": doc_id},
            {"$setOnInsert": doc},
            upsert=True,
        )
        facts_written += 1
        periods_seen.add(row["period_end"])

    # Invalidate derivation_log for this CIK (facts may have changed)
    del_result = await collections.derivation_log.delete_many({"cik": cik})
    logger.info("Invalidated %d derivation_log entries for CIK %s", del_result.deleted_count, cik)

    # Re-compute derived metrics for each period
    derived_written = 0
    for period_end in sorted(periods_seen):
        derived_written += await _compute_and_store_derived(cik, period_end)

    # Upsert filing_index
    await _upsert_filing_index(cik, ticker, periods_seen)

    # Text ingestion for the most recent period (other periods on explicit /api/ingest-text call)
    text_result = {"chunks_written": 0}
    if periods_seen:
        latest_period = max(periods_seen)
        try:
            from .text_ingestion import ingest_text
            text_result = await ingest_text(ticker, cik, latest_period)
        except Exception as exc:
            logger.warning("Text ingestion failed (non-fatal): %s", exc)

    return {
        "cik": cik,
        "ticker": ticker,
        "periods_processed": len(periods_seen),
        "facts_written": facts_written,
        "derived_written": derived_written,
        "text_chunks_written": text_result.get("chunks_written", 0),
    }


async def _resolve_and_cache_cik(ticker: str) -> Optional[str]:
    """Resolve ticker → CIK. Check ticker_lookup first, fall back to EDGAR."""
    doc = await collections.ticker_lookup.find_one({"_id": ticker})
    if doc:
        return doc["cik"]

    async with httpx.AsyncClient(headers=HEADERS) as client:
        resp = await client.get(TICKER_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()

    cik: Optional[str] = None
    batch: list = []
    for entry in data.values():
        t = entry.get("ticker", "").upper()
        c = str(entry.get("cik_str", "")).zfill(10)
        name = entry.get("title", "")
        batch.append({
            "_id": t,
            "cik": c,
            "canonical_ticker": t,
            "aliases": [t, f"${t}", f"NASDAQ:{t}", f"NYSE:{t}"],
            "company_name": name,
        })
        if t == ticker:
            cik = c

    # Bulk upsert all tickers into ticker_lookup
    if batch:
        from pymongo import UpdateOne
        ops = [
            UpdateOne({"_id": b["_id"]}, {"$setOnInsert": b}, upsert=True)
            for b in batch
        ]
        await collections.ticker_lookup.bulk_write(ops, ordered=False)
        logger.info("ticker_lookup: upserted %d entries", len(batch))

    return cik


def _all_annual_entries(units: dict) -> list[dict]:
    """Return every annual 10-K / 20-F entry across all unit keys."""
    entries: list[dict] = []
    for unit_key in ("USD", "USD/shares", "shares"):
        entries = units.get(unit_key, [])
        if entries:
            break
    if not entries:
        entries = next(iter(units.values()), [])
    return [
        e for e in entries
        if e.get("form") in ("10-K", "20-F", "10-K/A", "20-F/A")
        and len(e.get("end", "")) == 10
    ]


# Maps metric_alias (lowercase, from CONCEPT_MAP) to the canonical input names
# used by FormulaGraph formula nodes. Without this, "RevenueFromContract..."
# never gets stored as "Revenues", so GrossMarginPct and other formulas fail.
_ALIAS_TO_FORMULA_KEYS: dict[str, list[str]] = {
    "revenue":             ["Revenues"],
    "gross profit":        ["GrossProfit"],
    "operating income":    ["OperatingIncomeLoss"],
    "net income":          ["NetIncomeLoss"],
    "cash":                ["CashAndCashEquivalentsAtCarryingValue"],
    "total debt":          ["LongTermDebt"],
    "total assets":        ["Assets"],
    "equity":              ["StockholdersEquity"],
    "operating cash flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capital expenditures":["PaymentsToAcquirePropertyPlantAndEquipment"],
    "cost of goods sold":  ["CostOfRevenue"],
    "eps":                 ["EarningsPerShareDiluted"],
    "shares outstanding":  ["CommonStockSharesOutstanding"],
}


async def _compute_and_store_derived(cik: str, period_end: str) -> int:
    """
    Resolve all formula nodes in DEFAULT_GRAPH using the stored xbrl_facts for
    (cik, period_end) and write results to derivation_log.
    Returns count of documents written.
    """
    # Load known raw facts from xbrl_facts for this period
    cursor = collections.xbrl_facts.find({"cik": cik, "period_end": period_end})
    known: dict[str, float] = {}
    async for doc in cursor:
        alias = doc["metric_alias"]
        val = doc["value"]
        known.setdefault(alias, val)
        known.setdefault(doc["xbrl_concept"], val)
        # Add canonical FormulaGraph input names so formulas can resolve.
        # e.g. "RevenueFromContractWithCustomer..." stored as "Revenues"
        for fkey in _ALIAS_TO_FORMULA_KEYS.get(alias, []):
            known.setdefault(fkey, val)

    written = 0
    for concept in DEFAULT_GRAPH._nodes:
        resolution: MultiResolution = DEFAULT_GRAPH.resolve(concept, known)
        if not resolution.results:
            continue

        primary = resolution.primary
        all_defs = [
            {"source": r.source, "value": r.value}
            for r in resolution.results
        ]
        delta = resolution.delta or 0.0

        doc_id = f"{cik}__{period_end}__{concept}__DEFAULT_GRAPH"
        doc = {
            "_id": doc_id,
            "cik": cik,
            "period_end": period_end,
            "concept": concept,
            "formula_source": primary.source if primary else "unknown",
            "value": primary.value if primary else None,
            "inputs_used": primary.inputs_used if primary else {},
            "all_definitions": all_defs,
            "delta": delta,
            "computed_at": _now(),
        }
        await collections.derivation_log.update_one(
            {"_id": doc_id},
            {"$set": doc},
            upsert=True,
        )
        written += 1

    return written


async def _upsert_filing_index(cik: str, ticker: str, periods: set[str]) -> None:
    for period_end in periods:
        doc_id = f"{cik}__{period_end}__10-K"
        await collections.filing_index.update_one(
            {"_id": doc_id},
            {"$set": {
                "cik": cik,
                "ticker": ticker,
                "form": "10-K",
                "period_end": period_end,
                "xbrl_ingested": True,
                "derived_computed": True,
                "ingested_at": _now(),
            }},
            upsert=True,
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
