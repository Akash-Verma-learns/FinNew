from __future__ import annotations

"""
Non-GAAP reconciliation parser.

Entry point (ingestion): `ingest_8k_nongaap(ticker, cik)`
Entry point (lookup):    `lookup_nongaap(claim)`

Flow:
  1. Fetch recent 8-K filings from EDGAR submissions API
  2. Filter for earnings press releases (Exhibit 99.1 documents)
  3. Fetch HTML, find tables containing "non-gaap" / "adjusted" / "reconciliation"
  4. Parse each reconciliation row: metric name, GAAP value, non-GAAP value
  5. Store in non_gaap_metrics collection
  6. lookup_nongaap queries the collection and returns a ValidationResult
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import httpx
from bs4 import BeautifulSoup, Tag

from .db import collections, is_connected
from .models import Citation, ValidationResult, ValidationStatus
from .xbrl_lookup import EDGAR_BASE, HEADERS, _compare, _format_value, _normalize_ticker, _parse_value

logger = logging.getLogger(__name__)

MAX_8K_DOCS = 8          # how many 8-K press releases to process per ticker
MAX_HTML_CHARS = 300_000

# Keywords that identify a non-GAAP reconciliation table
_NONGAAP_TABLE_KEYWORDS = frozenset({
    "non-gaap", "non gaap", "adjusted", "reconciliation",
    "gaap to non-gaap", "non-gaap reconciliation",
})

# Keywords in metric names that mark common non-GAAP items
_NONGAAP_METRIC_KEYWORDS = frozenset({
    "non-gaap", "non gaap", "adjusted", "excluding", "ex-", "excl.",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ingest_8k_nongaap(ticker: str, cik: str) -> dict:
    """
    Fetch recent 8-K earnings press releases for `ticker/cik` and extract
    non-GAAP reconciliation metrics into the non_gaap_metrics collection.
    Returns {"metrics_written": int}.
    """
    if not is_connected():
        return {"metrics_written": 0, "status": "db_unavailable"}

    press_release_docs = await _get_8k_press_releases(cik)
    if not press_release_docs:
        logger.info("[nongaap] no 8-K press releases found for CIK=%s", cik)
        return {"metrics_written": 0, "status": "no_8k"}

    written = 0
    for doc_info in press_release_docs[:MAX_8K_DOCS]:
        try:
            html = await _fetch_html(doc_info["url"])
            if not html:
                continue
            metrics = _parse_nongaap_from_html(html, doc_info["period"])
            if not metrics:
                continue
            for m in metrics:
                doc_id = f"{cik}__{doc_info['period_end']}__{m['metric_name']}"
                doc = {
                    "_id": doc_id,
                    "cik": cik,
                    "ticker": ticker,
                    "period": doc_info["period"],
                    "period_end": doc_info["period_end"],
                    "metric_name": m["metric_name"],
                    "gaap_value": m.get("gaap_value"),
                    "non_gaap_value": m["non_gaap_value"],
                    "adjustments": m.get("adjustments", []),
                    "unit": m.get("unit", "USD"),
                    "accession_number": doc_info["accession"],
                    "filing_date": doc_info["filed"],
                    "edgar_url": doc_info["url"],
                    "ingested_at": _now(),
                }
                await collections.non_gaap_metrics.update_one(
                    {"_id": doc_id},
                    {"$setOnInsert": doc},
                    upsert=True,
                )
                written += 1
        except Exception as exc:
            logger.warning("[nongaap] failed to process 8-K %s: %s", doc_info.get("url"), exc)

    logger.info("[nongaap] wrote %d non-GAAP metrics for CIK=%s", written, cik)
    return {"metrics_written": written, "status": "ok"}


async def lookup_nongaap(claim) -> Optional[ValidationResult]:
    """
    Look up a non-GAAP metric claim against the non_gaap_metrics collection.
    Returns None if DB unavailable, no data, or metric not matchable.
    """
    try:
        if not is_connected() or not claim.ticker:
            return None

        ticker = _normalize_ticker(claim.ticker)
        cik_doc = await collections.ticker_lookup.find_one({"_id": ticker})
        if not cik_doc:
            return None
        cik = cik_doc["cik"]

        count = await collections.non_gaap_metrics.count_documents({"cik": cik}, limit=1)
        if count == 0:
            logger.info("  [nongaap] no data for CIK=%s — run ingest first", cik)
            return None

        metric_lower = (claim.metric or "").lower()
        query: dict = {"cik": cik}
        if claim.period:
            year = re.search(r"20\d{2}", claim.period)
            if year:
                query["period_end"] = {"$regex": year.group()}

        best_doc = None
        best_score = 0.0

        async for doc in collections.non_gaap_metrics.find(query):
            score = _metric_similarity(metric_lower, doc.get("metric_name", ""))
            if score > best_score:
                best_score = score
                best_doc = doc

        if best_doc is None or best_score < 0.5:
            logger.info("  [nongaap] no matching metric for %r (best_score=%.2f)", claim.metric, best_score)
            return None

        non_gaap_val = best_doc.get("non_gaap_value")
        if non_gaap_val is None:
            return None

        stated = _parse_value(claim.value or "")
        if stated is None:
            return None

        status, confidence, discrepancy = _compare(claim.value, non_gaap_val)
        formatted = _format_value(non_gaap_val)
        metric_name = best_doc["metric_name"]
        period = best_doc.get("period", best_doc.get("period_end", ""))
        edgar_url = best_doc.get("edgar_url", "")
        accession = best_doc.get("accession_number", "")

        result = ValidationResult(claim_id=claim.id)
        result.status = status
        result.confidence = confidence
        result.actual_value = formatted
        result.discrepancy = discrepancy
        result.filing_source = f"8-K Earnings Release (non-GAAP) — {metric_name}, {period}"
        result.cik = cik
        result.accession_number = accession or None
        result.filing_date = best_doc.get("filing_date")
        result.edgar_url = edgar_url or None
        result.citations = [c for c in [edgar_url] if c]
        result.structured_citations = [
            Citation(
                source="8K_NONGAAP",
                label=f"8-K Earnings Release — {metric_name}",
                url=edgar_url or None,
                ticker=ticker,
                filing=f"8-K {period}",
                accession=accession or None,
                field=metric_name,
                value=formatted,
                period=period,
            )
        ]
        result.reasoning = (
            f"Claimed: {claim.value} | 8-K non-GAAP ({metric_name}): {formatted} ({discrepancy})"
        )
        logger.info("  [nongaap] HIT metric=%r value=%s status=%s", metric_name, formatted, status.value)
        return result

    except Exception as exc:
        logger.debug("[nongaap] lookup_nongaap skipped (%s)", exc)
        return None


# ---------------------------------------------------------------------------
# Helpers — EDGAR 8-K discovery
# ---------------------------------------------------------------------------

async def _get_8k_press_releases(cik: str) -> list[dict]:
    """Return list of {url, period, period_end, accession, filed} for recent 8-K ex-99.1 docs."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        async with httpx.AsyncClient(headers=HEADERS) as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("[nongaap] submissions fetch failed: %s", exc)
        return []

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accns = filings.get("accessionNumber", [])
    filed_dates = filings.get("filingDate", [])
    report_dates = filings.get("reportDate", [])
    primary_docs = filings.get("primaryDocument", [])

    cik_bare = cik.lstrip("0")
    results: list[dict] = []

    for form, accn, filed, rdate, primary_doc in zip(forms, accns, filed_dates, report_dates, primary_docs):
        if form != "8-K":
            continue
        if not primary_doc:
            continue
        accn_clean = accn.replace("-", "")
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/{accn_clean}/{primary_doc}"
        # Derive a human-readable period (e.g. "Q4 2024") from reportDate
        period_label = _period_label(rdate)
        results.append({
            "url": doc_url,
            "period": period_label,
            "period_end": rdate or filed,
            "accession": accn,
            "filed": filed,
        })
        if len(results) >= MAX_8K_DOCS:
            break

    return results


def _period_label(rdate: str) -> str:
    """Convert '2024-09-28' → 'Q4 2024' approximately."""
    if not rdate or len(rdate) < 7:
        return rdate or ""
    try:
        month = int(rdate[5:7])
        year = rdate[:4]
        quarter = (month - 1) // 3 + 1
        return f"Q{quarter} {year}"
    except Exception:
        return rdate


# ---------------------------------------------------------------------------
# Helpers — HTML parsing
# ---------------------------------------------------------------------------

async def _fetch_html(url: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
            resp = await client.get(url, timeout=30)
            resp.raise_for_status()
            return resp.text[:MAX_HTML_CHARS]
    except Exception as exc:
        logger.debug("[nongaap] fetch failed (%s): %s", url, exc)
        return None


def _parse_nongaap_from_html(html: str, period: str) -> list[dict]:
    """
    Extract non-GAAP reconciliation rows from an 8-K HTML document.
    Returns list of {metric_name, gaap_value, non_gaap_value, adjustments, unit}.
    """
    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []

    for table in soup.find_all("table"):
        if not isinstance(table, Tag):
            continue
        table_text = table.get_text(" ", strip=True).lower()
        if not any(kw in table_text for kw in _NONGAAP_TABLE_KEYWORDS):
            continue
        parsed = _parse_reconciliation_table(table)
        results.extend(parsed)

    return results


def _parse_reconciliation_table(table: Tag) -> list[dict]:
    """
    Parse a single HTML table that contains non-GAAP reconciliation data.
    Returns list of {metric_name, gaap_value, non_gaap_value, adjustments, unit}.
    """
    rows = table.find_all("tr")
    if not rows:
        return []

    results: list[dict] = []
    gaap_value: Optional[float] = None
    adjustments: list[dict] = []
    non_gaap_value: Optional[float] = None
    current_metric: Optional[str] = None
    unit = "USD"

    for row in rows:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        label = cells[0].get_text(" ", strip=True).strip()
        label_lower = label.lower()

        # Detect per-share unit
        if "per share" in label_lower or "per diluted" in label_lower or "eps" in label_lower:
            unit = "USD/share"

        # Skip header rows without numeric data
        values = [_parse_cell_value(c.get_text(" ", strip=True)) for c in cells[1:]]
        # Take the first non-None value (leftmost period column)
        first_val = next((v for v in values if v is not None), None)

        if first_val is None:
            # This is likely a header or section label row
            if label and len(label) > 3:
                # Start a new metric block if label looks like a GAAP metric name
                if _is_gaap_header(label_lower):
                    # Save previous group if complete
                    if current_metric and non_gaap_value is not None:
                        results.append({
                            "metric_name": current_metric,
                            "gaap_value": gaap_value,
                            "non_gaap_value": non_gaap_value,
                            "adjustments": list(adjustments),
                            "unit": unit,
                        })
                    current_metric = label
                    gaap_value = None
                    adjustments = []
                    non_gaap_value = None
                    unit = "USD"
            continue

        if not current_metric:
            current_metric = label or "Non-GAAP Metric"

        # Identify this row as GAAP baseline, an adjustment, or the non-GAAP total
        if _is_nongaap_total(label_lower):
            non_gaap_value = first_val
        elif _is_gaap_total(label_lower) and gaap_value is None:
            gaap_value = first_val
        else:
            adjustments.append({"label": label, "value": first_val})

    # Save the last group
    if current_metric and non_gaap_value is not None:
        results.append({
            "metric_name": current_metric,
            "gaap_value": gaap_value,
            "non_gaap_value": non_gaap_value,
            "adjustments": list(adjustments),
            "unit": unit,
        })

    return results


def _parse_cell_value(text: str) -> Optional[float]:
    """Parse a table cell value — handles ($123.4), $123.4M, 123,456, —, etc."""
    text = text.strip()
    if not text or text in ("—", "-", "–", "N/A", "n/a", ""):
        return None
    # Parentheses = negative: (123.4) → -123.4
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    text = text.replace("$", "").replace(",", "").strip()
    # Handle millions/billions/thousands suffix
    multiplier = 1.0
    if text.endswith("M") or text.endswith("m"):
        multiplier = 1e6
        text = text[:-1]
    elif text.endswith("B") or text.endswith("b"):
        multiplier = 1e9
        text = text[:-1]
    elif text.endswith("K") or text.endswith("k"):
        multiplier = 1e3
        text = text[:-1]
    try:
        val = float(text) * multiplier
        return -val if negative else val
    except ValueError:
        return None


def _is_nongaap_total(label_lower: str) -> bool:
    return any(kw in label_lower for kw in ("non-gaap", "non gaap", "adjusted", "adj."))


def _is_gaap_total(label_lower: str) -> bool:
    return any(kw in label_lower for kw in ("gaap", "reported", "as reported"))


def _is_gaap_header(label_lower: str) -> bool:
    return any(kw in label_lower for kw in (
        "net income", "earnings per share", "eps", "operating income",
        "gross profit", "ebitda", "revenue", "operating expenses",
    ))


# ---------------------------------------------------------------------------
# Metric similarity matching
# ---------------------------------------------------------------------------

# Words that mark a metric as "non-GAAP" but don't identify *which* metric —
# stripped before comparison so "adjusted EPS" can match "Non-GAAP Diluted EPS".
_MARKER_WORDS = {"non", "gaap", "nongaap", "adjusted", "adj", "core", "underlying", "normalized"}

# The financial term that actually identifies the metric — at least one must
# overlap, or the match is rejected regardless of incidental word overlap.
_PRIMARY_TERMS = {
    "eps", "ebitda", "ebit", "earnings", "income", "margin",
    "revenue", "profit", "loss", "fcf", "cashflow", "expenses",
}

_STOP_WORDS = {"", "the", "a", "an", "of", "and", "to", "in", "for", "per", "diluted", "basic"}


def _metric_similarity(claim_metric: str, stored_metric: str) -> float:
    """Return 0-1 similarity between a claim metric string and a stored metric name."""
    cm = re.sub(r"[^\w\s]", " ", claim_metric.lower()).strip()
    sm = re.sub(r"[^\w\s]", " ", stored_metric.lower()).strip()
    if cm == sm:
        return 1.0
    if cm in sm or sm in cm:
        return 0.9

    def _core_words(s: str) -> set[str]:
        return set(s.split()) - _MARKER_WORDS - _STOP_WORDS

    cm_words = _core_words(cm)
    sm_words = _core_words(sm)
    if not cm_words or not sm_words:
        return 0.0

    jaccard = len(cm_words & sm_words) / len(cm_words | sm_words)
    primary_overlap = bool((cm_words & _PRIMARY_TERMS) & (sm_words & _PRIMARY_TERMS))

    # Require the core financial term (EPS, EBITDA, margin, ...) to match —
    # otherwise "adjusted operating margin" could falsely match "Non-GAAP Net Income".
    if primary_overlap:
        return max(0.6, jaccard)
    return jaccard * 0.5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
