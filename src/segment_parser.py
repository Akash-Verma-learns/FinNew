from __future__ import annotations

"""
ASC 280 segment table parser.

Entry point (ingestion): `ingest_segments(ticker, cik, period_end, html, source_url)`
Entry point (lookup):    `lookup_segment_fact(claim)`

Flow:
  1. Find the segment footnote section in the 10-K HTML using regex/heading patterns
  2. Parse all tables in that section — rows = segments, columns = metrics + periods
  3. Extract (segment_name, metric, value) for revenue and operating income
  4. Store in segment_facts collection
  5. lookup_segment_fact queries segment_facts and returns a ValidationResult
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from bs4 import BeautifulSoup, Tag

from .db import collections, is_connected
from .models import Citation, ValidationResult, ValidationStatus
from .xbrl_lookup import _compare, _format_value, _normalize_ticker, _parse_value

logger = logging.getLogger(__name__)

# Patterns that mark the start of the segment footnote section
_SEGMENT_HEADING_PATTERNS = [
    re.compile(r"segment\s+information", re.I),
    re.compile(r"operating\s+segment", re.I),
    re.compile(r"reportable\s+segment", re.I),
    re.compile(r"business\s+segment", re.I),
    re.compile(r"note\s+\d+\s*[:\-–]\s*segment", re.I),
]

# Metrics we recognize in column headers
_METRIC_ALIASES: dict[str, str] = {
    "revenue": "revenue",
    "net revenue": "revenue",
    "net revenues": "revenue",
    "net sales": "revenue",
    "sales": "revenue",
    "operating income": "operating_income",
    "operating profit": "operating_income",
    "income from operations": "operating_income",
    "operating earnings": "operating_income",
    "operating loss": "operating_income",
}

# Scale notations SEC filings use to indicate table units — detected from
# text surrounding the table since individual cell magnitudes are not a
# reliable signal (e.g. "167,045" in a "$ in millions" table means $167.045B).
_SCALE_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"in\s+billions", re.I), 1e9),
    (re.compile(r"\$\s*in\s+millions|in\s+millions|millions?,?\s*except", re.I), 1e6),
    (re.compile(r"\$\s*in\s+thousands|in\s+thousands|thousands?,?\s*except", re.I), 1e3),
]
_DEFAULT_SCALE = 1e6  # SEC 10-K segment disclosures for large filers are conventionally in millions

# Common segment names to validate rows against (partial matching)
_KNOWN_SEGMENT_HINTS = frozenset({
    "americas", "europe", "greater china", "china", "japan", "asia",
    "rest of asia", "consumer", "commercial", "enterprise", "cloud",
    "aws", "azure", "google cloud", "services", "products", "hardware",
    "software", "digital media", "digital marketing", "iphone", "ipad",
    "mac", "wearables", "accessories", "total", "consolidated",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ingest_segments(
    ticker: str,
    cik: str,
    period_end: str,
    html: str,
    source_url: str = "",
) -> dict:
    """
    Parse 10-K HTML for ASC 280 segment tables and store in segment_facts.
    `html` must already be fetched (avoids duplicate HTTP request).
    Returns {"facts_written": int}.
    """
    if not is_connected():
        return {"facts_written": 0, "status": "db_unavailable"}

    segment_section = _find_segment_section(html)
    if not segment_section:
        logger.info("[segment] no segment section found for CIK=%s period=%s", cik, period_end)
        return {"facts_written": 0, "status": "no_segment_section"}

    rows = _parse_segment_tables(segment_section)
    if not rows:
        logger.info("[segment] no segment rows parsed for CIK=%s period=%s", cik, period_end)
        return {"facts_written": 0, "status": "no_rows"}

    written = 0
    for row in rows:
        segment_name = row["segment_name"]
        metric = row["metric"]
        value = row["value"]
        section_label = row.get("section_label", "Segment Information")

        doc_id = f"{cik}__{period_end}__{segment_name}__{metric}"
        doc = {
            "_id": doc_id,
            "cik": cik,
            "ticker": ticker,
            "period_end": period_end,
            "segment_name": segment_name,
            "metric": metric,
            "value": value,
            "unit": "USD",
            "form": "10-K",
            "section": section_label,
            "edgar_url": source_url,
            "ingested_at": _now(),
        }
        await collections.segment_facts.update_one(
            {"_id": doc_id},
            {"$setOnInsert": doc},
            upsert=True,
        )
        written += 1

    logger.info("[segment] wrote %d segment facts for CIK=%s period=%s", written, cik, period_end)
    return {"facts_written": written, "status": "ok"}


async def lookup_segment_fact(claim) -> Optional[ValidationResult]:
    """
    Look up a segment-level claim against segment_facts.
    Returns None if DB unavailable, no data, or no matching segment+metric.
    """
    try:
        if not is_connected() or not claim.ticker:
            return None

        ticker = _normalize_ticker(claim.ticker)
        cik_doc = await collections.ticker_lookup.find_one({"_id": ticker})
        if not cik_doc:
            return None
        cik = cik_doc["cik"]

        count = await collections.segment_facts.count_documents({"cik": cik}, limit=1)
        if count == 0:
            logger.info("  [segment] no data for CIK=%s", cik)
            return None

        # Build query — filter by period if provided
        query: dict = {"cik": cik}
        if claim.period:
            year = re.search(r"20\d{2}", claim.period)
            if year:
                query["period_end"] = {"$regex": year.group()}

        metric_lower = (claim.metric or "").lower()
        canonical_metric = _canonicalize_metric(metric_lower)

        # Find all segment docs and rank by combined segment+metric similarity
        best_doc = None
        best_score = 0.0

        async for doc in collections.segment_facts.find(query):
            seg_score = _segment_similarity(metric_lower, doc.get("segment_name", ""))
            met_score = 1.0 if (canonical_metric and doc.get("metric") == canonical_metric) else 0.3
            score = seg_score * 0.6 + met_score * 0.4
            if score > best_score:
                best_score = score
                best_doc = doc

        if best_doc is None or best_score < 0.35:
            logger.info("  [segment] no match for metric=%r (best_score=%.2f)", claim.metric, best_score)
            return None

        actual_val = best_doc["value"]
        stated = _parse_value(claim.value or "")
        if stated is None:
            return None

        status, confidence, discrepancy = _compare(claim.value, actual_val)
        formatted = _format_value(actual_val)
        segment_name = best_doc["segment_name"]
        metric = best_doc["metric"]
        period_end = best_doc["period_end"]
        section = best_doc.get("section", "Segment Information")
        edgar_url = best_doc.get("edgar_url", "")
        cik_bare = cik.lstrip("0")
        xbrl_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

        result = ValidationResult(claim_id=claim.id)
        result.status = status
        result.confidence = confidence
        result.actual_value = formatted
        result.discrepancy = discrepancy
        result.filing_source = f"10-K {section} — {segment_name} {metric}, period ending {period_end}"
        result.cik = cik
        result.edgar_url = edgar_url or None
        result.citations = [c for c in [edgar_url, xbrl_url] if c]
        result.structured_citations = [
            Citation(
                source="SEGMENT",
                label=f"10-K {section}",
                url=edgar_url or None,
                ticker=ticker,
                filing=f"10-K {period_end}",
                field=f"{segment_name} — {metric}",
                value=formatted,
                section=section,
                period=period_end,
            )
        ]
        result.reasoning = (
            f"Claimed: {claim.value} | 10-K segment ({segment_name} / {metric}): "
            f"{formatted} ({discrepancy})"
        )
        logger.info(
            "  [segment] HIT segment=%r metric=%r value=%s status=%s",
            segment_name, metric, formatted, status.value,
        )
        return result

    except Exception as exc:
        logger.debug("[segment] lookup_segment_fact skipped (%s)", exc)
        return None


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

def _find_segment_section(html: str) -> Optional[str]:
    """
    Return the HTML substring that contains the segment footnote/section.
    Searches for a heading that matches segment patterns, then captures
    text until the next major heading.
    """
    soup = BeautifulSoup(html, "lxml")

    # Strategy 1: find heading tags (h1-h4, strong, b) matching segment patterns
    heading_tags = soup.find_all(["h1", "h2", "h3", "h4", "strong", "b", "p"])
    for tag in heading_tags:
        if not isinstance(tag, Tag):
            continue
        text = tag.get_text(" ", strip=True)
        if any(p.search(text) for p in _SEGMENT_HEADING_PATTERNS):
            # Capture this tag and siblings until next major heading
            section_parts = [str(tag)]
            for sib in tag.find_next_siblings():
                sib_text = sib.get_text(" ", strip=True) if isinstance(sib, Tag) else ""
                # Stop at the next major section heading
                if isinstance(sib, Tag) and sib.name in ("h1", "h2", "h3"):
                    break
                section_parts.append(str(sib))
                if len("".join(section_parts)) > 80_000:
                    break
            if len(section_parts) > 1:
                return "".join(section_parts)

    # Strategy 2: regex scan on full text (plain HTML)
    for pattern in _SEGMENT_HEADING_PATTERNS:
        m = pattern.search(html)
        if m:
            start = max(0, m.start() - 200)
            end = min(len(html), m.start() + 80_000)
            return html[start:end]

    return None


def _parse_segment_tables(html_fragment: str) -> list[dict]:
    """
    Parse all tables in the segment HTML fragment.
    Returns list of {segment_name, metric, value, section_label}.
    """
    soup = BeautifulSoup(html_fragment, "lxml")
    results: list[dict] = []

    section_label = "Segment Information"
    _HEADING_TAGS = ("h1", "h2", "h3", "h4", "strong", "b", "p")
    heading = soup.find(
        lambda t: isinstance(t, Tag) and t.name in _HEADING_TAGS
        and any(p.search(t.get_text(" ", strip=True)) for p in _SEGMENT_HEADING_PATTERNS)
    )
    if heading:
        section_label = heading.get_text(" ", strip=True)[:80]

    for table in soup.find_all("table"):
        if not isinstance(table, Tag):
            continue
        parsed = _parse_segment_table(table, section_label)
        results.extend(parsed)

    return results


def _detect_scale(table: Tag) -> float:
    """
    Determine the dollar scale ($ in millions/thousands/billions) for a table
    by inspecting its own text and up to ~3 preceding sibling elements.
    Defaults to millions — the SEC 10-K convention for large-filer segment notes.
    """
    texts = [table.get_text(" ", strip=True)[:300]]
    sib = table
    for _ in range(3):
        sib = sib.find_previous_sibling()
        if sib is None or not isinstance(sib, Tag):
            break
        texts.append(sib.get_text(" ", strip=True)[:300])

    context = " ".join(texts)
    for pattern, multiplier in _SCALE_PATTERNS:
        if pattern.search(context):
            return multiplier
    return _DEFAULT_SCALE


def _parse_segment_table(table: Tag, section_label: str) -> list[dict]:
    """
    Parse a single table looking for segment data.
    Handles two orientations:
      - Rows = segments, columns = metrics (most common in 10-Ks)
      - Rows = metrics, columns = segments (less common)
    Cell values are scaled to raw dollars using the table's detected unit notation.
    """
    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    scale = _detect_scale(table)

    # Extract all cell texts as a 2D list
    grid: list[list[str]] = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        grid.append([c.get_text(" ", strip=True).strip() for c in cells])

    if not grid:
        return []

    # Try to identify header row (first row with mostly non-numeric cells)
    header_row = grid[0]
    # Detect orientation: segments-as-rows vs segments-as-columns
    # Heuristic: if a column header is a known segment name, segments are in columns
    col_headers = [h.lower() for h in header_row[1:]]

    segments_in_cols = any(
        any(seg in h for seg in _KNOWN_SEGMENT_HINTS)
        for h in col_headers
    )
    metric_in_col0 = _canonicalize_metric(header_row[0].lower()) is not None if header_row else False

    results: list[dict] = []

    if segments_in_cols and metric_in_col0:
        # Orientation: col 0 = metric, cols 1+ = segments
        for row in grid[1:]:
            if not row:
                continue
            metric_label = row[0].lower()
            canonical = _canonicalize_metric(metric_label)
            if not canonical:
                continue
            for i, seg_header in enumerate(col_headers, start=1):
                if i >= len(row):
                    break
                seg_name = header_row[i]
                val = _parse_cell_value(row[i])
                if val is not None and seg_name:
                    results.append({
                        "segment_name": seg_name,
                        "metric": canonical,
                        "value": val * scale,
                        "section_label": section_label,
                    })
    else:
        # Default orientation: rows = segments, columns = metrics
        # First, identify which column indices map to which metric
        metric_cols: dict[int, str] = {}
        for i, h in enumerate(col_headers, start=1):
            canonical = _canonicalize_metric(h.lower())
            if canonical:
                metric_cols[i] = canonical

        if not metric_cols:
            return []

        for row in grid[1:]:
            if not row:
                continue
            seg_name = row[0].strip()
            if not seg_name or len(seg_name) < 2:
                continue
            # Check if this row looks like a segment row
            has_value = any(
                _parse_cell_value(row[i]) is not None
                for i in metric_cols
                if i < len(row)
            )
            if not has_value:
                continue
            for col_idx, metric in metric_cols.items():
                if col_idx >= len(row):
                    continue
                val = _parse_cell_value(row[col_idx])
                if val is not None:
                    results.append({
                        "segment_name": seg_name,
                        "metric": metric,
                        "value": val * scale,
                        "section_label": section_label,
                    })

    return results


def _parse_cell_value(text: str) -> Optional[float]:
    """
    Parse a raw numeric table cell — handles commas and parenthesized negatives.
    Returns the bare number; the caller applies the table's declared $ scale
    (see _detect_scale) since segment tables state units once, not per-cell.
    """
    text = text.strip()
    if not text or text in ("—", "-", "–", "N/A", "n/a", ""):
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    text = text.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        val = float(text)
        return -val if negative else val
    except ValueError:
        return None


def _canonicalize_metric(text: str) -> Optional[str]:
    """Map a column header text to a canonical metric key."""
    text = text.lower().strip()
    for alias, canonical in _METRIC_ALIASES.items():
        if alias in text:
            return canonical
    return None


def _segment_similarity(claim_metric: str, segment_name: str) -> float:
    """Score how well a claim metric description matches a stored segment name."""
    cm = claim_metric.lower()
    sn = segment_name.lower()
    if sn in cm or cm in sn:
        return 1.0
    cm_words = set(re.split(r"\W+", cm)) - {"", "revenue", "sales", "income", "profit", "loss", "segment"}
    sn_words = set(re.split(r"\W+", sn)) - {"", "segment", "total"}
    if not cm_words or not sn_words:
        return 0.0
    overlap = len(cm_words & sn_words) / len(cm_words | sn_words)
    return overlap


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
