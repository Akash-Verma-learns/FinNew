from __future__ import annotations

import logging
import re
from typing import Optional

import httpx

from .models import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov"
TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
HEADERS = {"User-Agent": "FinValidator research@finvalidator.com"}

_ticker_cache: dict[str, str] = {}

CONCEPT_MAP: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "gross profit": ["GrossProfit"],
    "operating income": ["OperatingIncomeLoss"],
    "net income": ["NetIncomeLoss"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "total debt": ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "total assets": ["Assets"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "operating cash flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capital expenditures": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "shares outstanding": ["CommonStockSharesOutstanding"],
    "cost of goods sold": ["CostOfGoodsSoldAndServicesSold", "CostOfRevenue"],
}

MARGIN_CONCEPTS: dict[str, tuple[str, str]] = {
    "gross margin": ("gross profit", "revenue"),
    "operating margin": ("operating income", "revenue"),
    "net margin": ("net income", "revenue"),
}


# Exchange prefixes that are definitively non-US — skip EDGAR for these
_NON_US_EXCHANGES = frozenset({
    "WSE", "GPW",              # Warsaw Stock Exchange (Poland)
    "LSE", "LON",              # London Stock Exchange
    "TSX", "TSE",              # Toronto Stock Exchange
    "ASX",                     # Australian Securities Exchange
    "HKG", "HKEX",            # Hong Kong
    "TYO", "TKS",              # Tokyo
    "SHE", "SHG",              # Shenzhen / Shanghai
    "NSE", "BSE",              # India
    "KRX",                     # Korea
    "EURONEXT", "EPA", "AMS",  # Euronext
    "STO", "OSL", "CPH",       # Nordics
    "FRA", "ETR",              # Frankfurt
    "BIT",                     # Milan
    "JSE",                     # Johannesburg
})


def _is_non_us_ticker(raw: str) -> bool:
    """Return True if the ticker has a non-US exchange prefix (e.g. WSE:DOM)."""
    upper = raw.strip().upper()
    if ":" in upper:
        prefix = upper.split(":", 1)[0]
        return prefix in _NON_US_EXCHANGES
    return False


def _normalize_ticker(raw: str) -> str:
    """Normalize ticker variants to the format SEC EDGAR uses.

    Handles:
      BRK.A / BRK/A   → BRK-A   (share class dot/slash notation)
      NYSE:AAPL        → AAPL    (exchange prefix)
      AAPL.US / AAPL-US → AAPL  (country suffix)
      $AAPL            → AAPL   (social-media dollar prefix)
      aapl             → AAPL   (lowercase)
    """
    t = raw.strip().upper()
    t = t.lstrip("$")
    # Strip exchange prefix (NYSE:, NASDAQ:, LON:, etc.)
    if ":" in t:
        t = t.split(":", 1)[1]
    # Strip country/exchange suffix (.US, -US, .L, .T, etc.)
    t = re.sub(r"\.(US|UK|L|T|HK|TO|AX)$", "", t)
    t = re.sub(r"-(US|UK)$", "", t)
    # Convert dot/slash class notation to SEC hyphen form (BRK.A → BRK-A)
    t = re.sub(r"[./]([A-Z])$", r"-\1", t)
    return t


async def get_cik(ticker: str) -> Optional[str]:
    global _ticker_cache
    ticker = _normalize_ticker(ticker)
    if ticker in _ticker_cache:
        return _ticker_cache[ticker]
    async with httpx.AsyncClient(headers=HEADERS) as client:
        resp = await client.get(TICKER_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    for entry in data.values():
        t = entry.get("ticker", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        _ticker_cache[t] = cik
    return _ticker_cache.get(ticker)


async def get_company_facts(cik: str) -> dict:
    url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    async with httpx.AsyncClient(headers=HEADERS) as client:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()


def find_concepts(metric: str) -> list[str]:
    metric_lower = metric.lower().strip()
    # Exact match first
    if metric_lower in CONCEPT_MAP:
        return CONCEPT_MAP[metric_lower]
    # Only match if the full key is contained in the metric (not the reverse).
    # This prevents "revenue" from matching "services revenue" or "segment revenue".
    for key, concepts in CONCEPT_MAP.items():
        if metric_lower in key:
            return concepts
    return []


def _parse_value(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"([\d.]+)\s*(trillion|billion|million|T|B|M)?", s, re.IGNORECASE)
    if not m:
        return None
    num = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    multipliers = {
        "trillion": 1e12, "t": 1e12,
        "billion": 1e9, "b": 1e9,
        "million": 1e6, "m": 1e6,
    }
    return num * multipliers.get(suffix, 1)


def _format_value(v: float) -> str:
    if abs(v) >= 1e12:
        return f"${v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.0f}"


def _compare(stated_str: str, actual_raw: float) -> tuple[ValidationStatus, float, str]:
    stated = _parse_value(stated_str)
    if stated is None or actual_raw == 0:
        return ValidationStatus.UNVERIFIABLE, 0.5, "Could not parse stated value"
    pct = abs(stated - actual_raw) / abs(actual_raw)
    if pct < 0.005:
        return ValidationStatus.VERIFIED, 0.99, f"{pct * 100:.2f}% difference"
    if pct < 0.02:
        return ValidationStatus.VERIFIED, 0.95, f"{pct * 100:.2f}% difference"
    if pct < 0.05:
        return ValidationStatus.PARTIALLY_VERIFIED, 0.80, f"{pct * 100:.1f}% difference"
    if pct < 0.15:
        return ValidationStatus.PARTIALLY_VERIFIED, 0.60, f"{pct * 100:.1f}% difference"
    return ValidationStatus.CONTRADICTED, 0.95, f"{pct * 100:.1f}% difference"


def _find_period_value(
    units: dict, period: Optional[str]
) -> Optional[tuple[float, str, str, str]]:
    """Return (value, end_date, accession_number, filed_date) or None."""
    entries: list[dict] = []
    for unit_key in ("USD", "USD/shares", "shares"):
        entries = units.get(unit_key, [])
        if entries:
            break
    if not entries:
        entries = next(iter(units.values()), [])

    annual = [
        e for e in entries
        if e.get("form") in ("10-K", "20-F", "10-K/A", "20-F/A") and len(e.get("end", "")) == 10
    ]
    if not annual:
        return None
    annual.sort(key=lambda e: e["end"], reverse=True)

    def _pack(e: dict) -> tuple[float, str, str, str]:
        return e["val"], e["end"], e.get("accn", ""), e.get("filed", "")

    if period:
        year = re.search(r"20\d{2}", period)
        if year:
            target = year.group()
            for e in annual:
                if target in e.get("end", ""):
                    return _pack(e)
    return _pack(annual[0])


async def _lookup_from_db(claim) -> Optional["ValidationResult"]:
    """
    Try to satisfy the claim from the local MongoDB xbrl_facts collection.
    Returns a populated ValidationResult if found, else None.
    """
    try:
        from .db import collections, is_connected
        if not is_connected():
            return None

        cik_doc = await collections.ticker_lookup.find_one(
            {"_id": _normalize_ticker(claim.ticker or "")}
        )
        if not cik_doc:
            logger.info("  [xbrl-db] ticker=%s not in ticker_lookup — will hit EDGAR live", claim.ticker)
            return None
        cik = cik_doc["cik"]

        concepts = find_concepts(claim.metric or "")
        if not concepts:
            logger.info("  [xbrl-db] no CONCEPT_MAP entry for metric=%r", claim.metric)
            return None

        query: dict = {"cik": cik, "xbrl_concept": {"$in": concepts}}
        if claim.period:
            import re as _re
            year = _re.search(r"20\d{2}", claim.period)
            if year:
                query["period_end"] = {"$regex": year.group()}

        logger.info("  [xbrl-db] querying xbrl_facts: ticker=%s concepts=%s", claim.ticker, concepts)
        doc = await collections.xbrl_facts.find_one(
            query,
            sort=[("period_end", -1)],
        )
        if not doc:
            logger.info("  [xbrl-db] MISS — no cached fact for ticker=%s metric=%r", claim.ticker, claim.metric)
            return None

        actual_val = doc["value"]
        actual_period = doc["period_end"]
        accn = doc.get("accession_number") or ""
        filed = doc.get("filing_date") or ""
        edgar_url = doc.get("edgar_url")
        concept = doc["xbrl_concept"]

        logger.info("  [xbrl-db] HIT: concept=%s value=%s period=%s filed=%s url=%s",
                    concept, _format_value(actual_val), actual_period, filed, edgar_url)

        status, confidence, discrepancy = _compare(claim.value or "", actual_val)
        result = ValidationResult(claim_id=claim.id)
        result.status = status
        result.confidence = confidence
        result.actual_value = _format_value(actual_val)
        result.discrepancy = discrepancy
        result.filing_source = f"MongoDB cache (EDGAR XBRL) — {concept}, period ending {actual_period}"
        result.cik = cik
        result.accession_number = accn or None
        result.filing_date = filed or None
        result.edgar_url = edgar_url
        result.citations = [c for c in [edgar_url, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"] if c]
        result.reasoning = f"Claimed: {claim.value} | SEC EDGAR (cached): {_format_value(actual_val)} ({discrepancy})"
        return result
    except Exception as exc:
        logger.debug("DB lookup skipped (%s) — falling back to live EDGAR", exc)
        return None


async def lookup_direct_fact(claim) -> ValidationResult:
    result = ValidationResult(claim_id=claim.id)
    try:
        # Non-US exchange tickers (WSE:DOM, LSE:VOD, etc.) are never in SEC EDGAR
        if claim.ticker and _is_non_us_ticker(claim.ticker):
            logger.info("  [xbrl-live] skipping EDGAR — non-US ticker %r", claim.ticker)
            result.reasoning = f"Non-US exchange ticker {claim.ticker} — not in SEC EDGAR"
            return result

        # DB-first: check MongoDB cache before hitting EDGAR
        cached = await _lookup_from_db(claim)
        if cached is not None:
            return cached

        cik = await get_cik(claim.ticker)
        if not cik:
            result.reasoning = f"Ticker {claim.ticker} not found in SEC EDGAR"
            return result

        url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        logger.info("  [xbrl-live] fetching EDGAR company facts: %s", url)
        facts = await get_company_facts(cik)
        gaap = facts.get("facts", {}).get("us-gaap", {})
        concepts = find_concepts(claim.metric or "")
        if not concepts:
            result.reasoning = f"No XBRL concept mapped for metric: {claim.metric}"
            logger.info("  [xbrl-live] no CONCEPT_MAP entry for metric=%r", claim.metric)
            return result

        logger.info("  [xbrl-live] trying concepts %s for metric=%r", concepts, claim.metric)
        for concept in concepts:
            if concept not in gaap:
                logger.info("  [xbrl-live] concept=%s not in us-gaap — skipping", concept)
                continue
            units = gaap[concept].get("units", {})
            found = _find_period_value(units, claim.period)
            if not found:
                logger.info("  [xbrl-live] concept=%s found but no annual entry for period=%r", concept, claim.period)
                continue
            actual_val, actual_period, accn, filed = found
            cik_bare = cik.lstrip("0")
            accn_clean = accn.replace("-", "")
            edgar_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/{accn_clean}/"
                if accn_clean else None
            )
            logger.info("  [xbrl-live] HIT: concept=%s value=%s period=%s accn=%s url=%s",
                        concept, _format_value(actual_val), actual_period, accn, edgar_url)
            status, confidence, discrepancy = _compare(claim.value or "", actual_val)
            result.status = status
            result.confidence = confidence
            result.actual_value = _format_value(actual_val)
            result.discrepancy = discrepancy
            result.filing_source = f"SEC EDGAR XBRL — {concept}, period ending {actual_period}"
            result.cik = cik
            result.accession_number = accn or None
            result.filing_date = filed or None
            result.edgar_url = edgar_url
            result.citations = [
                c for c in [
                    edgar_url,
                    f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                ] if c
            ]
            result.reasoning = (
                f"Claimed: {claim.value} | SEC EDGAR: {_format_value(actual_val)} ({discrepancy})"
            )
            return result

        result.reasoning = f"No XBRL data found for {claim.metric} ({', '.join(concepts)})"
        logger.info("  [xbrl-live] MISS — none of %s had annual data for period=%r", concepts, claim.period)
    except Exception as exc:
        logger.error("lookup_direct_fact error: %s", exc)
        result.status = ValidationStatus.ERROR
        result.reasoning = str(exc)
    return result


async def lookup_derived_ratio(claim) -> ValidationResult:
    result = ValidationResult(claim_id=claim.id)
    if claim.ticker and _is_non_us_ticker(claim.ticker):
        logger.info("  [live-ratio] skipping EDGAR — non-US ticker %r", claim.ticker)
        result.reasoning = f"Non-US exchange ticker {claim.ticker} — not in SEC EDGAR"
        return result
    metric_lower = (claim.metric or "").lower()
    if metric_lower not in MARGIN_CONCEPTS:
        logger.info("  [live-ratio] metric=%r not in MARGIN_CONCEPTS — skipping", claim.metric)
        result.reasoning = f"Unsupported derived metric: {claim.metric}"
        return result

    numerator_key, denominator_key = MARGIN_CONCEPTS[metric_lower]
    try:
        cik = await get_cik(claim.ticker)
        if not cik:
            result.reasoning = f"Ticker {claim.ticker} not found"
            return result

        logger.info("  [live-ratio] fetching EDGAR facts for cik=%s to compute %s/%s",
                    cik, numerator_key, denominator_key)
        facts = await get_company_facts(cik)
        gaap = facts.get("facts", {}).get("us-gaap", {})

        def _fetch(key: str) -> Optional[float]:
            for concept in CONCEPT_MAP.get(key, []):
                if concept in gaap:
                    units = gaap[concept].get("units", {})
                    found = _find_period_value(units, claim.period)
                    if found:
                        return found[0]
            return None

        num = _fetch(numerator_key)
        den = _fetch(denominator_key)
        logger.info("  [live-ratio] %s=%s  %s=%s", numerator_key, num, denominator_key, den)
        if num is None or den is None or den == 0:
            result.reasoning = "Could not fetch both numerator and denominator from XBRL"
            return result

        actual_pct = (num / den) * 100
        stated_pct = _parse_value(claim.value or "")
        if stated_pct is None:
            result.reasoning = "Could not parse stated margin value"
            return result

        diff = abs(actual_pct - stated_pct)
        if diff < 0.3:
            result.status, result.confidence = ValidationStatus.VERIFIED, 0.97
        elif diff < 1.0:
            result.status, result.confidence = ValidationStatus.VERIFIED, 0.90
        elif diff < 2.0:
            result.status, result.confidence = ValidationStatus.PARTIALLY_VERIFIED, 0.75
        else:
            result.status, result.confidence = ValidationStatus.CONTRADICTED, 0.90

        result.actual_value = f"{actual_pct:.2f}%"
        result.discrepancy = f"{diff:.2f}pp difference"
        result.reasoning = (
            f"Claimed: {claim.value} | Computed: {actual_pct:.2f}% "
            f"({num / 1e9:.2f}B / {den / 1e9:.2f}B)"
        )
        logger.info("  [live-ratio] computed %s=%.2f%% diff=%.2fpp → %s",
                    claim.metric, actual_pct, diff, result.status.value)
    except Exception as exc:
        logger.error("lookup_derived_ratio error: %s", exc)
        result.status = ValidationStatus.ERROR
        result.reasoning = str(exc)
    return result


# Keywords that restrict which XBRL concept names we search during value-matching.
# Keeps false-positive matches low (e.g. $96B debt won't match a "services revenue" claim).
_METRIC_CONCEPT_KEYWORDS: dict[str, list[str]] = {
    "revenue":   ["revenue", "sales"],
    "income":    ["income", "profit", "loss", "earning"],
    "cash":      ["cash"],
    "debt":      ["debt", "borrow", "liabilit", "note"],
    "asset":     ["asset"],
    "equity":    ["equity", "stockholder"],
    "expense":   ["expense", "cost"],
    "cashflow":  ["cashprovided", "cashused", "operating"],
}


def _concept_keywords_for_metric(metric: str) -> list[str]:
    """Return lowercase substrings to filter concept names when value-matching."""
    ml = metric.lower()
    for key, kws in _METRIC_CONCEPT_KEYWORDS.items():
        if key in ml:
            return kws
    # Default: use the first significant word of the metric
    first = ml.split()[0] if ml.split() else ""
    return [first] if first else []


async def lookup_xbrl_value_match(claim) -> ValidationResult:
    """
    Scan ALL XBRL concepts (us-gaap + company extensions) for an entry whose
    value is within 1.5% of the stated value, in the right period and form type.

    Used for segment metrics (services revenue, product-line revenue, etc.) that
    are tagged dimensionally or under a company-specific extension namespace and
    therefore missed by CONCEPT_MAP direct lookup.
    """
    result = ValidationResult(claim_id=claim.id)
    if not claim.ticker or not claim.value:
        return result
    if _is_non_us_ticker(claim.ticker):
        logger.info("  [value-match] skipping EDGAR — non-US ticker %r", claim.ticker)
        return result

    stated = _parse_value(claim.value)
    if stated is None or stated == 0:
        return result

    kws = _concept_keywords_for_metric(claim.metric or "")

    year_filter: Optional[str] = None
    if claim.period:
        m = re.search(r"20\d{2}", claim.period)
        if m:
            year_filter = m.group()

    try:
        cik = await get_cik(claim.ticker)
        if not cik:
            return result

        facts_json = await get_company_facts(cik)

        best_pct: float = float("inf")
        best: Optional[tuple] = None  # (concept, namespace, entry, actual_val)

        for namespace, ns_data in facts_json.get("facts", {}).items():
            for concept, concept_data in ns_data.items():
                # Filter by concept name keywords to reduce false positives
                if kws and not any(kw in concept.lower() for kw in kws):
                    continue
                for unit_key, entries in concept_data.get("units", {}).items():
                    if unit_key not in ("USD", "USD/shares"):
                        continue
                    for entry in entries:
                        if entry.get("form") not in ("10-K", "20-F", "10-K/A", "20-F/A"):
                            continue
                        if len(entry.get("end", "")) != 10:
                            continue
                        if year_filter and year_filter not in entry["end"]:
                            continue
                        val = entry.get("val")
                        if val is None or val == 0:
                            continue
                        pct = abs(val - stated) / abs(stated)
                        if pct < 0.015 and pct < best_pct:
                            best_pct = pct
                            best = (concept, namespace, entry, val)

        if best is None:
            return result

        concept, namespace, entry, actual_val = best
        status, confidence, discrepancy = _compare(claim.value, actual_val)
        cik_bare = cik.lstrip("0")
        accn = entry.get("accn", "")
        accn_clean = accn.replace("-", "")
        edgar_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/{accn_clean}/"
            if accn_clean else None
        )
        result.status = status
        result.confidence = confidence
        result.actual_value = _format_value(actual_val)
        result.discrepancy = discrepancy
        result.filing_source = (
            f"SEC EDGAR XBRL ({namespace}) — {concept}, period ending {entry['end']}"
        )
        result.cik = cik
        result.accession_number = accn or None
        result.filing_date = entry.get("filed") or None
        result.edgar_url = edgar_url
        result.citations = [
            c for c in [edgar_url, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"]
            if c
        ]
        result.reasoning = (
            f"Claimed: {claim.value} | EDGAR XBRL value match ({concept}): "
            f"{_format_value(actual_val)} ({discrepancy})"
        )
        logger.info(
            "Value match: %s claim=%s actual=%s pct=%.3f%% concept=%s",
            claim.ticker, claim.value, _format_value(actual_val), best_pct * 100, concept,
        )

    except Exception as exc:
        logger.debug("lookup_xbrl_value_match error: %s", exc)

    return result


async def get_latest_10k(ticker: str) -> Optional[dict]:
    if _is_non_us_ticker(ticker):
        return None
    try:
        cik = await get_cik(ticker)
        if not cik:
            return None
        url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        async with httpx.AsyncClient(headers=HEADERS) as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accns = filings.get("accessionNumber", [])
        for form, date, accn in zip(forms, dates, accns):
            if form == "10-K":
                accn_clean = accn.replace("-", "")
                cik_bare = cik.lstrip("0")
                return {
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/{accn_clean}/",
                    "date": date,
                }
    except Exception as exc:
        logger.warning("get_latest_10k error: %s", exc)
    return None
