from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

import httpx

from .models import Citation, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov"
TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
HEADERS = {"User-Agent": "FinValidator research@finvalidator.com"}

_ticker_cache: dict[str, str] = {}
# Single-flight locks: with hundreds of claims now reaching the local pipeline
# (post ticker-backfill fix), they all resolve the SAME ticker/CIK concurrently.
# Without these, every one of them races to fetch the (large) ticker list /
# company-facts JSON before the cache is warm — SEC EDGAR answers the herd with
# 429 Too Many Requests, those lookups raise, and claims that should cleanly
# resolve in step 1 (xbrl-direct) instead miss and cascade through every
# subsequent step. The lock makes the first caller fetch-and-cache while
# everyone else waits and then reads the cache — one fetch per ticker/CIK
# for the lifetime of the process, not one per claim.
_ticker_cache_lock = asyncio.Lock()
_company_facts_cache: dict[str, dict] = {}
_company_facts_locks: dict[str, asyncio.Lock] = {}
_company_facts_locks_guard = asyncio.Lock()

# Semantic concept index: per-CIK embedding matrix built once from EDGAR concept
# labels. Lets us match any metric phrase by vector similarity rather than keyword
# lookup — "provision for income taxes" finds IncomeTaxExpenseBenefit even though
# no keyword matches. Built lazily on first lookup for a ticker; single-flight so
# concurrent claims don't each trigger the (CPU-bound) encode pass.
_concept_index_cache: dict[str, tuple[list[str], Any]] = {}  # cik → (concepts, matrix)
_concept_index_locks: dict[str, asyncio.Lock] = {}
_concept_index_locks_guard = asyncio.Lock()

CONCEPT_MAP: dict[str, list[str]] = {
    # --- Income statement ---
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
    "operating expenses": ["OperatingExpenses"],
    "cost of goods sold": ["CostOfGoodsSoldAndServicesSold", "CostOfRevenue"],
    # Exact-key entries for phrases that contain a disqualifier word but ARE
    # themselves valid standalone metrics (exact match fires before disqualifier check)
    "cost of revenue": ["CostOfRevenue", "CostOfGoodsSoldAndServicesSold"],
    "research and development": ["ResearchAndDevelopmentExpense"],
    "sales and marketing": ["SellingAndMarketingExpense", "SellingGeneralAndAdministrativeExpense"],
    "general and administrative": ["GeneralAndAdministrativeExpense"],
    "income before taxes": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"
    ],
    "income tax": ["IncomeTaxExpenseBenefit"],
    "interest expense": ["InterestExpenseNonoperating", "InterestExpense"],
    "other income": ["NonoperatingIncomeExpense", "OtherNonoperatingIncomeExpense"],
    "depreciation and amortization": [
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
    ],
    "stock-based compensation": [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
    ],
    "dividends": ["PaymentsOfDividends"],
    # --- Balance sheet ---
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    # Full combined line item — must be longer than "short-term investments" so it
    # wins the longest-first key race when the metric contains both phrases.
    # Both the Oxford-comma and non-Oxford-comma forms are included because the
    # LLM extractor normalises PDF kerning artifacts differently across runs —
    # "Cash , cash equivalents , and short -term investments" can come back with
    # or without the comma before "and".  Without this alias the exact-match
    # fails, the long-key regex also fails (it requires the comma), and the
    # fallback "short-term investments" key matches instead → ShortTermInvestments
    # (~$64B) vs the claimed combined $94.6B → confident false CONTRADICTED.
    "cash, cash equivalents, and short-term investments": [
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "cash, cash equivalents and short-term investments": [
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "short-term investments": ["ShortTermInvestments"],
    "accounts receivable": ["AccountsReceivableNetCurrent"],
    "current assets": ["AssetsCurrent"],
    "total assets": ["Assets"],
    "goodwill": ["Goodwill"],
    "intangible assets": [
        "IntangibleAssetsNetExcludingGoodwill",
        "FiniteLivedIntangibleAssetsNet",
    ],
    "property and equipment": ["PropertyPlantAndEquipmentNet"],
    "equity and other investments": ["LongTermInvestments"],
    "long-term investments": ["LongTermInvestments"],
    "current liabilities": ["LiabilitiesCurrent"],
    "total liabilities": ["Liabilities"],
    "total debt": ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "deferred revenue": ["DeferredRevenueCurrent", "DeferredRevenue"],
    "unearned revenue": ["DeferredRevenueCurrent", "DeferredRevenue"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    # --- Cash flow ---
    "operating cash flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capital expenditures": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    # --- Share data ---
    "shares outstanding": ["CommonStockSharesOutstanding"],
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
    async with _ticker_cache_lock:
        # Re-check: another coroutine may have populated the cache while we waited.
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
    if cik in _company_facts_cache:
        return _company_facts_cache[cik]
    async with _company_facts_locks_guard:
        lock = _company_facts_locks.setdefault(cik, asyncio.Lock())
    async with lock:
        # Re-check: another coroutine may have populated the cache while we waited.
        if cik in _company_facts_cache:
            return _company_facts_cache[cik]
        url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        async with httpx.AsyncClient(headers=HEADERS) as client:
            resp = await client.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        _company_facts_cache[cik] = data
    return data


# Modifier words/phrases that mean the metric describes a derived ratio, growth
# rate, sub-component, or different line item than the canonical CONCEPT_MAP
# concept the matched key represents — e.g. "unearned revenue" is a contract-
# liability concept (not "Revenues"); "Microsoft 365 ... revenue growth" is a
# YoY percentage (not a dollar figure comparable to total revenue); "cost of
# revenue" / "commercial portion of revenue" are different/narrower line items.
# Matching these to the generic key would produce a confidently-wrong
# CONTRADICTED rather than a safe Unverifiable, so reject the match outright.
_CONCEPT_KEY_DISQUALIFIERS = (
    "growth", "rate", "ratio", "margin", "%", "percent", "yoy", "y/y", "qoq",
    "change", "increase", "decrease", "decline",
    "unearned", "deferred", "cost of", "portion of",
    "per share", "per unit", "mix", "breakdown", "allocation",
    "attributable", "contribution", "subscribers", "seats", "users",
    "method investment", "in earnings of", "noncontrolling",
    # Prevents "goodwill impairment" matching the Goodwill balance-sheet concept —
    # GoodwillImpairmentLoss is a separate XBRL concept; comparing impairment
    # against the goodwill balance produces a confident false CONTRADICTED.
    "impairment",
    # "stock-based compensation tax benefits" is a footnote sub-component, not the
    # total IncomeTaxExpenseBenefit reported on the income statement.
    "stock-based", "stock based",
    # Product/segment revenue lines (iPhone, Services, Mac, iPad, Google Cloud,
    # Azure, etc.) must not be looked up via the generic "revenue" XBRL concept —
    # the match would compare a product segment against consolidated total revenue.
    "iphone", "ipad", "mac ", "services revenue", "cloud revenue",
    "azure", "google cloud", "gaming revenue", "advertising revenue",
    "search revenue", "network services", "youtube",
    # Quarterly-period labels: EDGAR stores annual/quarterly totals but a claim
    # labelled "quarterly revenue" should not be matched against annual 10-K revenue.
    "quarterly", "q1 ", "q2 ", "q3 ", "q4 ", " q1", " q2", " q3", " q4",
)


# Single-word / short aggregate CONCEPT_MAP keys that represent consolidated
# company-wide totals. When such a key appears inside a longer phrase with a
# qualifying noun before it (e.g. "cloud revenue", "LinkedIn revenue",
# "services revenue"), the phrase is about a SEGMENT or PRODUCT, not the
# consolidated figure the XBRL concept holds.  Matching and then comparing a
# $168.9B cloud-segment figure against ~$279B total revenue produces a high-
# confidence CONTRADICTED verdict that kills the entire report score.
#
# Keys listed here are guarded: they only match if the metric phrase is
# exactly that concept (optionally preceded by "total" or "net").
# Any qualifier before the key word → [] → falls through to label/embedding.
_AGGREGATE_KEY_PATTERNS: dict[str, re.Pattern] = {
    # Each pattern allows a narrow set of prefixes ("total", "net") but rejects any
    # leading qualifier noun ("Intelligent Cloud operating income", "equity contracts
    # purchased", "income tax benefits related to stock-based compensation").  Without
    # these guards the substring loop matches the short key inside a longer phrase and
    # then compares a segment/sub-component figure against the consolidated XBRL total,
    # producing a confident false CONTRADICTED that floors the report score.
    "revenue": re.compile(r"^(?:total\s+|net\s+)?revenue\s*$"),
    "operating income": re.compile(r"^(?:total\s+)?operating\s+income(?:\s+loss)?\s*$"),
    "net income": re.compile(r"^(?:total\s+)?net\s+income(?:\s+loss)?\s*$"),
    "gross profit": re.compile(r"^(?:total\s+)?gross\s+profit\s*$"),
    "operating expenses": re.compile(r"^(?:total\s+)?operating\s+expenses?\s*$"),
    "income tax": re.compile(r"^(?:total\s+|provision\s+for\s+)?income\s+tax(?:es)?\s*(?:expense|benefit)?\s*$"),
    "equity": re.compile(r"^(?:total\s+|net\s+|stockholders['\s]+)?equity\s*$"),
    "accounts receivable": re.compile(r"^(?:net\s+|trade\s+)?accounts\s+receivable(?:\s+net)?\s*$"),
    "goodwill": re.compile(r"^goodwill\s*$"),
    "interest expense": re.compile(r"^(?:net\s+)?interest\s+expense\s*$"),
}


def find_concepts(metric: str) -> list[str]:
    metric_lower = metric.lower().strip()
    # Normalize PDF kerning artifacts so "cash , cash equivalents , and
    # short -term investments" matches the same CONCEPT_MAP keys as the
    # clean form "cash, cash equivalents, and short-term investments".
    # 1. Collapse spaces around commas: " , " → ", "
    metric_lower = re.sub(r"\s*,\s*", ", ", metric_lower)
    # 2. Collapse spaces around hyphens between word chars (only one side has
    #    the extra space, as produced by PDF kerning): "short -term" → "short-term".
    #    Pattern requires a word char immediately before/after the hyphen so that
    #    "accounts payable - related parties" (both sides have space) is preserved.
    metric_lower = re.sub(r"(?<=\w)\s+-(?=\w)|(?<=\w)-\s+(?=\w)", "-", metric_lower)
    # Exact match first — also bypasses the aggregate-key guard so that entries
    # like "cost of revenue" that ARE in CONCEPT_MAP verbatim still resolve.
    if metric_lower in CONCEPT_MAP:
        return CONCEPT_MAP[metric_lower]
    # Disqualifiers: metric contains a word/phrase that means it describes a
    # ratio, growth rate, sub-component, or segment rather than a consolidated
    # balance-sheet / income-statement line item.
    if any(d in metric_lower for d in _CONCEPT_KEY_DISQUALIFIERS):
        return []
    # Substring match — longest key first so "equity and other investments"
    # wins over the generic "equity" key.
    for key in sorted(CONCEPT_MAP, key=len, reverse=True):
        if not re.search(r"\b" + re.escape(key) + r"\b", metric_lower):
            continue
        # Aggregate-key guard: single-word total-level concepts must not match
        # qualified phrases ("cloud revenue", "LinkedIn revenue", etc.) — those
        # are segment figures and will produce false CONTRADICTED verdicts when
        # compared against the consolidated XBRL total.
        guard = _AGGREGATE_KEY_PATTERNS.get(key)
        if guard and not guard.match(metric_lower):
            continue
        return CONCEPT_MAP[key]
    return []


def _parse_value(s: str) -> Optional[float]:
    if not s:
        return None
    # PDF kerning artifact: decimal points get spaces — "$94 . 6 billion" → "$94.6 billion".
    s = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", s)
    # PDF kerning artifact: thousand separators get spaces — "281 , 724" → "281724".
    # Must run BEFORE the comma strip: stripping "," from "281 , 724" leaves "281  724"
    # and the regex then captures only "281" (stops at the space), producing a value
    # that is 10^6× too small and causes every "(in millions)" table entry to be
    # confidently CONTRADICTED against the correct XBRL raw-dollar figure.
    s = re.sub(r"(\d)\s*,\s*(\d)", r"\1\2", s)
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
    if abs(v) >= 1:
        return f"${v:,.2f}"
    return f"${v:.4f}"


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
    # Units-scale guard: 10-K/20-F tables state values "(in millions)" or "(in thousands)"
    # without a unit suffix on each cell.  The LLM extractor copies the raw number (e.g.
    # "281,724") while EDGAR stores raw dollars (281,724,000,000).  A 10^6 ratio is not a
    # contradiction — it is a units mismatch.  Emitting CONTRADICTED here would floor the
    # entire report score to 0 via the cascade-penalty + HIGH-flag mechanism.
    # Use abs() on both sides so sign conventions (expense as negative vs positive) don't
    # prevent the match (e.g. "(4,901)" interest income → stated=4901 vs actual=-4901000000).
    if stated != 0:
        for scale in (1_000, 1_000_000, 1_000_000_000):
            scaled_pct = abs(abs(stated) * scale - abs(actual_raw)) / abs(actual_raw)
            if scaled_pct < 0.02:
                return ValidationStatus.VERIFIED, 0.90, f"Units-scaled match ({scale:,}×): {scaled_pct * 100:.2f}% at scale"
            if scaled_pct < 0.05:
                return ValidationStatus.PARTIALLY_VERIFIED, 0.75, f"Units-scaled match ({scale:,}×): {scaled_pct * 100:.1f}% at scale"
            if scaled_pct < 0.15:
                return ValidationStatus.PARTIALLY_VERIFIED, 0.60, f"Units-scaled match ({scale:,}×): {scaled_pct * 100:.1f}% at scale"
    return ValidationStatus.CONTRADICTED, 0.95, f"{pct * 100:.1f}% difference"


def _pick_best_candidate(candidates: list, claimed_value: Optional[str], key):
    """
    A single CONCEPT_MAP key can map to several distinct XBRL concepts that
    report genuinely different numbers — e.g. "cash" covers both
    CashAndCashEquivalentsAtCarryingValue (~$30B for MSFT FY25) and the much
    larger CashCashEquivalentsAndShortTermInvestments (~$94.6B). Blindly taking
    the first textually-plausible concept produces a confident-but-wrong
    CONTRADICTED when it isn't the figure the claim is actually quoting.

    When the claim states a parseable number, prefer whichever candidate
    `_compare` would score most favourably (smallest % difference) — i.e. the
    one that actually matches what was claimed — instead of list order.
    `key` extracts the numeric value from a candidate (tuple or dict).
    """
    if len(candidates) == 1:
        return candidates[0]
    stated = _parse_value(claimed_value or "")
    if stated is None:
        return candidates[0]

    def _pct(c):
        actual = key(c)
        if not actual:
            return float("inf")
        raw = abs(stated - actual) / abs(actual)
        # "(in millions)" tables state 2385 while EDGAR stores 2385000000.
        # Raw comparison makes FY2025 ($2,385M exact match) look identical to
        # FY2023 ($1,968M, 21% off) — both ~99.9% when unstated scale.
        # Try the same 1K/1M/1B scale normalization as _compare so the
        # candidate whose value actually matches wins.
        best = raw
        for scale in (1_000, 1_000_000, 1_000_000_000):
            scaled = abs(abs(stated) * scale - abs(actual)) / abs(actual)
            if scaled < best:
                best = scaled
        return best

    return min(candidates, key=_pct)


def _parse_quarterly_period(period: str) -> Optional[tuple[int, int]]:
    """
    Extract (quarter, fiscal_year) from a period string, or None if not quarterly.

    Handles:  "Q1 FY2024"  "Q2 2024"  "first quarter FY2023"  "Q3 fiscal 2025"
    The returned fiscal_year matches EDGAR's fy field, which uses the company's
    own fiscal-year numbering (Apple Q1 FY2024 → fy=2024 in EDGAR, even though
    the calendar dates are Oct–Dec 2023).
    """
    p = period.lower().strip()
    # "Q1 FY2024", "Q2 2024", "q3 fiscal 2024"
    m = re.search(r"\bq([1-4])\b.*?(20\d{2})", p)
    if m:
        return int(m.group(1)), int(m.group(2))
    # "first quarter 2024", "second quarter FY2023"
    _ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
    for word, num in _ORDINALS.items():
        if word in p:
            m2 = re.search(r"20\d{2}", p)
            if m2:
                return num, int(m2.group())
    return None


def _find_quarterly_value(
    units: dict, quarter: int, fiscal_year: int
) -> Optional[tuple[float, str, str, str]]:
    """
    Find a single-quarter XBRL value from 10-Q filings.

    Matches entries where fp == "Q{n}" AND fy == fiscal_year AND form == "10-Q".
    Returns (value, period_label, accn, filed_date) or None.

    EDGAR stores single-quarter flow values (revenue, net income, EPS) with
    fp="Q1"/"Q2"/"Q3" and the company's fiscal year.  Q4 is usually only in the
    annual 10-K, not a separate 10-Q, so we silently return None for Q4 rather
    than risk comparing against a full-year total.
    """
    entries: list[dict] = []
    for unit_key in ("USD", "USD/shares", "shares"):
        entries = units.get(unit_key, [])
        if entries:
            break
    if not entries:
        entries = next(iter(units.values()), [])

    fp_target = f"Q{quarter}"
    # Q4 is almost never filed as a separate 10-Q — skip rather than fabricate
    if quarter == 4:
        return None

    matches = [
        e for e in entries
        if e.get("form") == "10-Q"
        and e.get("fp") == fp_target
        and e.get("fy") == fiscal_year
    ]
    if not matches:
        return None

    # EDGAR attributes fy to the FILING, so a 10-Q contains two rows for the
    # same fp/fy: one for the current period and one for the prior-year
    # comparative (same fp="Q1", fy=2024 but end="2022-12-31" vs "2023-12-30").
    # Step 1: keep only the most recent end-date (the actual current quarter).
    latest_end = max(e.get("end", "") for e in matches)
    current = [e for e in matches if e.get("end", "") == latest_end]

    # Step 2: 10-Qs report BOTH standalone quarter values ("Three months ended")
    # AND year-to-date cumulative values ("Six months ended" for Q2, "Nine months
    # ended" for Q3).  Both carry fp="Q2"/fp="Q3" with the same end-date and
    # frame=(none), making them indistinguishable except by value magnitude.
    # The standalone quarter is ALWAYS smaller in absolute value than the
    # cumulative (standalone Q2=$90.75B vs 6-month cumulative=$210.33B).
    # Pick the entry with the smallest absolute value to get the standalone quarter.
    if len(current) > 1:
        current = [min(current, key=lambda e: abs(e["val"]))]

    e = current[0]
    period_label = f"Q{quarter} FY{fiscal_year} (ended {e.get('end', '')})"
    return e["val"], period_label, e.get("accn", ""), e.get("filed", "")


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


def _find_all_annual_values(
    units: dict, max_years: int = 5
) -> list[tuple[float, str, str, str]]:
    """Return (value, end_date, accn, filed) for up to max_years most recent annual entries.

    Used when the claim has no period hint so _pick_best_candidate can find the
    year whose XBRL value numerically matches the stated value.

    5-year window (rather than 3) covers two common cases that a 3-year window
    misses:
      1. Historical comparison columns in 10-K tables (e.g. FY2021 figures
         shown alongside FY2025 for trend analysis).
      2. Purchase-price-allocation notes: Activision had ~$51B goodwill and
         ~$13B cash at acquisition; MSFT's consolidated goodwill and cash were
         at similar levels in FY2020-FY2021.  Without the 4-5 year lookback,
         the validator compares against the most-recent consolidated total
         ($119B goodwill in FY2024) and emits a confident false CONTRADICTED.
    """
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
        return []
    # Sort by (end_date DESC, filed_date DESC) so that for duplicate end_dates
    # (e.g. both a 10-K and a 10-K/A for the same fiscal year), the most recently
    # filed version comes first and gets kept by the deduplication step below.
    annual.sort(key=lambda e: (e["end"], e.get("filed", "")), reverse=True)
    # Deduplicate by end_date: one entry per fiscal year period.
    # Without this, 10-K/10-K/A pairs produce two identical (value, period) rows,
    # and lookup_growth_rate comparing annual_vals[n] with annual_vals[n+1] would
    # compare the same fiscal year against itself → 0% growth.
    seen_ends: set[str] = set()
    deduped: list[dict] = []
    for e in annual:
        end = e.get("end", "")
        if end not in seen_ends:
            seen_ends.add(end)
            deduped.append(e)
    return [(e["val"], e["end"], e.get("accn", ""), e.get("filed", "")) for e in deduped[:max_years]]


def _build_xbrl_citation(
    cik: str,
    ticker: str,
    concept: str,
    period: str,
    formatted_value: str,
    accession: Optional[str],
    edgar_url: Optional[str],
    namespace: str = "us-gaap",
    concept_label: Optional[str] = None,
) -> Citation:
    year = period[:4] if period else ""
    # Cite the official XBRL taxonomy label — the exact line-item wording the
    # filer itself reports (e.g. "Cash, Cash Equivalents, and Short-term
    # Investments") — rather than the bare machine concept name
    # ("CashCashEquivalentsAndShortTermInvestments"), so the citation reads as
    # a verifiable quote from the filing instead of an opaque XBRL tag.
    wording = concept_label or concept
    return Citation(
        source="EDGAR_XBRL",
        label=f"SEC EDGAR XBRL ({namespace}) — {wording}",
        url=edgar_url,
        ticker=ticker,
        filing=f"10-K {year}",
        accession=accession,
        field=concept,
        value=formatted_value,
        excerpt=f'"{wording}": {formatted_value} (period ended {period})',
        period=period,
    )


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
            logger.info("  [xbrl-db] no CONCEPT_MAP entry for metric=%r — skipping DB (label-match runs live)", claim.metric)
            return None

        query: dict = {"cik": cik, "xbrl_concept": {"$in": concepts}}
        if claim.period:
            import re as _re
            year = _re.search(r"20\d{2}", claim.period)
            if year:
                query["period_end"] = {"$regex": year.group()}

        logger.info("  [xbrl-db] querying xbrl_facts: ticker=%s concepts=%s", claim.ticker, concepts)
        docs = [d async for d in collections.xbrl_facts.find(query, sort=[("period_end", -1)])]
        if not docs:
            logger.info("  [xbrl-db] MISS — no cached fact for ticker=%s metric=%r", claim.ticker, claim.metric)
            return None

        # A CONCEPT_MAP key can resolve to several distinct concepts that get
        # cached as separate xbrl_facts docs for the same period (e.g. "cash"
        # -> both CashAndCashEquivalentsAtCarryingValue and the larger
        # CashCashEquivalentsAndShortTermInvestments). Pick the most recent
        # period first (matches the old find_one(sort=period_end desc)
        # behaviour), then disambiguate by value within that period so we land
        # on the concept the claim is actually quoting, not just whichever
        # name sorted/inserted first.
        #
        # Exception: if the claim has no period hint (e.g. a bare table cell
        # "(2,935)" with no year context), search across ALL cached periods so
        # _pick_best_candidate can find the year whose value actually matches.
        # This fixes FY2024/FY2023 historical values being compared against the
        # latest-period cached entry and producing false CONTRADICTED verdicts.
        if claim.period and re.search(r"20\d{2}", claim.period):
            latest_period = docs[0]["period_end"]
            same_period = [d for d in docs if d["period_end"] == latest_period]
        else:
            same_period = docs
        doc = _pick_best_candidate(same_period, claim.value, key=lambda d: d["value"])

        actual_val = doc["value"]
        actual_period = doc["period_end"]
        accn = doc.get("accession_number") or ""
        filed = doc.get("filing_date") or ""
        edgar_url = doc.get("edgar_url")
        concept = doc["xbrl_concept"]

        logger.info("  [xbrl-db] HIT: concept=%s value=%s period=%s filed=%s url=%s (chosen from %d candidate(s) for period=%s: %s)",
                    concept, _format_value(actual_val), actual_period, filed, edgar_url,
                    len(same_period), actual_period, [d["xbrl_concept"] for d in same_period])

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
        result.structured_citations = [
            _build_xbrl_citation(
                cik, _normalize_ticker(claim.ticker or ""), concept, actual_period,
                _format_value(actual_val), accn or None, edgar_url,
                concept_label=doc.get("concept_label"),
            )
        ]
        result.reasoning = f"Claimed: {claim.value} | SEC EDGAR (cached): {_format_value(actual_val)} ({discrepancy})"
        return result
    except Exception as exc:
        logger.debug("DB lookup skipped (%s) — falling back to live EDGAR", exc)
        return None


async def _get_concept_index(cik: str, gaap: dict) -> tuple[list[str], Any]:
    """
    Build (once per company per process lifetime) a matrix of concept label
    embeddings so every subsequent claim for the same ticker is O(1) dot-product
    lookup rather than a fresh encode pass.

    Single-flight: the first claim that reaches this function for a given CIK
    acquires the lock and runs the CPU-bound encode; all concurrent claims for the
    same ticker wait and then read the warm cache.
    """
    if cik in _concept_index_cache:
        return _concept_index_cache[cik]

    async with _concept_index_locks_guard:
        lock = _concept_index_locks.setdefault(cik, asyncio.Lock())
    async with lock:
        if cik in _concept_index_cache:
            return _concept_index_cache[cik]

        concepts: list[str] = []
        labels: list[str] = []
        for concept, data in gaap.items():
            label = data.get("label", "")
            if label:
                concepts.append(concept)
                labels.append(label)

        if not concepts:
            _concept_index_cache[cik] = ([], None)
            return [], None

        try:
            import numpy as np
            from .embeddings import embed

            loop = asyncio.get_event_loop()
            vecs = await loop.run_in_executor(None, embed, labels)
            matrix = np.array(vecs, dtype="float32")
            _concept_index_cache[cik] = (concepts, matrix)
            logger.info("  [xbrl-embed] built concept index cik=%s — %d concepts", cik, len(concepts))
        except Exception as exc:
            logger.debug("  [xbrl-embed] index build failed (%s) — semantic search unavailable", exc)
            _concept_index_cache[cik] = ([], None)

    return _concept_index_cache[cik]


async def _find_concepts_by_embedding(gaap: dict, metric: str, cik: str) -> list[str]:
    """
    Semantic tier: embed the metric phrase and return the top-matching XBRL
    concepts by cosine similarity against the company's label index.

    Handles cases word-overlap misses entirely:
      "provision for income taxes"  → IncomeTaxExpenseBenefit
      "SG&A"                        → SellingGeneralAndAdministrativeExpense
      "earnings before interest"    → OperatingIncomeLoss
    Threshold 0.60 keeps false positives low; _pick_best_candidate then picks
    the numerically closest one when several concepts clear the bar.
    """
    try:
        import numpy as np
        from .embeddings import embed_one

        loop = asyncio.get_event_loop()
        metric_vec = await loop.run_in_executor(None, embed_one, metric)
        metric_arr = np.array(metric_vec, dtype="float32")

        concepts, matrix = await _get_concept_index(cik, gaap)
        if matrix is None or len(concepts) == 0:
            return []

        sims = matrix @ metric_arr
        top_idx = int(sims.argmax())
        top_sim = float(sims[top_idx])

        if top_sim < 0.60:
            return []

        # Collect all concepts within 0.03 of the top score — they're
        # essentially tied and _pick_best_candidate will resolve by value.
        threshold = top_sim - 0.03
        matched = [concepts[i] for i, s in enumerate(sims) if float(s) >= threshold][:5]
        logger.info("  [xbrl-embed] metric=%r → %s (top_sim=%.3f)", metric, matched[:3], top_sim)
        return matched
    except Exception as exc:
        logger.debug("  [xbrl-embed] lookup error: %s", exc)
        return []


def _find_concepts_by_label(gaap: dict, metric: str) -> list[str]:
    """
    Semantic fallback: scan every us-gaap concept's official XBRL label and return
    the best-matching concept(s) for the given metric phrase.

    This replaces keyword guessing with the filing's own vocabulary — every company
    fact in EDGAR already carries a human-readable label (e.g. "Cash, Cash
    Equivalents, and Short-term Investments") that we can match directly against the
    claim's metric phrase, no hard-coded dictionary required.

    Scoring: count how many significant words from the metric appear in the label
    (and vice versa), normalised by the longer word-set size.  Require a score ≥ 0.5
    so partial noise doesn't produce false matches.  When multiple concepts tie,
    return all of them — _pick_best_candidate will disambiguate by value.
    """
    if not metric or not gaap:
        return []

    _STOPWORDS = frozenset({
        "and", "or", "of", "the", "in", "a", "an", "to", "for", "net",
        "total", "basic", "diluted", "current", "non", "other", "from",
        "attributable", "including", "excluding", "per", "as",
    })

    metric_words = {
        w for w in re.findall(r"[a-z]+", metric.lower()) if w not in _STOPWORDS and len(w) > 1
    }
    if not metric_words:
        return []

    best_score = 0.0
    best_concepts: list[str] = []

    for concept, concept_data in gaap.items():
        label = concept_data.get("label", "")
        if not label:
            continue
        label_words = {
            w for w in re.findall(r"[a-z]+", label.lower()) if w not in _STOPWORDS and len(w) > 1
        }
        if not label_words:
            continue
        overlap = len(metric_words & label_words)
        if overlap == 0:
            continue
        score = overlap / max(len(metric_words), len(label_words))
        if score > best_score:
            best_score = score
            best_concepts = [concept]
        elif score == best_score and score > 0:
            best_concepts.append(concept)

    if best_score >= 0.5:
        logger.info("  [xbrl-live] label-match: metric=%r → concepts=%s (score=%.2f)",
                    metric, best_concepts[:3], best_score)
        return best_concepts[:3]  # cap at 3; value-proximity picks the right one
    return []


async def lookup_direct_fact(claim) -> ValidationResult:
    result = ValidationResult(claim_id=claim.id)
    try:
        # Non-US exchange tickers (WSE:DOM, LSE:VOD, etc.) are never in SEC EDGAR
        if claim.ticker and _is_non_us_ticker(claim.ticker):
            logger.info("  [xbrl-live] skipping EDGAR — non-US ticker %r", claim.ticker)
            result.reasoning = f"Non-US exchange ticker {claim.ticker} — not in SEC EDGAR"
            return result

        # DB-first: check MongoDB cache before hitting EDGAR.
        # Short-circuit rules:
        # 1. CONTRADICTED from DB — always fall through to live EDGAR; the cache
        #    may hold the wrong period and a false CONTRADICTED floors the score.
        # 2. PARTIALLY_VERIFIED from DB with no year hint — fall through too.
        #    Without a year anchor the cache may return a "close but wrong year"
        #    result (e.g. FY2024 $245B at 14.9% off when stated=$281.7B FY2025).
        #    Live EDGAR with _find_all_annual_values will find the exact match.
        # 3. VERIFIED, or PARTIALLY_VERIFIED when a year hint confirms the period —
        #    accept from DB (no need to hit EDGAR again).
        _year_hint_cache = re.search(r"20\d{2}", claim.period or "")
        cached = await _lookup_from_db(claim)
        _accept_cache = (
            cached is not None
            and cached.status == ValidationStatus.VERIFIED
        ) or (
            cached is not None
            and cached.status == ValidationStatus.PARTIALLY_VERIFIED
            and _year_hint_cache is not None
        )
        if _accept_cache:
            return cached

        cik = await get_cik(claim.ticker)
        if not cik:
            result.reasoning = f"Ticker {claim.ticker} not found in SEC EDGAR"
            return result

        url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        logger.info("  [xbrl-live] fetching EDGAR company facts: %s", url)
        facts = await get_company_facts(cik)
        gaap = facts.get("facts", {}).get("us-gaap", {})
        # Track concept source so we know how confident the match is.
        # CONCEPT_MAP hits are exact/curated → can produce CONTRADICTED.
        # Semantic matches (label-overlap / embedding) are fuzzy → a large
        # value discrepancy probably means we matched the wrong concept, NOT
        # that the claim is wrong. Downgrade those to UNVERIFIABLE rather
        # than producing a confident false CONTRADICTED verdict.
        _semantic_match = False
        _quarterly_period = _parse_quarterly_period(claim.period or "")

        # When a quarterly period is detected and the metric begins with a temporal
        # qualifier ("quarterly revenue", "Q1 revenue", "first quarter EPS"), strip
        # the qualifier before concept lookup so the CONCEPT_MAP / disqualifier
        # checks see the base metric ("revenue", "EPS") rather than the decorated
        # phrase.  Segment-specific qualifiers ("iPhone", "Services", "Azure") are
        # intentionally left in place — those are NOT in EDGAR and should fall
        # through to Tavily without an XBRL attempt.
        _metric_for_lookup = claim.metric or ""
        if _quarterly_period:
            _metric_for_lookup = re.sub(
                r"^\s*(?:quarterly|annual|q[1-4]|fy\s*\d{4})\s+",
                "",
                _metric_for_lookup,
                flags=re.IGNORECASE,
            ).strip()

        concepts = find_concepts(_metric_for_lookup)
        if not concepts:
            concepts = _find_concepts_by_label(gaap, _metric_for_lookup)
            _semantic_match = bool(concepts)
        if not concepts:
            concepts = await _find_concepts_by_embedding(gaap, _metric_for_lookup, cik)
            _semantic_match = bool(concepts)
        if not concepts:
            result.reasoning = f"No XBRL concept mapped for metric: {claim.metric}"
            logger.info("  [xbrl-live] no concept found (CONCEPT_MAP + label + embedding) for metric=%r (lookup=%r)",
                        claim.metric, _metric_for_lookup)
            return result

        logger.info("  [xbrl-live] trying concepts %s for metric=%r", concepts, claim.metric)
        candidates: list[tuple[str, float, str, str, str]] = []
        _year_hint = re.search(r"20\d{2}", claim.period or "")
        for concept in concepts:
            if concept not in gaap:
                logger.info("  [xbrl-live] concept=%s not in us-gaap — skipping", concept)
                continue
            units = gaap[concept].get("units", {})
            if _quarterly_period:
                # Quarterly claim: look for 10-Q entries with matching fp/fy.
                # Don't mix with annual values — quarterly and annual revenue are
                # different numbers (e.g. Apple Q1=$119B vs FY=$391B).
                found = _find_quarterly_value(units, *_quarterly_period)
                if found:
                    actual_val, actual_period, accn, filed = found
                    candidates.append((concept, actual_val, actual_period, accn, filed))
                    logger.info("  [xbrl-live] 10-Q HIT: concept=%s q=%s fy=%s val=%s",
                                concept, _quarterly_period[0], _quarterly_period[1],
                                _format_value(actual_val))
                else:
                    logger.info("  [xbrl-live] concept=%s no 10-Q entry for Q%s FY%s",
                                concept, *_quarterly_period)
            elif _year_hint:
                # Annual period specified: one entry per concept (existing behaviour)
                found = _find_period_value(units, claim.period)
                if not found:
                    logger.info("  [xbrl-live] concept=%s found but no annual entry for period=%r", concept, claim.period)
                    continue
                actual_val, actual_period, accn, filed = found
                candidates.append((concept, actual_val, actual_period, accn, filed))
            else:
                # No period hint: add one candidate per recent annual period so
                # _pick_best_candidate can find the year whose value matches best.
                for actual_val, actual_period, accn, filed in _find_all_annual_values(units):
                    candidates.append((concept, actual_val, actual_period, accn, filed))

        # _find_period_value() falls back to "most recent annual entry for THIS
        # concept" whenever the claim's period isn't found — which means two
        # candidates gathered above can legitimately come from different
        # reporting periods (e.g. "Revenues" last reported FY2010 = $62.48B,
        # "RevenueFromContract...ExcludingAssessedTax" last reported FY2025 =
        # $281.72B). When a year IS specified, align all candidates to that
        # period before disambiguating.
        # When NO year is specified (bare table cells, historical comparisons),
        # keep all periods so _pick_best_candidate finds the year whose value
        # numerically matches the stated value (e.g. FY2024 interest expense
        # $2,935M vs FY2025 $2,385M — the correct match is FY2024).
        if candidates and _year_hint:
            aligned = [c for c in candidates if _year_hint.group() in c[2]]
            if not aligned:
                latest_period = max(c[2] for c in candidates)
                aligned = [c for c in candidates if c[2] == latest_period]
            if len(aligned) != len(candidates):
                logger.info("  [xbrl-live] period-aligned candidates %s -> %s (claim.period=%r)",
                            [(c[0], c[2]) for c in candidates], [(c[0], c[2]) for c in aligned], claim.period)
            candidates = aligned

        if candidates:
            concept, actual_val, actual_period, accn, filed = _pick_best_candidate(
                candidates, claim.value, key=lambda c: c[1]
            )
            cik_bare = cik.lstrip("0")
            accn_clean = accn.replace("-", "")
            edgar_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/{accn_clean}/"
                if accn_clean else None
            )
            logger.info("  [xbrl-live] HIT: concept=%s value=%s period=%s accn=%s url=%s (chosen from %d candidate concept(s): %s)",
                        concept, _format_value(actual_val), actual_period, accn, edgar_url,
                        len(candidates), [c[0] for c in candidates])
            status, confidence, discrepancy = _compare(claim.value or "", actual_val)

            # Year-correction fallback: if the extractor assigned the wrong fiscal year
            # (e.g. LLM wrote "FY2024" for a letter about FY2025 results), the
            # year-hint lookup returns CONTRADICTED or a poor PARTIALLY_VERIFIED
            # against the wrong period.  Retry without the year restriction so
            # _pick_best_candidate can find the period whose XBRL value matches best.
            # Only replace the year-hint result if the all-period candidate is
            # strictly better (e.g. VERIFIED > PARTIALLY_VERIFIED > CONTRADICTED).
            # A truly wrong stated value won't match ANY period, so false negatives rare.
            # NOT applied to quarterly claims: _find_all_annual_values returns 10-K
            # totals (e.g. FY annual revenue $391B) which would always dwarf a
            # single-quarter value ($119B) and produce a confident false CONTRADICTED.
            _STATUS_RANK = {
                ValidationStatus.VERIFIED: 3,
                ValidationStatus.PARTIALLY_VERIFIED: 2,
                ValidationStatus.UNVERIFIABLE: 1,
                ValidationStatus.CONTRADICTED: 0,
            }
            _is_poor_match = (
                status == ValidationStatus.CONTRADICTED
                or (status == ValidationStatus.PARTIALLY_VERIFIED and actual_val != 0
                    and abs(abs(_parse_value(claim.value or "") or 0) * 1 - abs(actual_val)) / abs(actual_val) > 0.05)
            )
            if _is_poor_match and _year_hint and not _quarterly_period:
                logger.info(
                    "  [xbrl-live] poor year-hint match (%s, hint=%r) — retrying across all periods",
                    status.value, _year_hint.group(),
                )
                all_candidates: list[tuple[str, float, str, str, str]] = []
                for c_concept in concepts:
                    if c_concept not in gaap:
                        continue
                    c_units = gaap[c_concept].get("units", {})
                    for av, ap, aa, af in _find_all_annual_values(c_units):
                        all_candidates.append((c_concept, av, ap, aa, af))
                if all_candidates:
                    bc = _pick_best_candidate(all_candidates, claim.value, key=lambda c: c[1])
                    b_status, b_conf, b_disc = _compare(claim.value or "", bc[1])
                    if _STATUS_RANK.get(b_status, 0) > _STATUS_RANK.get(status, 0):
                        concept, actual_val, actual_period, accn, filed = bc
                        status, confidence, discrepancy = b_status, b_conf, b_disc
                        logger.info(
                            "  [xbrl-live] year-correction: period corrected to %s (%s)",
                            actual_period, b_status.value,
                        )

            # Semantic matches (label-overlap / embedding) may have resolved the
            # wrong concept.  A large discrepancy under CONCEPT_MAP is a real
            # contradiction; under a fuzzy semantic match it just means we
            # matched the wrong line item.  Downgrade CONTRADICTED and
            # PARTIALLY_VERIFIED to UNVERIFIABLE so the pipeline continues to
            # value-match, RAG, and web-search rather than short-circuiting with
            # a confident false verdict against the wrong XBRL concept.
            if _semantic_match and status in (
                ValidationStatus.CONTRADICTED, ValidationStatus.PARTIALLY_VERIFIED
            ):
                status = ValidationStatus.UNVERIFIABLE
                confidence = 0.5
                discrepancy = f"semantic match uncertain (concept={concept}, diff={discrepancy})"
            result.status = status
            result.confidence = confidence
            result.actual_value = _format_value(actual_val)
            result.discrepancy = discrepancy
            _form_label = "10-Q" if _quarterly_period else "10-K"
            result.filing_source = f"SEC EDGAR XBRL ({_form_label}) — {concept}, {actual_period}"
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
            result.structured_citations = [
                _build_xbrl_citation(
                    cik, claim.ticker or "", concept, actual_period,
                    _format_value(actual_val), accn or None, edgar_url,
                    concept_label=gaap[concept].get("label"),
                )
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
        # If the stated value is very large (> 100), it's an absolute figure extracted
        # from an "(in millions)" table rather than a percentage — e.g. the row
        # "Gross margin | 193,893 | 171,008 | 13%" where the LLM may tag
        # metric="gross margin" but value="193,893" (the gross profit dollar amount).
        # Comparing $193,893M against a computed 68.8% produces a guaranteed
        # CONTRADICTED that is completely wrong.  Return UNVERIFIABLE so the
        # pipeline continues to the value-match step.
        if stated_pct > 100:
            result.reasoning = (
                f"Stated value {stated_pct:,.0f} is an absolute figure, not a "
                f"percentage — skipping margin comparison for metric={claim.metric!r}"
            )
            return result

        diff = abs(actual_pct - stated_pct)
        # A discrepancy > 40pp almost certainly means the claimed value is a
        # YoY growth rate (e.g. "13% growth in gross margin") rather than the
        # margin level itself (~68.8%).  Comparing a growth rate against a level
        # metric always produces a false CONTRADICTED.  Return UNVERIFIABLE so
        # the pipeline continues to RAG / web-search.
        if diff > 40:
            result.reasoning = (
                f"Stated {stated_pct:.1f}% differs from computed {actual_pct:.2f}% by "
                f"{diff:.1f}pp — likely a YoY growth rate vs margin level for "
                f"metric={claim.metric!r}"
            )
            return result
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
        result.structured_citations = [
            Citation(
                source="FORMULA",
                label=f"Computed: {numerator_key} / {denominator_key}",
                ticker=claim.ticker,
                field=claim.metric,
                value=f"{actual_pct:.2f}%",
                excerpt=f"{numerator_key}={_format_value(num)}, {denominator_key}={_format_value(den)}",
            )
        ]
        logger.info("  [live-ratio] computed %s=%.2f%% diff=%.2fpp → %s",
                    claim.metric, actual_pct, diff, result.status.value)
    except Exception as exc:
        logger.error("lookup_derived_ratio error: %s", exc)
        result.status = ValidationStatus.ERROR
        result.reasoning = str(exc)
    return result


async def lookup_growth_rate(claim) -> ValidationResult:
    """Verify a YoY percentage growth claim (e.g. 'revenue up 15%') from EDGAR.

    Computes (current_period - prior_period) / |prior_period| × 100 using the
    two most recent annual XBRL entries for the underlying concept, then
    compares against the stated growth percentage.
    """
    result = ValidationResult(claim_id=claim.id)

    metric_lower = (claim.metric or "").lower()
    # Strip growth-qualifier words so "revenue growth" resolves to "revenue"
    for strip_word in ("growth", "increase", "change", "growth rate"):
        metric_lower = metric_lower.replace(strip_word, "").strip()

    concepts = find_concepts(metric_lower)
    if not concepts:
        result.reasoning = f"No XBRL concept for growth metric: {claim.metric!r}"
        return result

    stated_pct = _parse_value(claim.value or "")
    if stated_pct is None or stated_pct > 200:
        result.reasoning = "Could not parse stated growth rate (or value is not a percentage)"
        return result

    if claim.ticker and _is_non_us_ticker(claim.ticker):
        result.reasoning = f"Non-US exchange ticker {claim.ticker} — not in SEC EDGAR"
        return result

    try:
        cik = await get_cik(claim.ticker)
        if not cik:
            result.reasoning = f"Ticker {claim.ticker} not found"
            return result

        facts = await get_company_facts(cik)
        gaap = facts.get("facts", {}).get("us-gaap", {})

        for concept in concepts:
            if concept not in gaap:
                continue
            units = gaap[concept].get("units", {})
            annual_vals = _find_all_annual_values(units, max_years=3)
            if len(annual_vals) < 2:
                continue

            # If a year hint is present, anchor the "current" period to that year
            year_hint = re.search(r"20\d{2}", claim.period or "")
            if year_hint:
                target = year_hint.group()
                anchored = [(v, p, a, f) for v, p, a, f in annual_vals if target in p]
                if anchored:
                    idx = annual_vals.index(anchored[0])
                    if idx + 1 < len(annual_vals):
                        current_val, current_period = annual_vals[idx][0], annual_vals[idx][1]
                        prior_val, prior_period = annual_vals[idx + 1][0], annual_vals[idx + 1][1]
                    else:
                        continue
                else:
                    current_val, current_period = annual_vals[0][0], annual_vals[0][1]
                    prior_val, prior_period = annual_vals[1][0], annual_vals[1][1]
            else:
                current_val, current_period = annual_vals[0][0], annual_vals[0][1]
                prior_val, prior_period = annual_vals[1][0], annual_vals[1][1]

            if prior_val == 0:
                continue

            actual_growth = (current_val - prior_val) / abs(prior_val) * 100
            diff = abs(actual_growth - stated_pct)

            # Year-correction fallback: if the year-hint produces a poor growth
            # match (diff > 6pp), also try the most-recent period pair without
            # the year constraint.  This fixes "FY2024" anchoring FY2023→FY2024
            # when the letter is actually about FY2024→FY2025.  Only replace
            # if the no-hint computation is strictly closer to the stated value.
            if diff > 6.0 and year_hint and len(annual_vals) >= 2:
                nh_curr_val, nh_curr_period = annual_vals[0][0], annual_vals[0][1]
                nh_prior_val, nh_prior_period = annual_vals[1][0], annual_vals[1][1]
                if nh_prior_val != 0:
                    nh_growth = (nh_curr_val - nh_prior_val) / abs(nh_prior_val) * 100
                    nh_diff = abs(nh_growth - stated_pct)
                    if nh_diff < diff:
                        logger.info(
                            "  [growth-rate] year-correction: hint gave %.1fpp diff, "
                            "no-hint gives %.1fpp → using %s→%s",
                            diff, nh_diff, nh_prior_period, nh_curr_period,
                        )
                        actual_growth = nh_growth
                        diff = nh_diff
                        current_val, current_period = nh_curr_val, nh_curr_period
                        prior_val, prior_period = nh_prior_val, nh_prior_period

            if diff < 1.0:
                status, confidence = ValidationStatus.VERIFIED, 0.95
            elif diff < 3.0:
                status, confidence = ValidationStatus.VERIFIED, 0.85
            elif diff < 6.0:
                status, confidence = ValidationStatus.PARTIALLY_VERIFIED, 0.75
            elif diff < 12.0:
                status, confidence = ValidationStatus.PARTIALLY_VERIFIED, 0.60
            else:
                status, confidence = ValidationStatus.CONTRADICTED, 0.88

            result.status = status
            result.confidence = confidence
            result.actual_value = f"{actual_growth:.1f}%"
            result.discrepancy = f"{diff:.1f}pp difference"
            result.filing_source = (
                f"SEC EDGAR XBRL — {concept} YoY growth, "
                f"{prior_period} → {current_period}"
            )
            result.reasoning = (
                f"Stated: {stated_pct:.1f}% | Computed: {actual_growth:.1f}% "
                f"({_format_value(prior_val)} → {_format_value(current_val)})"
            )
            logger.info(
                "  [growth-rate] %s: stated=%.1f%% actual=%.1f%% diff=%.1fpp → %s",
                concept, stated_pct, actual_growth, diff, status.value,
            )
            return result

        result.reasoning = f"Could not compute YoY growth for {claim.metric!r}"
    except Exception as exc:
        logger.error("lookup_growth_rate error: %s", exc)
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
    if not kws:
        # No semantic anchor — scanning all ~1000 concepts by value alone
        # guarantees spurious matches (e.g. $9.4B "shareholder returns" hitting
        # "Land" because MSFT's land value is also $9.34B).  Skip rather than
        # produce a confidently-wrong VERIFIED verdict.
        logger.info("  [value-match] no keyword filter for metric=%r — skipping to avoid spurious matches", claim.metric)
        return result

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
        best: Optional[tuple] = None  # (concept, namespace, entry, actual_val, concept_label)

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
                            best = (concept, namespace, entry, val, concept_data.get("label"))

        if best is None:
            return result

        concept, namespace, entry, actual_val, concept_label = best
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
        result.structured_citations = [
            _build_xbrl_citation(
                cik, claim.ticker or "", concept, entry["end"],
                _format_value(actual_val), accn or None, edgar_url, namespace,
                concept_label=concept_label,
            )
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
