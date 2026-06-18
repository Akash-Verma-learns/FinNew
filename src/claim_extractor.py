from __future__ import annotations

import logging
import re
from typing import Optional

from .groq_client import chat, parse_json
from .models import Claim, ClaimType

logger = logging.getLogger(__name__)

VALID_TYPES = {t.value for t in ClaimType}

EXTRACTION_PROMPT = """You are a financial analyst extracting verifiable claims from a financial document.

RULES:
1. Extract ONLY claims that contain a concrete, verifiable numerical value OR a specific forward-looking prediction.
2. For markdown tables, extract one claim per data cell that contains a financial metric AND its numeric value.
3. Extract 10-20 claims per section. If the report has tables, extract claims from EVERY data row (not header rows).
4. Copy ALL digits EXACTLY as written. Never truncate: "$391B" not "$91B", "$108.8B" not "$8.8B".
5. DO NOT extract:
   - Table headers or column labels (e.g. "Year Ended June 30,", "2024", "2023")
   - Unit labels (e.g. "(In millions)", "(In thousands)")
   - Pure accounting policy boilerplate sentences with no quantitative content
   - Section titles or footnote labels
   - Sentences that describe methods without any numbers (e.g. "We review investments quarterly")
   - Numbers that are part of a product or feature name (e.g. "Microsoft 365", "Majorana-1", "Level 2 quantum", "GPT-4")
6. metric: KEEP THE FULL QUALIFIER. Never strip context words.
   - "Microsoft Cloud revenue increased 23% to $168.9 billion" → metric: "Microsoft Cloud revenue", value: "$168.9 billion"
   - "LinkedIn revenue increased 9%" → metric: "LinkedIn revenue", value: "9%"
   - "Azure cloud services revenue growth of 34%" → metric: "Azure cloud services revenue growth", value: "34%"
   - "Server products and cloud services revenue increased 23%" → metric: "Server products and cloud services revenue", value: "23%"
   - "net income" (a standalone total) → metric: "net income"  ← only use bare terms for consolidated totals
   - WRONG: metric: "revenue" for a segment line. RIGHT: metric: "Microsoft Cloud revenue"
   - QUALITATIVE claims must still have a metric describing WHAT the number measures: metric: "Carvana inbound calls reduction", metric: "LinkedIn members", metric: "GitHub Copilot users", metric: "datacenters operated"
7. type — assign based on what the claim is about:
   - DIRECT_FACT: ONLY the reporting company's own consolidated financial statement line items: total revenue, net income, operating income, gross profit, EPS, cash and equivalents, total assets, total debt, shares outstanding. Dollar amounts that appear on a GAAP income statement, balance sheet, or cash flow statement.
   - DERIVED_METRIC: ONLY the reporting company's own calculated financial ratios or YoY growth rates using GAAP financial statement data — e.g. "revenue grew 15%", "operating margin of 46%", "EPS grew 10%". Never use DERIVED_METRIC for third-party company metrics, ESG metrics, or operational product metrics even if they contain percentages.
   - FORWARD_PROJECTION: Stated investments, targets, or plans for a future period not yet completed.
   - QUALITATIVE: Use for ALL of the following, even when they include a specific number or percentage:
     * Third-party customer or partner outcomes — ANY stat about another company: "Carvana reduced inbound calls by 45%", "Mercy saved 100,000 hours", "Barclays deploying AI to 100,000 employees". Third-party metrics are QUALITATIVE even if expressed as a growth rate.
     * Product/platform operational metrics of the reporting company: monthly active users, subscriber counts, paid customer counts, model counts, member counts
     * Infrastructure and headcount metrics: datacenter count, region count, engineer equivalents, employee headcount
     * ESG and sustainability metrics: gigawatts capacity, metric tons of carbon removal, renewable energy figures
     * Comparative or superlative claims: "world's first", "fastest-growing", "more than any other provider", "10x performance"
   - RULE: Before assigning DERIVED_METRIC, scan the raw_text for company names that are NOT the reporting company. If you find any (e.g. "Carvana", "Mercy", "Barclays", customer/partner names), the claim is about a third party and MUST be QUALITATIVE. Example: "Carvana has reduced inbound calls per sale by 45 percent" → QUALITATIVE (third-party stat, not a reporting-company financial ratio).
   - RULE: ESG/sustainability percentages (renewable energy, carbon reduction, water usage) are QUALITATIVE even if expressed as growth rates.
8. value: For a sentence stating both a growth % AND an absolute figure, prefer the absolute figure (e.g. "$168.9 billion" over "23%") unless the claim is clearly about the growth rate.

PERIOD RULES (critical — wrong period causes false contradictions):
- Look for the document's reporting year first. A shareholder/CEO letter signed in mid-2025 or later that discusses "this year's" financials is reporting on the most recently completed fiscal year (e.g., "FY2025" for a company with June fiscal year end).
- Set period to the FISCAL YEAR the financial metric belongs to, not a calendar year that appears nearby in the text for a different purpose (e.g., "34 gigawatts in 2024" refers to the environmental metric for 2024, not to revenue).
- For core P&L line items (revenue, operating income, net income, EPS) with no explicit year stated: use the fiscal year of the document's primary reporting period.
- QUARTERLY period format: If the text explicitly mentions a quarter (Q1, Q2, Q3, Q4, "first quarter", "second quarter", "third quarter", "fourth quarter"), you MUST include the quarter in the period field. Use format "Q1 FY2024", "Q2 FY2024", etc. NEVER collapse "Q1 FY2024" to just "FY2024" — that causes quarterly values to be compared against annual totals and produces false contradictions.
  Examples: "Q1 FY2024 revenue of $119.58B" → period="Q1 FY2024"; "second quarter net income" → period="Q2 FY2024"; "in the first quarter of fiscal 2024" → period="Q1 FY2024"
- ANNUAL period format: "FY2025", "FY2024", etc. — NEVER just "2025" or "2024" without the "FY" prefix for annual financial results.

Return ONLY a JSON array (no markdown, no explanation). Each item:
- id: string (e.g. "claim_1")
- raw_text: string (exact quote from table row or sentence)
- type: one of DIRECT_FACT | DERIVED_METRIC | ACCOUNTING_POLICY | MODEL_ASSUMPTION | FORWARD_PROJECTION | RECOMMENDATION | QUALITATIVE
- company: string or null
- ticker: string or null — ALWAYS include the exchange prefix for non-US stocks (e.g. "TSX:CJT", "WSE:DOM", "LSE:VOD", "ASX:BHP"). For US stocks use plain ticker (e.g. "AAPL"). Never omit the exchange prefix for non-US companies.
- metric: string or null — full qualifier required for segment/product metrics (see rule 5)
- value: string or null (e.g. "$168.9 billion", "$391B", "46.2%", "29x")
- period: string or null — "Q1 FY2024" for quarterly, "FY2024" for annual. NEVER strip the quarter prefix from a quarterly metric.
- checkable: boolean (true if the claim contains a specific numeric value that could be confirmed or refuted against an authoritative source; false if no numeric value, or if the value is a vague superlative like "world's first" or "10x" with no independent reference)

Example for a table row "| Revenue | 394.3 | 383.3 | 391.0 |":
Extract THREE claims: revenue FY2022=$394.3B, revenue FY2023=$383.3B, revenue FY2024=$391.0B

JSON array only:

{text}"""


# Matches monetary values: $391 billion, $108.8B, 46.2%, $96.2 billion, etc.
_NUM_PATTERN = re.compile(
    r"\$?\s*(\d[\d,]*\.?\d*)\s*(trillion|billion|million|T|B|M)?",
    re.IGNORECASE,
)
_VALUE_PATTERN = re.compile(
    r"[\$€£]?\s*\d[\d,]*\.?\d*\s*(?:trillion|billion|million|%|T|B|M|x)?",
    re.IGNORECASE,
)


def _parse_numeric(s: str) -> Optional[float]:
    """'$391B' / '$391 billion' / '46.2%' → float (raw units, e.g. 391e9)."""
    if not s:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"([\d.]+)\s*(trillion|billion|million|T|B|M)?", s, re.IGNORECASE)
    if not m:
        return None
    num = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    mult = {"trillion": 1e12, "t": 1e12, "billion": 1e9, "b": 1e9, "million": 1e6, "m": 1e6}
    return num * mult.get(suffix, 1)


def _all_numbers_in_text(text: str) -> list[tuple[float, str]]:
    """Return (numeric_value, original_string) for every number in `text`."""
    results = []
    for m in _VALUE_PATTERN.finditer(text):
        raw = m.group(0).strip()
        val = _parse_numeric(raw)
        if val is not None and val != 0:
            results.append((val, raw))
    return results


_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def _is_product_or_temporal_extraction(value_str: str, raw_text: str) -> bool:
    """Return True when a bare integer value was extracted from a product name or temporal phrase.

    Targets patterns the LLM is supposed to skip (prompt rule 5) but occasionally violates:
      - Hyphenated product names: "Majorana-1", "GPT-4", "Level-2"
      - Capitalized-word + number: "Microsoft 365", "Windows 11", "Office 365"
      - Temporal context: "in 10 years", "over 5 years"
    """
    digits = re.sub(r"[,\s]", "", value_str or "").strip()
    if not re.fullmatch(r"\d+", digits) or not raw_text:
        return False
    n = re.escape(digits)
    return bool(re.search(
        rf'\b\w+-{n}\b'                                         # "Majorana-1", "GPT-4"
        rf'|\b{n}-\w+\b'                                        # "2-factor"
        rf'|\b[A-Z][a-z]\w*(?:\s+[A-Z][a-z]\w*)*\s+{n}\b'     # "Microsoft 365", "Windows 11"
        rf'|\b{n}\s+(?:year|month|day|week|decade)s?\b',       # "10 years"
        raw_text, re.IGNORECASE
    ))


def _looks_like_year(raw: str) -> bool:
    """True if the raw string is just a calendar year (e.g. '2022', '2017')."""
    digits = re.sub(r"[^\d]", "", raw.split()[0]) if raw.split() else ""
    return bool(_YEAR_RE.match(digits))


def _fix_value_against_source(extracted_value: str, source_text: str) -> str:
    """
    If the LLM dropped leading digits (e.g. '$91B' from '$391B'), find the
    correct number in the original source text and return it.

    Heuristic: if a number in the source text ends with the same digit-string as the
    extracted value AND is at least 2× larger, the LLM truncated it — use source.

    Guards against false positives caused by year numbers (2017, 2022, 2024):
    - Skip any source string whose first token is a calendar year (19xx / 20xx)
    - Cap ratio at 1000 — legitimate digit-drops are at most ~30× (e.g. $3.7B→$93.7B)
    """
    extracted_num = _parse_numeric(extracted_value)
    if not extracted_num:  # None or 0 — can't compute ratio, nothing to fix
        return extracted_value

    source_numbers = _all_numbers_in_text(source_text)
    best_raw = extracted_value
    best_ratio = 0.0

    extracted_digits = re.sub(r"[^\d]", "", extracted_value)

    for src_val, src_raw in source_numbers:
        if src_val <= extracted_num:
            continue
        ratio = src_val / extracted_num
        # Artifact guard: "T" after newline inflates years to trillions
        if ratio > 1000:
            continue
        # Year guard: skip "2022", "2017", "2024" etc.
        if _looks_like_year(src_raw):
            continue
        # Only candidate if extracted digits are a suffix of source digits
        source_digits = re.sub(r"[^\d]", "", src_raw)
        if not source_digits.endswith(extracted_digits):
            continue
        if ratio > best_ratio:
            best_ratio = ratio
            best_raw = src_raw

    if best_ratio >= 2.0:
        logger.warning(
            "Value fix: LLM extracted %r but source has %r (ratio=%.1f×) — using source",
            extracted_value, best_raw, best_ratio,
        )
        return best_raw.strip()
    return extracted_value


async def extract_claims(text: str) -> list[Claim]:
    claims = await _extract_claims_raw(text)
    return _backfill_primary_ticker(claims)


async def _extract_claims_raw(text: str) -> list[Claim]:
    """Single-pass extraction + sanitize, with NO cross-claim ticker backfill.

    Used directly by `extract_claims_from_sections` (per section) so that a
    section-local majority vote can't contaminate claims with the wrong
    company's ticker; the caller backfills once, globally, at the end.
    """
    import json as _json
    prompt = EXTRACTION_PROMPT.format(text=text[:8000])
    raw = await chat(prompt)
    try:
        items: list[dict] = parse_json(raw)
    except (_json.JSONDecodeError, ValueError) as exc:
        logger.warning("Claim extraction: JSON parse failed (%s) — raw response: %.200s", exc, raw)
        return []
    if not isinstance(items, list):
        logger.warning("Claim extraction returned non-list; wrapping.")
        items = [items]
    claims = []
    for item in items:
        try:
            claims.append(Claim.model_validate(_coerce(item)))
        except Exception as exc:
            logger.debug("Skipping malformed claim item (%s): %s", exc, item)
    return _sanitize(claims, original_text=text)


async def extract_claims_from_sections(sections: dict[str, str]) -> list[Claim]:
    """
    Multi-pass claim extraction for long documents.

    Runs extract_claims on each section independently (no truncation),
    re-IDs claims to avoid collisions, and deduplicates by (metric, value, period).

    Used by /api/analyze-pdf when PageIndex successfully builds a section tree.
    """
    all_claims: list[Claim] = []
    seen: set[tuple] = set()
    claim_counter = 1

    # Priority order: financial sections first so the best claims get low IDs
    priority = ["financials", "valuation", "investment", "thes", "mda", "business", "overview"]

    def _section_priority(title: str) -> int:
        tl = title.lower()
        for i, kw in enumerate(priority):
            if kw in tl:
                return i
        return len(priority)

    ordered = sorted(sections.items(), key=lambda kv: _section_priority(kv[0]))

    for section_title, text in ordered:
        if len(text.strip()) < 100:
            continue
        logger.info("[extractor] section=%r chars=%d", section_title, len(text))
        try:
            section_claims = await _extract_claims_raw(text)
        except Exception as exc:
            logger.warning("[extractor] section=%r failed: %s", section_title, exc)
            continue

        for claim in section_claims:
            key = (
                (claim.metric or "").lower(),
                (claim.value or "").lower(),
                (claim.period or "").lower(),
            )
            if key in seen and any(k for k in key):
                continue
            seen.add(key)
            claim.id = f"claim_{claim_counter}"
            claim_counter += 1
            all_claims.append(claim)

    logger.info("[extractor] multi-pass complete: %d unique claims from %d sections",
                len(all_claims), len(sections))
    return _backfill_primary_ticker(all_claims)


def _backfill_primary_ticker(claims: list[Claim]) -> list[Claim]:
    """Fill missing `ticker`/`company` with the document's dominant ticker.

    The LLM infers `ticker` per-claim from each sentence's local context, but a
    single-company filing (10-K, shareholder letter, etc.) is full of sentences
    that never restate the company name ("Cash and cash equivalents totaled
    $94.6 billion") — those claims come back with `ticker=None`. That is fatal:
    `validate_direct_fact` (and the RAG/non-GAAP/segment/XBRL paths it gates)
    only run `if claim.ticker:` — an untagged claim skips all local lookups and
    lands on the disabled web-search fallback as a sourceless "Unverifiable".
    Since the whole document is about ONE filer, whichever ticker the extractor
    *did* manage to tag most often is overwhelmingly likely to be that filer —
    so backfill the rest with it.
    """
    from collections import Counter

    counts = Counter(
        (c.ticker or "").strip().upper()
        for c in claims if c.ticker and c.ticker.strip()
    )
    if not counts:
        return claims

    primary = counts.most_common(1)[0][0]

    # Use the most common company name (not the ticker string) so Tavily queries
    # say "Microsoft" instead of "MSFT".
    company_counts = Counter(
        c.company.strip()
        for c in claims
        if c.company and c.company.strip() and c.company.strip().upper() != c.ticker
    )
    primary_company = company_counts.most_common(1)[0][0] if company_counts else primary

    backfilled = 0
    for c in claims:
        if not c.ticker or not c.ticker.strip():
            c.ticker = primary
            if not c.company:
                c.company = primary_company
            backfilled += 1
    if backfilled:
        logger.info("[extractor] backfilled ticker=%s onto %d/%d claim(s) lacking a per-sentence company tag",
                    primary, backfilled, len(claims))
    return claims


def _coerce(item: dict) -> dict:
    if item.get("type") not in VALID_TYPES:
        item["type"] = "QUALITATIVE"
    return item


def _sanitize(claims: list[Claim], original_text: str = "") -> list[Claim]:
    for c in claims:
        # Cross-check extracted value against original text to catch digit-drop bugs
        if c.value and original_text:
            c.value = _fix_value_against_source(c.value, original_text)

        # Fallback: extract value from raw_text if still missing
        if not c.value and c.raw_text:
            m = _VALUE_PATTERN.search(c.raw_text)
            if m:
                c.value = m.group(0).strip()

        if not c.company and c.ticker:
            c.company = c.ticker
        if c.checkable and not c.value:
            c.checkable = False
        elif c.checkable and c.value:
            val = c.value.strip()
            parsed = _parse_numeric(val)
            has_pct = "%" in val
            if parsed is None and not has_pct:
                # Descriptive phrase with no number — e.g. "voice communication",
                # "expedited due process" — cannot be confirmed or refuted by data.
                c.checkable = False
            elif re.fullmatch(r"\d+x", val, re.IGNORECASE):
                # Bare multiplier ratio ("10x", "2x") — no absolute reference point.
                c.checkable = False
            elif c.type == ClaimType.QUALITATIVE and parsed is not None and not has_pct:
                # For QUALITATIVE claims with a bare integer value (no units):
                # only mark non-checkable when it looks like a product name / temporal
                # extraction ("Majorana-1" → "1", "Microsoft 365" → "365", "10 years" → "10").
                # Legitimate count metrics ("34,000 engineers", "25,000 customers") stay
                # checkable so the grounding pass can still promote them.
                units_remain = re.sub(r"[\d,.\s]+", "", val).strip()
                if not units_remain and _is_product_or_temporal_extraction(val, c.raw_text or ""):
                    c.checkable = False

    # Drop pure-noise claims that have no verification signal:
    # 1. QUALITATIVE with no value — accounting policy boilerplate sentences,
    #    table section headers, running commentary from the filing text.
    #    These all land UNVERIFIABLE and drag the weighted-average score toward
    #    0.25 even when every numeric claim is correct.
    # 2. Any claim with no metric AND no value — orphaned table cells like a
    #    bare "22" or a column label "2024 $" that the LLM mis-extracted.
    # ACCOUNTING_POLICY is intentionally excluded from filter (1): even without
    # a numeric value it has its own RAG validation path.
    before = len(claims)
    claims = [
        c for c in claims
        if not (
            c.type == ClaimType.QUALITATIVE and not c.value
        ) and not (
            not c.metric and not c.value
        ) and not (
            # Drop QUALITATIVE claims where the value is a product-name or temporal
            # number extraction ("Majorana-1"→"1", "Microsoft 365"→"365", "10 years"→"10").
            c.type == ClaimType.QUALITATIVE
            and c.value
            and _is_product_or_temporal_extraction(c.value, c.raw_text or "")
        )
    ]
    if before != len(claims):
        logger.info(
            "[extractor] dropped %d noise claims (QUALITATIVE-no-value or no-metric-no-value); %d remain",
            before - len(claims), len(claims),
        )

    # Post-process type corrections that the LLM misses consistently:
    # 1. DERIVED_METRIC with a dollar value is almost always a DIRECT_FACT.
    #    "Microsoft Cloud revenue $168.9B up 23%" → LLM picks DERIVED_METRIC because of
    #    the growth context, but the value is absolute and should be DIRECT_FACT.
    # 2. ESG/sustainability metrics should be QUALITATIVE regardless of how expressed.
    _ESG_RE = re.compile(
        r"\b(renewable|solar|wind|carbon|emission|gigawatt|gw|kwh|mwh|gwh|"
        r"water|sanitation|recycl|circular|packaging|landfill|biodiversity|"
        r"sustainability|volunteer|nonprofit|donation|community|philanthrop|"
        r"clean\s+water|waste|reuse)\b",
        re.IGNORECASE,
    )
    _DOLLAR_RE = re.compile(r"\$\s*[\d,.]+\s*(billion|million|trillion|B|M|T)\b", re.IGNORECASE)
    for c in claims:
        metric_raw_combined = (c.metric or "") + " " + (c.raw_text or "")
        if _ESG_RE.search(metric_raw_combined):
            # ESG/CSR/sustainability/philanthropy metrics → QUALITATIVE regardless of original type
            c.type = ClaimType.QUALITATIVE
            logger.debug("[extractor] reclassified ESG→QUALITATIVE: %r", c.metric)
        elif c.type == ClaimType.DERIVED_METRIC and c.value and _DOLLAR_RE.search(c.value):
            # Dollar-valued DERIVED_METRIC is an absolute fact, not a ratio
            c.type = ClaimType.DIRECT_FACT
            logger.debug("[extractor] reclassified dollar DERIVED_METRIC→DIRECT_FACT: %r", c.metric)

    # Deduplicate within-section: same sentence extracted as multiple types
    # (e.g. DIRECT_FACT + QUALITATIVE for the same raw_text and value).
    # Keep the higher-weight type (DIRECT_FACT > DERIVED_METRIC > QUALITATIVE).
    _TYPE_RANK = {
        ClaimType.DIRECT_FACT: 5, ClaimType.DERIVED_METRIC: 4,
        ClaimType.ACCOUNTING_POLICY: 3, ClaimType.FORWARD_PROJECTION: 2,
        ClaimType.MODEL_ASSUMPTION: 2, ClaimType.RECOMMENDATION: 1,
        ClaimType.QUALITATIVE: 0,
    }
    seen_rv: dict[tuple, Claim] = {}
    for c in claims:
        key = ((c.raw_text or "")[:80].lower(), (c.value or "").lower())
        if key in seen_rv:
            existing = seen_rv[key]
            if _TYPE_RANK.get(c.type, 0) > _TYPE_RANK.get(existing.type, 0):
                seen_rv[key] = c
        else:
            seen_rv[key] = c
    deduped = list(seen_rv.values())
    if len(deduped) != len(claims):
        logger.info("[extractor] deduped %d within-section duplicate(s); %d remain",
                    len(claims) - len(deduped), len(deduped))
    return deduped
