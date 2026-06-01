from __future__ import annotations

import logging
import re
from typing import Optional

from .groq_client import chat, parse_json
from .models import Claim, ClaimType

logger = logging.getLogger(__name__)

VALID_TYPES = {t.value for t in ClaimType}

EXTRACTION_PROMPT = """You are a financial analyst extracting verifiable claims from a research report.

RULES:
1. Extract ALL numeric claims — every figure in every table row counts as a separate claim.
2. For markdown tables, extract one claim per data cell that contains a financial metric.
3. Extract 10-20 claims. If the report has tables, extract claims from EVERY row.
4. Copy ALL digits EXACTLY as written. Never truncate: "$391B" not "$91B", "$108.8B" not "$8.8B".

Return ONLY a JSON array (no markdown, no explanation). Each item:
- id: string (e.g. "claim_1")
- raw_text: string (exact quote from table row or sentence)
- type: one of DIRECT_FACT | DERIVED_METRIC | ACCOUNTING_POLICY | MODEL_ASSUMPTION | FORWARD_PROJECTION | RECOMMENDATION | QUALITATIVE
- company: string or null
- ticker: string or null — ALWAYS include the exchange prefix for non-US stocks (e.g. "TSX:CJT", "WSE:DOM", "LSE:VOD", "ASX:BHP"). For US stocks use plain ticker (e.g. "AAPL"). Never omit the exchange prefix for non-US companies.
- metric: string or null (e.g. "revenue", "gross margin", "P/E ratio")
- value: string or null (e.g. "$391B", "46.2%", "29x")
- period: string or null (e.g. "FY2024", "FY2023")
- checkable: boolean (true if numeric and verifiable against financial data)

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
            section_claims = await extract_claims(text)
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
    return all_claims


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
    return claims
