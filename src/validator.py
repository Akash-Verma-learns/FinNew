from __future__ import annotations

import logging
import os
import re
from typing import Optional

_WEB_SEARCH_DISABLED = os.getenv("DISABLE_WEB_SEARCH", "").lower() in ("true", "1", "yes")

from .evidence_search import build_search_query, get_domains, tavily_search
from .groq_client import chat, parse_json
from .models import Citation, Claim, ClaimType, FormulaDefinition, SourceConflict, ValidationResult, ValidationStatus
from .nongaap_parser import lookup_nongaap
from .rag import rag_verify
from .segment_parser import lookup_segment_fact
from .xbrl_lookup import (
    MARGIN_CONCEPTS,
    _compare,
    _format_value,
    _normalize_ticker,
    _parse_value,
    find_concepts,
    get_cik,
    get_latest_10k,
    lookup_derived_ratio,
    lookup_direct_fact,
    lookup_growth_rate,
    lookup_xbrl_value_match,
)

logger = logging.getLogger(__name__)

# In-memory dedup cache — keyed on (type, ticker, metric, value, period).
# Identical claims in the same report (e.g. "strong market position" repeated
# across sections) reuse the first result without a second Tavily+LLM call.
_validation_cache: dict[tuple, ValidationResult] = {}


def _cache_key(claim: Claim) -> tuple:
    return (
        claim.type,
        (claim.ticker or "").upper(),
        (claim.metric or "").lower().strip(),
        (claim.value or "").strip(),
        (claim.period or "").strip(),
    )


def clear_validation_cache() -> None:
    _validation_cache.clear()


VALIDATION_PROMPT = """You are a financial analyst validating a claim from a report against web evidence.

Claim: {raw_text}
Company: {company} ({ticker})
Type: {claim_type}
Stated value: {value}
Period: {period}

Evidence:
{evidence}

Validation rules — VERIFIED is the default when evidence confirms the company and metric:

VERIFIED (confidence 0.85-1.0): Use VERIFIED when the evidence confirms the claim for the correct company and the same metric/activity. These all count as VERIFIED:
  - The stated number appears in the evidence (even rounded: "~$75B" confirms "$75 billion")
  - Threshold language matches: "surpassed $75B", "exceeded $75B", "reached $75B", "more than $75B", "over $75B" ALL verify a stated value of "$75 billion"
  - "More than 20 million users" in evidence verifies a stated "more than 20 million users"
  - A company press release, earnings call transcript, official blog, or case study from the SAME company confirming the same customer statistic (e.g., Microsoft blog confirming Carvana's 45% call reduction)
  - Announcement claims (commitments, investments, partnerships): evidence confirming the announcement was made counts as VERIFIED for that claim
  - Set confidence 0.9 when the exact figure appears; 0.85 when confirmed via threshold/approximate language

PARTIALLY_VERIFIED (confidence 0.55-0.79): Evidence is about the same company and same metric but the specific number is MATERIALLY different (>15% off) from what is stated, OR refers to a meaningfully different time period with no equivalent current-period data available.

CONTRADICTED (confidence 0.85-1.0): ONLY use this when evidence from the SAME period explicitly states a MATERIALLY DIFFERENT value (>15% off) for the identical metric and company. Do NOT use CONTRADICTED for minor rounding differences, different-period comparisons, or when evidence uses approximate language. When in doubt between CONTRADICTED and PARTIALLY_VERIFIED, choose PARTIALLY_VERIFIED.
  MANDATORY quarterly period matching rule: If the claim has a quarterly period (contains Q1/Q2/Q3/Q4 or "first/second/third/fourth quarter"), you MUST explicitly identify the quarter in the evidence before returning CONTRADICTED. If the evidence mentions a DIFFERENT quarter (even from the same company in the same fiscal year), return UNVERIFIABLE — different quarters ALWAYS have different revenue/earnings, so they cannot contradict each other. For example: a claim of "Q1 FY2024 revenue $119.6B" CANNOT be CONTRADICTED by evidence of "Q2 FY2024 revenue $90.8B" — the quarters differ and different-quarter results are expected to differ. CONTRADICTED requires the evidence and claim to reference the IDENTICAL quarter-year combination.

UNVERIFIABLE (confidence 0.0): Evidence is about a different company, a completely different metric/activity, or a non-comparable time period (e.g., a decade-old statistic for a "this year" claim). Use UNVERIFIABLE only when the evidence cannot speak to the claim at all.

Critical rules:
- Do NOT use CONTRADICTED for threshold language mismatches ("more than 400" vs "400") — these are the same claim
- Do NOT use CONTRADICTED when evidence mentions a different fiscal year's number without also contradicting the claimed year
- For FORWARD_PROJECTION claims: if evidence confirms the commitment/investment/target was announced, return VERIFIED
- Non-US companies: accept annual reports, MD&A, press releases, and financial data sites as authoritative
- Set "actual_value" ONLY to a figure that the evidence reports for the SAME metric, in a comparable period

Return JSON only, no markdown:
{{"status": "VERIFIED|PARTIALLY_VERIFIED|UNVERIFIABLE|CONTRADICTED", "confidence": 0.0-1.0, "reasoning": "one sentence", "actual_value": "value found or null"}}"""


_NONGAAP_KEYWORDS = frozenset({
    "non-gaap", "non gaap", "adjusted", "adj.", "adj ",
    "non gaap eps", "adjusted ebitda", "adjusted earnings",
    "adjusted operating", "core earnings", "ex-items",
})

# Detect delta/change language ("increased $2.8B") vs result language ("grew to $168.9B").
# Used in validate_derived_metric to avoid comparing a YoY delta against an XBRL total.
_DELTA_VERB_RE = re.compile(r"\b(increased?|decreased?|grew|declined?|fell|rose)\b", re.IGNORECASE)
_TO_AMOUNT_RE = re.compile(r"\bto\s+\$?\s*[\d]", re.IGNORECASE)


def _is_nongaap_claim(claim: Claim) -> bool:
    metric_lower = (claim.metric or "").lower()
    return any(kw in metric_lower for kw in _NONGAAP_KEYWORDS)


def _detect_conflicts(
    primary: ValidationResult,
    secondary: Optional[ValidationResult],
    primary_source: str,
    secondary_source: str,
) -> list[SourceConflict]:
    """
    Compare two ValidationResult values. If they disagree by > 5%, return a SourceConflict.
    Both results must be non-UNVERIFIABLE and have parseable actual_value fields.
    """
    if secondary is None:
        return []
    if primary.status in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
        return []
    if secondary.status in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
        return []
    val_a_str = primary.actual_value or ""
    val_b_str = secondary.actual_value or ""
    if not val_a_str or not val_b_str:
        return []
    from .xbrl_lookup import _parse_value
    val_a = _parse_value(val_a_str)
    val_b = _parse_value(val_b_str)
    if val_a is None or val_b is None or val_a == 0:
        return []
    diff_pct = abs(val_a - val_b) / abs(val_a) * 100
    if diff_pct < 5.0:
        return []
    return [
        SourceConflict(
            source_a=primary_source,
            source_b=secondary_source,
            value_a=val_a_str,
            value_b=val_b_str,
            difference_pct=round(diff_pct, 2),
            message=(
                f"{primary_source} reports {val_a_str} but {secondary_source} reports {val_b_str} "
                f"({diff_pct:.1f}% difference). The report may be citing a non-GAAP / adjusted figure."
            ),
        )
    ]


def _log_step(claim_id: str, step: int, name: str, msg: str) -> None:
    logger.info("  [%s] step %d (%s): %s", claim_id, step, name, msg)


def _log_result(claim_id: str, step: int, name: str, status: ValidationStatus, conf: float) -> None:
    logger.info("  [%s] step %d (%s) → %s (conf=%.2f) ✓ short-circuit", claim_id, step, name, status.value, conf)


async def _search_and_reason(claim: Claim) -> ValidationResult:
    result = ValidationResult(claim_id=claim.id)
    try:
        query = build_search_query(claim)
        domains = get_domains(claim)
        logger.info("  [%s] step 6 (web-search): query=%r domains=%s", claim.id, query, domains)
        search = await tavily_search(query, domains)
        result.citations = search["citations"]
        result.evidence = search["context"][:500]
        result.structured_citations = [
            Citation(source="WEB", label="Web Search", url=url)
            for url in search["citations"]
            if url
        ]
        logger.info("  [%s] step 6 (web-search): got %d citations: %s",
                    claim.id, len(result.citations), result.citations)

        prompt = VALIDATION_PROMPT.format(
            raw_text=claim.raw_text,
            company=claim.company or "unknown",
            ticker=claim.ticker or "unknown",
            claim_type=claim.type,
            value=claim.value or "not specified",
            period=claim.period or "not specified",
            evidence=search["context"][:4000],
        )
        raw = await chat(prompt)
        data = parse_json(raw)
        raw_status = data.get("status", "UNVERIFIABLE")
        try:
            result.status = ValidationStatus(raw_status)
        except ValueError:
            result.status = ValidationStatus.UNVERIFIABLE
        result.confidence = float(data.get("confidence", 0.5))
        result.reasoning = data.get("reasoning", "")
        result.actual_value = data.get("actual_value")
        logger.info("  [%s] step 6 (web-search) → %s (conf=%.2f)", claim.id, result.status.value, result.confidence)
    except Exception as exc:
        logger.warning("_search_and_reason error for %s: %s — returning UNVERIFIABLE", claim.id, exc)
        result.status = ValidationStatus.UNVERIFIABLE
        result.reasoning = f"Web search unavailable: {exc}"
    return result


async def validate_direct_fact(claim: Claim) -> ValidationResult:
    logger.info("[validate] claim=%s type=DIRECT_FACT ticker=%s metric=%r value=%r period=%r",
                claim.id, claim.ticker, claim.metric, claim.value, claim.period)

    if claim.ticker:
        # 1. Direct XBRL concept lookup (standard GAAP concepts)
        _log_step(claim.id, 1, "xbrl-direct", f"looking up {claim.metric!r} for {claim.ticker}")
        result = await lookup_direct_fact(claim)
        if result.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
            _log_result(claim.id, 1, "xbrl-direct", result.status, result.confidence)
            # If the claim may be quoting a non-GAAP figure, cross-check 8-K for conflicts
            if _is_nongaap_claim(claim):
                nongaap_check = await lookup_nongaap(claim)
                result.source_conflicts = _detect_conflicts(
                    result, nongaap_check, "SEC EDGAR XBRL (GAAP)", "8-K Non-GAAP"
                )
            return result
        _log_step(claim.id, 1, "xbrl-direct", f"miss → {result.status.value} — trying next")

        # 2. Pre-computed FormulaGraph (derivation_log in MongoDB)
        _log_step(claim.id, 2, "formula-graph-db", f"derivation_log lookup for metric={claim.metric!r}")
        derived = await _lookup_derived_from_db(claim)
        if derived is not None:
            _log_result(claim.id, 2, "formula-graph-db", derived.status, derived.confidence)
            return derived
        _log_step(claim.id, 2, "formula-graph-db", "miss (no concept mapping or no DB entry)")

        # 3. Live ratio computation for margin/ratio metrics
        if claim.metric:
            _log_step(claim.id, 3, "live-ratio", f"computing {claim.metric!r} live from XBRL")
            live = await lookup_derived_ratio(claim)
            if live.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
                _log_result(claim.id, 3, "live-ratio", live.status, live.confidence)
                return live
            _log_step(claim.id, 3, "live-ratio", f"miss → {live.status.value}")

        # 4. Value-match scan across all XBRL namespaces
        if claim.value:
            _log_step(claim.id, 4, "value-match", f"scanning all XBRL namespaces for {claim.value!r}")
            match = await lookup_xbrl_value_match(claim)
            if match.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
                _log_result(claim.id, 4, "value-match", match.status, match.confidence)
                return match
            _log_step(claim.id, 4, "value-match", "no match within 1.5% tolerance")

        # 5. Non-GAAP lookup — 8-K earnings release metrics (adjusted EPS, EBITDA, etc.)
        _log_step(claim.id, 5, "nongaap", f"checking non_gaap_metrics DB for {claim.metric!r}")
        nongaap = await lookup_nongaap(claim)
        if nongaap is not None and nongaap.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
            _log_result(claim.id, 5, "nongaap", nongaap.status, nongaap.confidence)
            # Contradiction check: compare 8-K non-GAAP against any prior XBRL result
            xbrl_check = await lookup_direct_fact(claim)
            nongaap.source_conflicts = _detect_conflicts(nongaap, xbrl_check, "8-K Non-GAAP", "SEC EDGAR XBRL")
            return nongaap
        _log_step(claim.id, 5, "nongaap", f"miss → {nongaap.status.value if nongaap else 'no data'}")

        # 6. Segment lookup — ASC 280 segment-level facts from 10-K notes
        _log_step(claim.id, 6, "segment", f"checking segment_facts DB for {claim.metric!r}")
        seg = await lookup_segment_fact(claim)
        if seg is not None and seg.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
            _log_result(claim.id, 6, "segment", seg.status, seg.confidence)
            return seg
        _log_step(claim.id, 6, "segment", f"miss → {seg.status.value if seg else 'no data'}")

        # 7. RAG — retrieve from 10-K text chunks, verify with LLM
        _log_step(claim.id, 7, "rag", "Atlas Vector Search on 10-K text chunks")
        rag = await rag_verify(claim)
        if rag is not None and rag.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
            _log_result(claim.id, 7, "rag", rag.status, rag.confidence)
            return rag
        _log_step(claim.id, 7, "rag", f"miss → {rag.status.value if rag else 'no chunks / index not ready'}")

    # 8. Web search (disabled when DISABLE_WEB_SEARCH=true)
    if _WEB_SEARCH_DISABLED:
        logger.info("  [%s] step 8 (web-search): disabled — returning UNVERIFIABLE", claim.id)
        result = ValidationResult(claim_id=claim.id)
        result.status = ValidationStatus.UNVERIFIABLE
        result.reasoning = "Local sources exhausted; web search is disabled."
        return result
    _log_step(claim.id, 8, "web-search", "falling back to Tavily + LLM reasoning")
    return await _search_and_reason(claim)


async def _lookup_derived_from_db(claim: Claim) -> Optional[ValidationResult]:
    """
    Try derivation_log first (pre-computed via FormulaGraph), then fall back to
    live ratio computation. Returns None if DB is unavailable or no entry found.
    """
    try:
        from .db import collections, is_connected
        if not is_connected() or not claim.ticker or not claim.metric:
            return None

        from .xbrl_lookup import _normalize_ticker
        ticker = _normalize_ticker(claim.ticker)
        cik_doc = await collections.ticker_lookup.find_one({"_id": ticker})
        if not cik_doc:
            return None
        cik = cik_doc["cik"]

        # Map claim.metric to the FormulaGraph concept name
        metric_lower = claim.metric.lower()
        concept_map = {
            "gross margin": "GrossMarginPct",
            "operating margin": "OperatingMarginPct",
            "net margin": "NetMarginPct",
            "ebitda margin": "EBITDAMarginPct",
            "free cash flow": "FreeCashFlow",
            "net debt": "NetDebt",
            "working capital": "WorkingCapital",
            "current ratio": "CurrentRatio",
            "debt to equity": "DebtToEquity",
            "interest coverage": "InterestCoverage",
            "roe": "ROE",
            "roa": "ROA",
            "ebitda": "EBITDA",
        }
        concept = next((v for k, v in concept_map.items() if k in metric_lower), None)
        if not concept:
            return None

        query: dict = {"cik": cik, "concept": concept}
        if claim.period:
            import re as _re
            year = _re.search(r"20\d{2}", claim.period)
            if year:
                query["period_end"] = {"$regex": year.group()}

        logger.info("  [%s] formula-graph-db: querying derivation_log concept=%s cik=%s", claim.id, concept, cik)
        doc = await collections.derivation_log.find_one(query, sort=[("period_end", -1)])
        if not doc or doc.get("value") is None:
            logger.info("  [%s] formula-graph-db: no derivation_log entry for concept=%s", claim.id, concept)
            return None

        actual_val = doc["value"]
        period_end = doc["period_end"]
        formula_src = doc.get("formula_source", "textbook")
        inputs: dict = doc.get("inputs_used", {})
        logger.info("  [%s] formula-graph-db: HIT concept=%s value=%s period=%s source=%s inputs=%s",
                    claim.id, concept, actual_val, period_end, formula_src,
                    {k: f"{v:,.0f}" for k, v in inputs.items()})

        stated = _parse_value(claim.value or "")
        if stated is None:
            return None

        is_pct = concept.endswith("Pct") or "margin" in metric_lower

        if is_pct:
            diff = abs(actual_val - stated)
            discrepancy_str = f"{diff:.2f}pp difference"
            if diff < 0.3:
                status, conf = ValidationStatus.VERIFIED, 0.97
            elif diff < 1.0:
                status, conf = ValidationStatus.VERIFIED, 0.90
            elif diff < 2.0:
                status, conf = ValidationStatus.PARTIALLY_VERIFIED, 0.75
            else:
                status, conf = ValidationStatus.CONTRADICTED, 0.90
        else:
            pct = abs(actual_val - stated) / abs(actual_val) if actual_val != 0 else 0
            discrepancy_str = f"{pct * 100:.2f}% difference"
            if pct < 0.005:
                status, conf = ValidationStatus.VERIFIED, 0.99
            elif pct < 0.02:
                status, conf = ValidationStatus.VERIFIED, 0.95
            elif pct < 0.05:
                status, conf = ValidationStatus.PARTIALLY_VERIFIED, 0.80
            elif pct < 0.15:
                status, conf = ValidationStatus.PARTIALLY_VERIFIED, 0.60
            else:
                status, conf = ValidationStatus.CONTRADICTED, 0.95

        formatted_val = f"{actual_val:.2f}%" if is_pct else f"{actual_val:,.2f}"

        _source_labels = {
            "company_10k": "Company 10-K",
            "fasb_linkbase": "FASB Linkbase",
            "textbook": "Textbook / CFA",
        }
        raw_defs: list[dict] = doc.get("all_definitions", [])
        all_defs = [
            FormulaDefinition(
                source=d["source"],
                value=d["value"],
                label=_source_labels.get(d["source"], d["source"]),
            )
            for d in raw_defs
        ]
        delta = doc.get("delta")

        result = ValidationResult(claim_id=claim.id)
        result.status = status
        result.confidence = conf
        result.actual_value = formatted_val
        result.discrepancy = discrepancy_str
        result.filing_source = f"MongoDB derivation_log — {concept}, period ending {period_end}"
        result.cik = cik
        result.formula_concept = concept
        result.formula_source = formula_src
        result.formula_inputs = inputs
        result.all_definitions = all_defs
        result.definition_delta = delta if delta is not None else None
        result.reasoning = (
            f"Claimed: {claim.value} | Computed (FormulaGraph/{_source_labels.get(formula_src, formula_src)}): "
            f"{formatted_val} [{', '.join(f'{k}={v:,.0f}' for k, v in inputs.items())}]"
        )
        return result
    except Exception as exc:
        logger.debug("derivation_log lookup skipped (%s) — falling back", exc)
        return None


async def validate_derived_metric(claim: Claim) -> ValidationResult:
    logger.info("[validate] claim=%s type=DERIVED_METRIC ticker=%s metric=%r value=%r period=%r",
                claim.id, claim.ticker, claim.metric, claim.value, claim.period)

    # 1. Pre-computed derivation_log (FormulaGraph, DB)
    _log_step(claim.id, 1, "formula-graph-db", f"derivation_log lookup for metric={claim.metric!r}")
    cached = await _lookup_derived_from_db(claim)
    if cached is not None:
        _log_result(claim.id, 1, "formula-graph-db", cached.status, cached.confidence)
        return cached
    _log_step(claim.id, 1, "formula-graph-db", "miss")

    # 2. Live ratio computation from XBRL
    if claim.ticker and claim.metric:
        _log_step(claim.id, 2, "live-ratio", f"computing {claim.metric!r} live from XBRL")
        result = await lookup_derived_ratio(claim)
        if result.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
            _log_result(claim.id, 2, "live-ratio", result.status, result.confidence)
            return result
        _log_step(claim.id, 2, "live-ratio", f"miss → {result.status.value}")

    # 3. YoY growth rate computation — for percentage claims about income-statement
    # line items (e.g. "Revenue grew 15%", "Operating income up 17%").
    # lookup_derived_ratio handles margin levels; this step handles growth rates.
    # It fetches two consecutive annual XBRL periods and computes % change.
    _value_str = (claim.value or "").strip()
    _looks_like_pct = "%" in _value_str
    if claim.ticker and claim.metric and claim.value and _looks_like_pct:
        _log_step(claim.id, 3, "growth-rate", f"computing YoY growth for {claim.metric!r}")
        growth = await lookup_growth_rate(claim)
        if growth.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
            _log_result(claim.id, 3, "growth-rate", growth.status, growth.confidence)
            return growth
        _log_step(claim.id, 3, "growth-rate", f"miss → {growth.status.value}")

    # 4. XBRL direct/value-match — for DERIVED_METRIC claims that carry an
    # absolute dollar value (e.g. "Microsoft Cloud revenue increased 23% to
    # $168.9 billion"), run the full XBRL pipeline so the label/embedding
    # matching and value-scan can find the segment concept in EDGAR.
    # Percentage-only claims (value ends with %) are skipped — XBRL doesn't
    # store year-over-year growth rates as a direct fact.
    # Delta-claim guard: "Cost of revenue increased $2.8B or 14%" describes a
    # YoY change, not the total EDGAR stores.  The LLM strips "increased" from
    # the metric ("cost of revenue") so it gets an exact CONCEPT_MAP hit and
    # compares the $2.8B delta against the ~$88B total — a confident false
    # CONTRADICTED.  Skip XBRL direct when raw_text has increase/decrease
    # language but no "to $X" phrase (which marks the resulting total, not a delta).
    _is_delta_claim = (
        bool(claim.raw_text and _DELTA_VERB_RE.search(claim.raw_text))
        and not bool(claim.raw_text and _TO_AMOUNT_RE.search(claim.raw_text))
    )
    if claim.ticker and claim.value and not _looks_like_pct and not _is_delta_claim:
        _log_step(claim.id, 4, "xbrl-direct", f"derived metric has absolute value {claim.value!r} — trying XBRL")
        xbrl_result = await lookup_direct_fact(claim)
        if xbrl_result.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
            _log_result(claim.id, 4, "xbrl-direct", xbrl_result.status, xbrl_result.confidence)
            return xbrl_result
        _log_step(claim.id, 4, "xbrl-direct", f"miss → {xbrl_result.status.value}")

        # Segment/extension fallback: company-specific XBRL concepts (e.g.
        # "Microsoft Cloud revenue") live outside us-gaap, so lookup_direct_fact
        # won't find them. Value-match scans ALL namespaces by value.
        _log_step(claim.id, 4, "xbrl-value-match", f"scanning all XBRL namespaces for {claim.value!r}")
        vm_result = await lookup_xbrl_value_match(claim)
        if vm_result.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
            _log_result(claim.id, 4, "xbrl-value-match", vm_result.status, vm_result.confidence)
            return vm_result
        _log_step(claim.id, 4, "xbrl-value-match", "no match within tolerance")

    # 5. Non-GAAP lookup — adjusted/non-GAAP derived metrics from 8-K
    _log_step(claim.id, 5, "nongaap", f"checking non_gaap_metrics DB for {claim.metric!r}")
    nongaap = await lookup_nongaap(claim)
    if nongaap is not None and nongaap.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
        _log_result(claim.id, 5, "nongaap", nongaap.status, nongaap.confidence)
        return nongaap
    _log_step(claim.id, 5, "nongaap", f"miss → {nongaap.status.value if nongaap else 'no data'}")

    # 6. Segment lookup — segment-level metrics (e.g. Services margin, AWS income)
    _log_step(claim.id, 6, "segment", f"checking segment_facts DB for {claim.metric!r}")
    seg = await lookup_segment_fact(claim)
    if seg is not None and seg.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
        _log_result(claim.id, 6, "segment", seg.status, seg.confidence)
        return seg
    _log_step(claim.id, 6, "segment", f"miss → {seg.status.value if seg else 'no data'}")

    # 7. RAG — 10-K text chunks (e.g. segment margin tables in MD&A)
    _log_step(claim.id, 7, "rag", "Atlas Vector Search on 10-K text chunks")
    rag = await rag_verify(claim)
    if rag is not None and rag.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
        _log_result(claim.id, 7, "rag", rag.status, rag.confidence)
        return rag
    _log_step(claim.id, 7, "rag", f"miss → {rag.status.value if rag else 'no chunks / index not ready'}")

    # 8. Web search (disabled when DISABLE_WEB_SEARCH=true)
    if _WEB_SEARCH_DISABLED:
        logger.info("  [%s] step 8 (web-search): disabled — returning UNVERIFIABLE", claim.id)
        result = ValidationResult(claim_id=claim.id)
        result.status = ValidationStatus.UNVERIFIABLE
        result.reasoning = "Local sources exhausted; web search is disabled."
        return result
    _log_step(claim.id, 8, "web-search", "falling back to Tavily + LLM reasoning")
    return await _search_and_reason(claim)


async def validate_accounting_policy(claim: Claim) -> ValidationResult:
    logger.info("[validate] claim=%s type=ACCOUNTING_POLICY ticker=%s metric=%r",
                claim.id, claim.ticker, claim.metric)
    result = ValidationResult(claim_id=claim.id)

    # Web search (disabled when DISABLE_WEB_SEARCH=true)
    if _WEB_SEARCH_DISABLED:
        logger.info("  [%s] sec-search: disabled — returning UNVERIFIABLE", claim.id)
        result.status = ValidationStatus.UNVERIFIABLE
        result.reasoning = "Local sources exhausted; web search is disabled."
        return result

    try:
        filing_info = await get_latest_10k(claim.ticker) if claim.ticker else None
        query = build_search_query(claim)
        domains = get_domains(claim)
        logger.info("  [%s] step 1 (sec-search): query=%r domains=%s", claim.id, query, domains)
        search = await tavily_search(query, domains)
        result.citations = search["citations"]
        logger.info("  [%s] step 1 (sec-search): citations=%s", claim.id, result.citations)
        if filing_info:
            result.filing_source = filing_info["url"]
            if filing_info["url"] not in result.citations:
                result.citations.append(filing_info["url"])

        prompt = VALIDATION_PROMPT.format(
            raw_text=claim.raw_text,
            company=claim.company or "unknown",
            ticker=claim.ticker or "unknown",
            claim_type=claim.type,
            value=claim.value or "not specified",
            period=claim.period or "not specified",
            evidence=search["context"][:4000],
        )
        raw = await chat(prompt)
        data = parse_json(raw)
        raw_status = data.get("status", "UNVERIFIABLE")
        try:
            result.status = ValidationStatus(raw_status)
        except ValueError:
            result.status = ValidationStatus.UNVERIFIABLE
        result.confidence = float(data.get("confidence", 0.5))
        result.reasoning = data.get("reasoning", "")
        result.actual_value = data.get("actual_value")
        logger.info("  [%s] step 2 (llm) → %s (conf=%.2f)", claim.id, result.status.value, result.confidence)
    except Exception as exc:
        logger.error("validate_accounting_policy error for %s: %s", claim.id, exc)
        result.status = ValidationStatus.ERROR
        result.reasoning = str(exc)
    return result


async def validate_claim(claim: Claim) -> ValidationResult:
    key = _cache_key(claim)
    if key in _validation_cache:
        cached = _validation_cache[key]
        logger.info("  [%s] cache hit → reusing result from earlier claim (status=%s)", claim.id, cached.status.value)
        result = cached.model_copy()
        result.claim_id = claim.id
        return result

    if claim.type == ClaimType.DIRECT_FACT:
        result = await validate_direct_fact(claim)
    elif claim.type == ClaimType.DERIVED_METRIC:
        result = await validate_derived_metric(claim)
    elif claim.type == ClaimType.ACCOUNTING_POLICY:
        result = await validate_accounting_policy(claim)
    elif _WEB_SEARCH_DISABLED:
        logger.info("  [%s] web-search disabled — returning UNVERIFIABLE for type=%s", claim.id, claim.type)
        result = ValidationResult(claim_id=claim.id)
        result.status = ValidationStatus.UNVERIFIABLE
        result.reasoning = "Web search is disabled; claim type requires external sources."
    else:
        logger.info("[validate] claim=%s type=%s — routing to web-search", claim.id, claim.type)
        result = await _search_and_reason(claim)

    # QUALITATIVE claims (MAU, subscriber counts, product metrics) are non-GAAP
    # operational figures that vary by scope, period, and source methodology.
    # A web search finding a different number from a different period or scope
    # is NOT a definitive contradiction — cap at PARTIALLY_VERIFIED.
    if result.status == ValidationStatus.CONTRADICTED and claim.type == ClaimType.QUALITATIVE:
        result.status = ValidationStatus.PARTIALLY_VERIFIED
        result.confidence = min(result.confidence, 0.60)
        result.reasoning = (result.reasoning or "").rstrip(".") + ". Non-GAAP product metric; conflicting source may reflect different scope or period — capped at PARTIALLY_VERIFIED."
        logger.info("  [%s] qualitative cap: CONTRADICTED → PARTIALLY_VERIFIED (conf=%.2f)", claim.id, result.confidence)

    # Quarterly DIRECT_FACT/DERIVED_METRIC claims frequently trigger false
    # CONTRADICTEDs: web search returns a different quarter's results and the LLM
    # treats them as a same-period contradiction. Guard: parse both values once,
    # classify into three zones:
    #   < 2×   → normal quarter-to-quarter variance → cap to PARTIALLY_VERIFIED
    #   2–10×  → too large for quarter variance, plausible fraud range → keep CONTRADICTED
    #   > 10×  → Tavily found a completely different metric (e.g. $3.9B for iPhone
    #             revenue is clearly wrong data) → irrelevant evidence, cap
    import re as _re
    if (
        result.status == ValidationStatus.CONTRADICTED
        and claim.type in (ClaimType.DIRECT_FACT, ClaimType.DERIVED_METRIC)
        and claim.period
        and _re.search(r"\bQ[1-4]\b|\b(first|second|third|fourth)\s+quarter\b",
                       str(claim.period), _re.IGNORECASE)
    ):
        _ratio: float | None = None
        if result.actual_value and claim.value:
            try:
                def _pv(s: str) -> float | None:
                    s = str(s).lower().replace(",", "")
                    m = _re.search(r"([\d.]+)\s*([tb])", s)
                    if m:
                        v = float(m.group(1))
                        return v * (1_000_000_000_000 if m.group(2) == "t" else 1_000_000_000)
                    m = _re.search(r"[\d.]+", s)
                    return float(m.group()) if m else None
                _av, _cv = _pv(result.actual_value), _pv(claim.value)
                if _av and _cv and _av > 0 and _cv > 0:
                    _ratio = max(_av, _cv) / min(_av, _cv)
            except Exception:
                pass

        _in_fraud_range = _ratio is not None and 2.0 < _ratio < 10.0
        if not _in_fraud_range:
            # Grade confidence by magnitude match:
            # ≤1.5× → different quarter, same ballpark → 0.75
            # >1.5× or no comparison → more uncertain → 0.70
            _qcap_conf = 0.75 if (_ratio is not None and _ratio <= 1.5) else 0.70
            result.status = ValidationStatus.PARTIALLY_VERIFIED
            result.confidence = min(result.confidence, _qcap_conf)
            result.reasoning = (result.reasoning or "").rstrip(".") + ". Quarterly claim — evidence may cover a different quarter; different quarters always differ. Capped at PARTIALLY_VERIFIED."
            logger.info("  [%s] quarterly cap: CONTRADICTED → PARTIALLY_VERIFIED (conf=%.2f, ratio=%s)",
                        claim.id, result.confidence, f"{_ratio:.1f}×" if _ratio else "n/a")

    # Grounding pass: if all sources exhausted with UNVERIFIABLE and the claim is
    # from the company's own official document (has a ticker), no contradicting
    # evidence was found — treat it as grounded rather than uncertain.
    # "Couldn't verify" ≠ "wrong" for a company's own disclosures.
    _GROUNDING_CONFIDENCE = {
        ClaimType.DIRECT_FACT:        0.75,
        ClaimType.DERIVED_METRIC:     0.74,
        ClaimType.QUALITATIVE:        0.68,
        ClaimType.FORWARD_PROJECTION: 0.65,
        ClaimType.ACCOUNTING_POLICY:  0.63,
    }
    if (
        result.status == ValidationStatus.UNVERIFIABLE
        and claim.ticker
        and claim.checkable
        and claim.type in _GROUNDING_CONFIDENCE
    ):
        conf = _GROUNDING_CONFIDENCE[claim.type]
        result.status = ValidationStatus.PARTIALLY_VERIFIED
        result.confidence = conf
        result.reasoning = (
            (result.reasoning.rstrip(".") + ". " if result.reasoning else "")
            + "No contradicting evidence found; official source statement treated as grounded."
        )
        logger.info("  [%s] grounding pass: UNVERIFIABLE → PARTIALLY_VERIFIED (conf=%.2f)", claim.id, conf)

    _validation_cache[key] = result
    return result
