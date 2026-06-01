from __future__ import annotations

import logging
import os
from typing import Optional

_WEB_SEARCH_DISABLED = os.getenv("DISABLE_WEB_SEARCH", "").lower() in ("true", "1", "yes")

from .evidence_search import build_search_query, get_domains, tavily_search
from .groq_client import chat, parse_json
from .models import Claim, ClaimType, FormulaDefinition, ValidationResult, ValidationStatus
from .rag import rag_verify
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


VALIDATION_PROMPT = """You are a financial analyst validating a claim from a research report against web evidence.

Claim: {raw_text}
Company: {company} ({ticker})
Type: {claim_type}
Stated value: {value}
Period: {period}

Evidence:
{evidence}

Validation rules:
- VERIFIED (confidence 0.8-1.0): Evidence explicitly confirms the stated value or fact for the correct company.
- PARTIALLY_VERIFIED (confidence 0.5-0.79): Evidence is from the correct company and directionally consistent, but the exact figure is not present or is for a slightly different period.
- CONTRADICTED (confidence 0.8-1.0): Evidence explicitly shows a different value for the same metric and company.
- UNVERIFIABLE (confidence 0.0): Evidence is about a different company entirely, OR the evidence is completely unrelated to the claim.

Important:
- Non-US companies (e.g. Canadian, European) file annual reports on their local exchange — not SEC 10-K. Accept annual reports, MD&A, press releases, and financial data sites as valid evidence.
- If the evidence mentions the correct company and confirms the general direction (e.g. revenue growth, margin improvement) even without the exact number, use PARTIALLY_VERIFIED.
- Only return UNVERIFIABLE if the evidence is clearly about a different company or completely irrelevant.
- Do NOT return UNVERIFIABLE just because the exact figure is missing — use PARTIALLY_VERIFIED for directional confirmation.

Return JSON only, no markdown:
{{"status": "VERIFIED|PARTIALLY_VERIFIED|UNVERIFIABLE|CONTRADICTED", "confidence": 0.0-1.0, "reasoning": "one sentence", "actual_value": "value found or null"}}"""


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
        logger.error("_search_and_reason error for %s: %s", claim.id, exc)
        result.status = ValidationStatus.ERROR
        result.reasoning = str(exc)
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

        # 5. RAG — retrieve from 10-K text chunks, verify with LLM
        _log_step(claim.id, 5, "rag", "Atlas Vector Search on 10-K text chunks")
        rag = await rag_verify(claim)
        if rag is not None and rag.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
            _log_result(claim.id, 5, "rag", rag.status, rag.confidence)
            return rag
        _log_step(claim.id, 5, "rag", f"miss → {rag.status.value if rag else 'no chunks / index not ready'}")

    # 6. Web search (disabled when DISABLE_WEB_SEARCH=true)
    if _WEB_SEARCH_DISABLED:
        logger.info("  [%s] step 6 (web-search): disabled — returning UNVERIFIABLE", claim.id)
        result = ValidationResult(claim_id=claim.id)
        result.status = ValidationStatus.UNVERIFIABLE
        result.reasoning = "Local sources exhausted; web search is disabled."
        return result
    _log_step(claim.id, 6, "web-search", "falling back to Tavily + LLM reasoning")
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

    # 3. RAG — 10-K text chunks (e.g. segment margin tables in MD&A)
    _log_step(claim.id, 3, "rag", "Atlas Vector Search on 10-K text chunks")
    rag = await rag_verify(claim)
    if rag is not None and rag.status not in (ValidationStatus.UNVERIFIABLE, ValidationStatus.ERROR):
        _log_result(claim.id, 3, "rag", rag.status, rag.confidence)
        return rag
    _log_step(claim.id, 3, "rag", f"miss → {rag.status.value if rag else 'no chunks / index not ready'}")

    # 4. Web search (disabled when DISABLE_WEB_SEARCH=true)
    if _WEB_SEARCH_DISABLED:
        logger.info("  [%s] step 4 (web-search): disabled — returning UNVERIFIABLE", claim.id)
        result = ValidationResult(claim_id=claim.id)
        result.status = ValidationStatus.UNVERIFIABLE
        result.reasoning = "Local sources exhausted; web search is disabled."
        return result
    _log_step(claim.id, 4, "web-search", "falling back to Tavily + LLM reasoning")
    return await _search_and_reason(claim)


async def validate_accounting_policy(claim: Claim) -> ValidationResult:
    logger.info("[validate] claim=%s type=ACCOUNTING_POLICY ticker=%s metric=%r",
                claim.id, claim.ticker, claim.metric)
    result = ValidationResult(claim_id=claim.id)
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

    _validation_cache[key] = result
    return result
