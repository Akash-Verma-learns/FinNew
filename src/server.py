from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from .claim_extractor import extract_claims, extract_claims_from_sections
from .db import close_db, collections, init_db, is_connected
from .embeddings import load_model as load_embedding_model
from .models import Claim, ValidationResult, ValidationStatus
from .pdf_parser import extract_text_from_pdf, prepare_text
from .scorer import score_report
from .validator import clear_validation_cache, validate_claim


async def _bg_load_model() -> None:
    try:
        await asyncio.get_event_loop().run_in_executor(None, load_embedding_model)
        logger.info("Embedding model ready")
    except Exception as exc:
        logger.warning("Embedding model failed to load: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FinValidator v4.0 starting on port %s", os.getenv("PORT", "8000"))
    try:
        await init_db()
    except Exception as exc:
        logger.warning("MongoDB unavailable — running without DB cache: %s", exc)
    asyncio.create_task(_bg_load_model())
    yield
    await close_db()
    logger.info("FinValidator shutting down")


app = FastAPI(title="FinValidator", version="4.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Single-session guard: only one validation pipeline may run at a time.
# Auto-expires after SESSION_TTL seconds so a crashed client never bricks the server.
SESSION_TTL = 300  # 5 minutes — enough for 8 claims × 22s each with headroom
_session_active: bool = False
_session_started: float = 0.0


def _check_and_claim_session() -> None:
    global _session_active, _session_started
    now = time.monotonic()
    if _session_active and (now - _session_started) < SESSION_TTL:
        elapsed = int(now - _session_started)
        remaining = max(0, SESSION_TTL - elapsed)
        raise HTTPException(
            status_code=503,
            detail=f"Server is already processing a report (started {elapsed}s ago, ~{remaining}s left). "
                   "Please wait for it to finish or click 'New Report' to cancel.",
        )
    _session_active = True
    _session_started = time.monotonic()
    logger.info("Session started")


def _release_session() -> None:
    global _session_active
    _session_active = False
    logger.info("Session released")


class AnalyzeRequest(BaseModel):
    report: str


async def _run_pipeline_with_claims(claims) -> dict:
    """Run validation + scoring for pre-extracted claims (PageIndex / Docling path)."""
    from .models import ValidationResult, ValidationStatus
    clear_validation_cache()
    vid = str(uuid.uuid4())
    logger.info("Validating %d pre-extracted claims (id=%s)...", len(claims), vid)

    t0 = time.monotonic()
    raw_results = await asyncio.gather(*[validate_claim(c) for c in claims], return_exceptions=True)
    t_validate = time.monotonic() - t0

    validations: dict[str, ValidationResult] = {}
    for claim, result in zip(claims, raw_results):
        if isinstance(result, Exception):
            logger.error("Validation error for %s: %s", claim.id, result)
            validations[claim.id] = ValidationResult(
                claim_id=claim.id,
                status=ValidationStatus.ERROR,
                reasoning=str(result),
            )
        else:
            validations[claim.id] = result

    t_score_start = time.monotonic()
    scoring = score_report(claims, validations)
    t_score = time.monotonic() - t_score_start

    return {
        **scoring,
        "claims": [c.model_dump() for c in claims],
        "validations": {k: v.model_dump() for k, v in validations.items()},
        "_claims_internal": claims,
        "_validation_id": vid,
        "_timings": {
            "extraction_seconds": 0.0,   # extraction done outside this function
            "validation_seconds": round(t_validate, 2),
            "scoring_seconds": round(t_score, 2),
            "total_seconds": round(time.monotonic() - t0, 2),
        },
    }


async def _run_pipeline(text: str) -> dict:
    clear_validation_cache()
    vid = str(uuid.uuid4())
    logger.info("Extracting claims from %d chars (id=%s)", len(text), vid)

    t0 = time.monotonic()
    claims = await extract_claims(text)
    t_extract = time.monotonic() - t0
    logger.info("Extracted %d claims — validating...", len(claims))

    t_val_start = time.monotonic()
    raw_results = await asyncio.gather(*[validate_claim(c) for c in claims], return_exceptions=True)
    t_validate = time.monotonic() - t_val_start

    validations: dict[str, ValidationResult] = {}
    for claim, result in zip(claims, raw_results):
        if isinstance(result, Exception):
            logger.error("Validation error for %s: %s", claim.id, result)
            validations[claim.id] = ValidationResult(
                claim_id=claim.id,
                status=ValidationStatus.ERROR,
                reasoning=str(result),
            )
        else:
            validations[claim.id] = result

    t_score_start = time.monotonic()
    scoring = score_report(claims, validations)
    t_score = time.monotonic() - t_score_start

    return {
        **scoring,
        "claims": [c.model_dump() for c in claims],
        "validations": {k: v.model_dump() for k, v in validations.items()},
        "_claims_internal": claims,
        "_validation_id": vid,
        "_timings": {
            "extraction_seconds": round(t_extract, 2),
            "validation_seconds": round(t_validate, 2),
            "scoring_seconds": round(t_score, 2),
            "total_seconds": round(time.monotonic() - t0, 2),
        },
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "4.0.0"}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.report or not req.report.strip():
        raise HTTPException(status_code=400, detail="report field is required")
    result = await _run_pipeline(req.report.strip())
    claims_internal = result.pop("_claims_internal", [])
    timings = result.pop("_timings", {})
    vid = result.pop("_validation_id", None)
    validations_list = list(result["validations"].values())
    scoring = {k: v for k, v in result.items() if k not in ("claims", "validations")}
    from .audit import save_audit_log
    await save_audit_log(
        validation_id=vid,
        input_text=req.report.strip(),
        input_type="text",
        ticker=None,
        claims=claims_internal,
        validations=validations_list,
        scoring=scoring,
        timings=timings,
    )
    result["validation_id"] = vid
    return result


def _detect_ticker_from_claims(claims: list):
    """Return the most common non-None ticker found in extracted claims, or None."""
    from collections import Counter
    tickers = [
        (getattr(c, "ticker", None) or (c.get("ticker") if isinstance(c, dict) else None))
        for c in claims
    ]
    counts = Counter(t for t in tickers if t)
    return counts.most_common(1)[0][0].upper() if counts else None


@app.post("/api/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...), ticker: str = Form("")):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="A .pdf file is required")
    data = await file.read()
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 30 MB)")

    _, page_count = extract_text_from_pdf(data)
    logger.info("PDF: %d pages — converting with Docling", page_count)

    result: dict = {}
    claims_for_trend: list = []
    _docling_ok = False
    _ticker = ticker.strip().upper() if ticker.strip() else None

    # Primary: Docling → structured markdown → section-aware claim extraction
    try:
        from .docling_parser import process_pdf_bytes, split_sections
        markdown = await asyncio.get_event_loop().run_in_executor(
            None, process_pdf_bytes, data, file.filename
        )
        sections = split_sections(markdown)
        logger.info("[docling] %d sections, %d chars from %s", len(sections), len(markdown), file.filename)
        claims = await extract_claims_from_sections(sections)
        if not _ticker:
            _ticker = _detect_ticker_from_claims(claims)
            if _ticker:
                logger.info("[pdf] auto-detected ticker from claims: %s", _ticker)
        result = await _run_pipeline_with_claims(claims)
        result["pdf_truncated"] = False
        result["pdf_truncation_warning"] = None
        result["indexer"] = "docling"
        result["sections_found"] = list(sections.keys())
        claims_for_trend = result.pop("_claims_internal", [])
        timings = result.pop("_timings", {})
        vid = result.pop("_validation_id", None)
        validations_list = list(result["validations"].values())
        scoring = {k: v for k, v in result.items() if k not in ("claims", "validations")}
        from .audit import save_audit_log
        await save_audit_log(
            validation_id=vid,
            input_text=file.filename or "",
            input_type="pdf",
            ticker=_ticker,
            claims=claims_for_trend,
            validations=validations_list,
            scoring=scoring,
            timings=timings,
        )
        result["validation_id"] = vid
        _docling_ok = True
    except Exception as exc:
        logger.warning("Docling unavailable (%s) — falling back to flat text", exc)

    if not _docling_ok:
        # Fallback: flat pdfplumber text with truncation
        full_text, _ = extract_text_from_pdf(data)
        text, truncated, warning = prepare_text(full_text)
        logger.info("Flat fallback: %d chars (truncated=%s)", len(text), truncated)
        result = await _run_pipeline(text)
        result["pdf_truncated"] = truncated
        result["pdf_truncation_warning"] = warning
        result["indexer"] = "flat"
        claims_for_trend = result.pop("_claims_internal", [])
        if not _ticker:
            _ticker = _detect_ticker_from_claims(claims_for_trend)
            if _ticker:
                logger.info("[pdf] auto-detected ticker (flat): %s", _ticker)
        timings = result.pop("_timings", {})
        vid = result.pop("_validation_id", None)
        validations_list = list(result["validations"].values())
        scoring = {k: v for k, v in result.items() if k not in ("claims", "validations")}
        from .audit import save_audit_log as _save_audit
        await _save_audit(
            validation_id=vid,
            input_text=file.filename or "",
            input_type="pdf",
            ticker=_ticker,
            claims=claims_for_trend,
            validations=validations_list,
            scoring=scoring,
            timings=timings,
        )
        result["validation_id"] = vid

    # Trend enrichment — only when a ticker is provided
    if _ticker:
        try:
            from .historical_claims import save_snapshot
            from .trend_engine import build_trend_insight
            _vlist = list(result["validations"].values())
            _scoring = {k: v for k, v in result.items() if k not in ("claims", "validations", "validation_id")}
            await save_snapshot(
                ticker=_ticker,
                score_result=_scoring,
                claims=claims_for_trend,
                validations=_vlist,
                source_label=f"PDF {file.filename or 'upload'}",
            )
            trend = await build_trend_insight(
                ticker=_ticker,
                validations=_vlist,
                claims=claims_for_trend,
                scoring=_scoring,
            )
            result["trend_insight"] = trend.model_dump()
        except Exception as exc:
            logger.warning("Trend enrichment failed for PDF ticker=%s: %s", _ticker, exc)

    return result


@app.post("/api/extract")
async def extract_only(req: AnalyzeRequest):
    if not req.report or not req.report.strip():
        raise HTTPException(status_code=400, detail="report field is required")
    _check_and_claim_session()
    try:
        claims = await extract_claims(req.report.strip())
        return {"claims": [c.model_dump() for c in claims]}
    except Exception:
        _release_session()
        raise


class ValidateRequest(BaseModel):
    claim: dict


@app.post("/api/validate")
async def validate_single(req: ValidateRequest):
    claim = Claim.model_validate(req.claim)
    result = await validate_claim(claim)
    return result.model_dump()


@app.post("/api/session/done")
async def session_done():
    """Frontend calls this when all /api/validate calls for a session are complete."""
    _release_session()
    return {"ok": True}


@app.post("/api/session/cancel")
async def session_cancel():
    """Frontend calls this when the user clicks 'New Report' mid-run."""
    _release_session()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Ingestion endpoint — pre-loads a ticker's XBRL data into MongoDB
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    ticker: str


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    if not req.ticker or not req.ticker.strip():
        raise HTTPException(status_code=400, detail="ticker field is required")
    if not is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected — cannot ingest")
    try:
        from .ingestion import ingest_ticker
        summary = await ingest_ticker(req.ticker.strip())
        return summary
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Ingestion error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/db/status")
async def db_status():
    return {"connected": is_connected()}


class IngestTextRequest(BaseModel):
    ticker: str
    period: str = ""   # e.g. "2024-09-28" — defaults to most recent 10-K


@app.post("/api/ingest-text")
async def ingest_text_endpoint(req: IngestTextRequest):
    """Fetch the 10-K HTML, chunk, embed, and store in text_chunks for RAG."""
    if not req.ticker.strip():
        raise HTTPException(status_code=400, detail="ticker is required")
    if not is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected")
    try:
        from .text_ingestion import ingest_text
        from .xbrl_lookup import _normalize_ticker, get_cik
        ticker = _normalize_ticker(req.ticker.strip())
        cik = await get_cik(ticker)
        if not cik:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found in EDGAR")
        result = await ingest_text(ticker, cik, req.period.strip())
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ingest-text error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/formula-graph")
async def formula_graph_endpoint(ticker: str = "", period: str = ""):
    """
    Returns the full FormulaGraph as nodes + edges for frontend rendering.
    Optional ?ticker=AAPL&period=2024-09-28 annotates nodes with actual values
    from derivation_log if they exist in the DB.
    """
    from .formula_graph import DEFAULT_GRAPH, graph_to_json

    active_concepts: set[str] = set()
    node_values: dict[str, str] = {}

    if ticker and is_connected():
        try:
            from .xbrl_lookup import _normalize_ticker
            from pymongo import DESCENDING
            norm = _normalize_ticker(ticker)
            cik_doc = await collections.xbrl_facts.find_one({"ticker": norm})
            if not cik_doc:
                cik_doc = await collections.ticker_lookup.find_one({"_id": norm})
            cik = cik_doc["cik"] if cik_doc else None

            if cik:
                query: dict = {"cik": cik}
                if period:
                    query["period_end"] = period
                cursor = collections.derivation_log.find(
                    query, sort=[("period_end", DESCENDING)]
                )
                seen_concepts: set[str] = set()
                async for doc in cursor:
                    c = doc["concept"]
                    if c not in seen_concepts:
                        seen_concepts.add(c)
                        active_concepts.add(c)
                        val = doc.get("value")
                        if val is not None:
                            is_pct = c.endswith("Pct")
                            node_values[c] = f"{val:.2f}%" if is_pct else f"{val:,.2f}"
        except Exception as exc:
            logger.debug("formula-graph annotation failed: %s", exc)

    data = graph_to_json(DEFAULT_GRAPH, active_concepts)

    # Annotate nodes with real values where available
    for node in data["nodes"]:
        if node["id"] in node_values:
            node["value"] = node_values[node["id"]]

    return data


class AnalyzeWithTrendsRequest(BaseModel):
    report: str
    ticker: str


@app.post("/api/analyze-with-trends")
async def analyze_with_trends(req: AnalyzeWithTrendsRequest):
    """
    Run the full validation pipeline and augment the result with a TrendInsight.
    Saves a CredibilitySnapshot to MongoDB for future trend analysis.
    For educational purposes only — not investment advice.
    """
    if not req.report or not req.report.strip():
        raise HTTPException(status_code=400, detail="report field is required")
    if not req.ticker or not req.ticker.strip():
        raise HTTPException(status_code=400, detail="ticker field is required")

    _check_and_claim_session()
    try:
        result = await _run_pipeline(req.report.strip())
        claims_internal = result.pop("_claims_internal", [])
        timings = result.pop("_timings", {})
        vid = result.pop("_validation_id", None)
        validations_list = list(result["validations"].values())
        scoring = {k: v for k, v in result.items() if k not in ("claims", "validations")}

        from datetime import datetime as _dt
        from .audit import save_audit_log
        from .historical_claims import save_snapshot
        from .trend_engine import build_trend_insight

        await save_audit_log(
            validation_id=vid,
            input_text=req.report.strip(),
            input_type="text",
            ticker=req.ticker.strip(),
            claims=claims_internal,
            validations=validations_list,
            scoring=scoring,
            timings=timings,
        )
        await save_snapshot(
            ticker=req.ticker.strip(),
            score_result=scoring,
            claims=claims_internal,
            validations=validations_list,
            source_label=f"Analyzed {_dt.utcnow().strftime('%Y-%m-%d')}",
        )

        trend = await build_trend_insight(
            ticker=req.ticker.strip(),
            validations=validations_list,
            claims=claims_internal,
            scoring=scoring,
        )

        result["validation_id"] = vid
        result["trend_insight"] = trend.model_dump()
        return result
    finally:
        _release_session()


@app.get("/api/ticker/{ticker}/trend")
async def get_ticker_trend(ticker: str):
    """
    Return the latest cached TrendInsight for a ticker from MongoDB,
    or generate a new one from stored history if no cache entry exists.
    For educational purposes only — not investment advice.
    """
    if not ticker or not ticker.strip():
        raise HTTPException(status_code=400, detail="ticker is required")

    ticker_upper = ticker.strip().upper()

    try:
        from .historical_claims import get_ticker_history
        from .trend_engine import build_trend_insight

        snapshots = await get_ticker_history(ticker_upper, limit=10)
        if not snapshots:
            raise HTTPException(
                status_code=404,
                detail=f"No credibility history found for {ticker_upper}. "
                       "Run /api/analyze-with-trends first to build history.",
            )

        # Build a trend insight from history only (no current report)
        last = snapshots[0]
        dummy_scoring = {
            "overall_score": last.overall_score,
            "credibility_rating": last.credibility_rating,
            "analyst_bias": last.analyst_bias,
            "red_flags": [{"severity": "HIGH", "message": m} for m in last.high_severity_flags],
        }
        trend = await build_trend_insight(
            ticker=ticker_upper,
            validations=[],
            claims=[],
            scoring=dummy_scoring,
        )
        return trend.model_dump()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Trend endpoint error for %s: %s", ticker_upper, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/ticker/{ticker}/stock-score")
async def get_ticker_stock_score(ticker: str):
    """
    Compute the composite Stock Score for a ticker.
    F from EDGAR XBRL (growth, margins, ROE, debt),
    T from yfinance price history (trend + RSI),
    Q from most recent CredibilitySnapshot (lazy — runs /analyze first),
    M neutral 5.0 (sector data source pending).
    For educational purposes only — not investment advice.
    """
    if not ticker or not ticker.strip():
        raise HTTPException(status_code=400, detail="ticker is required")
    try:
        from .stock_scorer import compute_stock_score
        result = await compute_stock_score(ticker.strip().upper())
        return result.model_dump()
    except Exception as exc:
        logger.error("stock-score error for %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Layer 6 — Audit & Feedback endpoints
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    validation_id: str
    feedback_type: str      # evidence_relevance | claim_quality | contradiction_accuracy | overall
    feedback_value: str     # positive | negative | 1-5
    claim_id: str | None = None
    notes: str | None = None


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Submit analyst feedback on a validation result or a specific claim.
    Signals accumulate in feedback_signals for future retrieval retraining.
    """
    from .feedback import save_feedback
    try:
        feedback_id = await save_feedback(
            validation_id=req.validation_id,
            feedback_type=req.feedback_type,
            feedback_value=req.feedback_value,
            claim_id=req.claim_id,
            notes=req.notes,
        )
        return {"ok": True, "feedback_id": feedback_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/audit/{validation_id}")
async def get_audit(validation_id: str):
    """
    Retrieve the full immutable execution trace for a validation run,
    including every claim, validation result, timings, and model versions.
    """
    from .audit import get_audit_log
    record = await get_audit_log(validation_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"No audit record found for validation_id={validation_id}",
        )
    return record


@app.get("/api/audit")
async def list_audits(ticker: str = "", limit: int = 20):
    """
    List recent audit records (newest first), optionally filtered by ticker.
    Claims and validations are excluded from the list view — use
    GET /api/audit/{id} for the full record.
    """
    from .audit import list_audit_logs
    if limit > 100:
        limit = 100
    records = await list_audit_logs(
        ticker=ticker.strip().upper() if ticker.strip() else None,
        limit=limit,
    )
    return {"records": records, "count": len(records)}


@app.get("/api/feedback/stats")
async def feedback_stats(ticker: str = ""):
    """
    Aggregate feedback signal counts by type and value.
    Used by the governance layer to decide when to trigger a retraining run.
    """
    from .feedback import get_feedback_stats
    return await get_feedback_stats(ticker=ticker.strip().upper() if ticker.strip() else None)


@app.get("/api/feedback/{validation_id}")
async def get_validation_feedback(validation_id: str):
    """All feedback signals submitted for a specific validation run."""
    from .feedback import get_feedback_for_validation
    signals = await get_feedback_for_validation(validation_id)
    return {"validation_id": validation_id, "signals": signals, "count": len(signals)}


@app.get("/api/ticker/{ticker}/price-history")
async def get_price_history(ticker: str):
    """Returns 1-year daily price history (date, close, volume) for the Technical chart."""
    from .price_fetcher import fetch_price_context
    try:
        ctx = await fetch_price_context(ticker.strip().upper())
        return {
            "ticker": ticker.strip().upper(),
            "available": ctx["price_available"],
            "current_price": ctx.get("current_price"),
            "price_52w_high": ctx.get("price_52w_high"),
            "price_52w_low": ctx.get("price_52w_low"),
            "price_change_30d_pct": ctx.get("price_change_30d_pct"),
            "price_change_90d_pct": ctx.get("price_change_90d_pct"),
            "price_change_12m_pct": ctx.get("price_change_12m_pct"),
            "sma_50": ctx.get("sma_50"),
            "sma_200": ctx.get("sma_200"),
            "volume_3m_avg": ctx.get("volume_3m_avg"),
            "volume_current": ctx.get("volume_current"),
            "history": ctx.get("_price_history_vol", []),  # [(date, close, volume)]
        }
    except Exception as exc:
        logger.error("price-history error for %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/ticker/{ticker}/credibility-history")
async def get_credibility_history(ticker: str, limit: int = 24):
    """Returns historical credibility snapshots for the Composite trend chart."""
    if not is_connected():
        return {"ticker": ticker.strip().upper(), "available": False, "snapshots": []}
    try:
        from .historical_claims import get_ticker_history
        snaps = await get_ticker_history(ticker.strip().upper(), limit=min(limit, 50))
        return {
            "ticker": ticker.strip().upper(),
            "available": bool(snaps),
            "snapshots": [
                {"date": s.timestamp.isoformat(), "score": round(s.overall_score, 1), "rating": s.credibility_rating}
                for s in snaps
            ],
        }
    except Exception as exc:
        logger.error("credibility-history error for %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/ticker/{ticker}/recommendation")
async def get_ticker_recommendation(ticker: str):
    """
    Full dashboard payload for the Empirical Stock Score dashboard.
    All scores are on a 0-100 scale.
    Returns: composite/fundamental/technical/quality/macro gauges,
             tagged breakdown rows, short/medium/long-term recommendations,
             and price chart data.
    For educational purposes only — not investment advice.
    """
    if not ticker or not ticker.strip():
        raise HTTPException(status_code=400, detail="ticker is required")

    ticker_upper = ticker.strip().upper()
    try:
        from .stock_scorer import compute_stock_score, DISCLAIMER
        from .recommendation_engine import (
            compute_recommendations, overall_rating,
            tag_fundamental, tag_technical, tag_quality,
        )
        from .price_fetcher import fetch_price_context

        stock = await compute_stock_score(ticker_upper)
        sd = stock.model_dump()

        fund_bd  = sd.get("fundamental_breakdown") or {}
        tech_bd  = sd.get("technical_breakdown")   or {}
        qual_bd  = sd.get("qualitative_breakdown")  or {}

        # Scale 0-10 → 0-100
        def _s(v):
            return round(v * 10, 1) if v is not None else None

        composite_100   = _s(sd["composite_score"])
        fundamental_100 = _s(sd.get("fundamental_score"))
        technical_100   = _s(sd.get("technical_score"))
        quality_100     = _s(sd.get("qualitative_score"))
        macro_100       = _s(sd.get("macro_score"))

        # Build a flat dict matching recommendation_engine's expected shape
        _score_for_rec = {
            "composite": sd["composite_score"],
            "macro_score": sd.get("macro_score", 5.0),
            "fundamental": {
                "score":     sd.get("fundamental_score") or 5.0,
                "available": sd.get("fundamental_available", False),
                "breakdown": fund_bd,
            },
            "technical": {
                "score":     sd.get("technical_score") or 5.0,
                "available": sd.get("technical_available", False),
                "breakdown": tech_bd,
            },
            "quality": {
                "score":     sd.get("qualitative_score", 5.0),
                "breakdown": qual_bd,
            },
        }
        recommendations = compute_recommendations(_score_for_rec)

        # Price chart
        price_ctx = await fetch_price_context(ticker_upper)
        price_chart = {
            "available": price_ctx.get("price_available", False),
            "current_price": price_ctx.get("current_price"),
            "price_52w_high": price_ctx.get("price_52w_high"),
            "price_52w_low":  price_ctx.get("price_52w_low"),
            "sma_50":  price_ctx.get("sma_50"),
            "sma_200": price_ctx.get("sma_200"),
            "history": price_ctx.get("_price_history_vol", []),  # [(date, close, volume)]
        }

        return {
            "ticker": ticker_upper,
            "generated_at": sd["generated_at"],
            "overall_rating": overall_rating(composite_100 or 50),
            "disclaimer": DISCLAIMER,

            # Gauge scores (0-100)
            "scores": {
                "composite":   composite_100,
                "fundamental": fundamental_100,
                "technical":   technical_100,
                "quality":     quality_100,
                "macro":       macro_100,
            },
            "availability": {
                "composite":   True,
                "fundamental": sd.get("fundamental_available", False),
                "technical":   sd.get("technical_available", False),
                "quality":     True,   # always available — defaults to neutral 5.0 when no snapshot
                "macro":       True,   # always available — placeholder 5.0
            },

            # Tagged rows for breakdown tables
            "fundamental_rows": tag_fundamental(fund_bd),
            "technical_rows":   tag_technical(tech_bd, price_ctx),
            "quality_rows":     tag_quality(qual_bd),

            # ST/MT/LT recommendation cards
            "recommendations": recommendations,

            # Price chart
            "price_chart": price_chart,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("recommendation error for %s: %s", ticker_upper, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/demo")
async def serve_demo():
    """Serves the public one-liner claim validator demo page."""
    from pathlib import Path
    from fastapi.responses import FileResponse
    p = Path(__file__).parent.parent / "demo.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="demo.html not found")
    return FileResponse(p, media_type="text/html")


class OneLinerRequest(BaseModel):
    claim: str
    ticker: str | None = None


@app.post("/api/validate-one-liner")
async def validate_one_liner(req: OneLinerRequest):
    """
    Validate a single financial claim (up to 500 chars) through the full pipeline.
    Uses Groq when hosted (OLLAMA_BASE_URL not set), Ollama locally.
    """
    claim_text = req.claim.strip()
    if not claim_text:
        raise HTTPException(status_code=400, detail="claim is required")
    if len(claim_text) > 500:
        raise HTTPException(status_code=400, detail="claim must be 500 characters or fewer")

    try:
        from .claim_extractor import extract_claims
        from .validator import validate_claim

        # Prepend ticker hint so extraction can tag claims correctly
        text = claim_text
        if req.ticker:
            text = f"[{req.ticker.strip().upper()}] {claim_text}"

        claims = await extract_claims(text)
        if not claims:
            return {
                "input": claim_text,
                "ticker": req.ticker,
                "claims_extracted": 0,
                "results": [],
                "message": "No verifiable claim detected. Try including a specific number and company name.",
            }

        # Validate all extracted claims (usually 1–3 for a one-liner)
        import asyncio as _asyncio
        raw_results = await _asyncio.gather(*[validate_claim(c) for c in claims], return_exceptions=True)

        results = []
        for claim, res in zip(claims, raw_results):
            if isinstance(res, Exception):
                continue
            results.append({
                "claim_text":   claim.raw_text,
                "type":         claim.type.value,
                "metric":       claim.metric,
                "value":        claim.value,
                "period":       claim.period,
                "ticker":       claim.ticker,
                "checkable":    claim.checkable,
                "verdict":      res.status.value,
                "confidence":   round(res.confidence, 2),
                "actual_value": res.actual_value,
                "discrepancy":  res.discrepancy,
                "reasoning":    res.reasoning,
                "source":       res.filing_source or "web search",
                "edgar_url":    res.edgar_url,
                "citations":    res.citations[:3],
            })

        return {
            "input":             claim_text,
            "ticker":            req.ticker,
            "claims_extracted":  len(claims),
            "results":           results,
            "backend":           "ollama" if os.getenv("OLLAMA_BASE_URL") else "groq",
        }

    except Exception as exc:
        logger.error("validate-one-liner error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/dashboard")
async def serve_dashboard():
    """Serves the standalone stock analysis dashboard HTML."""
    from pathlib import Path
    from fastapi.responses import FileResponse
    p = Path(__file__).parent.parent / "dashboard.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found in project root")
    return FileResponse(p, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
