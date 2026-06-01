from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FinValidator v4.0 starting on port %s", os.getenv("PORT", "8000"))
    try:
        await init_db()
    except Exception as exc:
        logger.warning("MongoDB unavailable — running without DB cache: %s", exc)
    try:
        await asyncio.get_event_loop().run_in_executor(None, load_embedding_model)
    except Exception as exc:
        logger.warning("Embedding model failed to load: %s", exc)
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
    """Run validation + scoring for pre-extracted claims (PageIndex path)."""
    from .models import ValidationResult, ValidationStatus
    clear_validation_cache()
    logger.info("Validating %d pre-extracted claims...", len(claims))
    raw_results = await asyncio.gather(*[validate_claim(c) for c in claims], return_exceptions=True)
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
    scoring = score_report(claims, validations)
    return {
        **scoring,
        "claims": [c.model_dump() for c in claims],
        "validations": {k: v.model_dump() for k, v in validations.items()},
    }


async def _run_pipeline(text: str) -> dict:
    clear_validation_cache()
    logger.info("Extracting claims from %d chars", len(text))
    claims = await extract_claims(text)
    logger.info("Extracted %d claims — validating...", len(claims))

    raw_results = await asyncio.gather(*[validate_claim(c) for c in claims], return_exceptions=True)

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

    scoring = score_report(claims, validations)
    return {
        **scoring,
        "claims": [c.model_dump() for c in claims],
        "validations": {k: v.model_dump() for k, v in validations.items()},
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "4.0.0"}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.report or not req.report.strip():
        raise HTTPException(status_code=400, detail="report field is required")
    return await _run_pipeline(req.report.strip())


@app.post("/api/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="A .pdf file is required")
    data = await file.read()
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 30 MB)")

    _, page_count = extract_text_from_pdf(data)
    logger.info("PDF: %d pages — converting with Docling", page_count)

    # Primary: Docling → structured markdown → section-aware claim extraction
    try:
        from .docling_parser import process_pdf_bytes, split_sections
        markdown = await asyncio.get_event_loop().run_in_executor(
            None, process_pdf_bytes, data, file.filename
        )
        sections = split_sections(markdown)
        logger.info("[docling] %d sections, %d chars from %s", len(sections), len(markdown), file.filename)
        claims = await extract_claims_from_sections(sections)
        result = await _run_pipeline_with_claims(claims)
        result["pdf_truncated"] = False
        result["pdf_truncation_warning"] = None
        result["indexer"] = "docling"
        result["sections_found"] = list(sections.keys())
        return result
    except Exception as exc:
        logger.warning("Docling unavailable (%s) — falling back to flat text", exc)

    # Fallback: flat pdfplumber text with truncation
    full_text, _ = extract_text_from_pdf(data)
    text, truncated, warning = prepare_text(full_text)
    logger.info("Flat fallback: %d chars (truncated=%s)", len(text), truncated)
    result = await _run_pipeline(text)
    result["pdf_truncated"] = truncated
    result["pdf_truncation_warning"] = warning
    result["indexer"] = "flat"
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
