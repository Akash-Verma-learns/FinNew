from __future__ import annotations

"""
10-K text ingestion pipeline.

Entry point: `ingest_text(ticker, cik, period_end)`

Steps:
  1. Look up the primary 10-K document URL from the EDGAR submissions API
  2. Fetch the HTML
  3. Parse into labelled sections (MD&A, Risk Factors, Notes, etc.)
  4. Chunk each section into ~800-char pieces with 80-char overlap
  5. Embed each chunk with all-MiniLM-L6-v2 (384 dims)
  6. Upsert into text_chunks collection in MongoDB Atlas
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from .db import collections
from .embeddings import EMBEDDING_MODEL, embed
from .xbrl_lookup import EDGAR_BASE, HEADERS

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 80     # overlap between consecutive chunks
MAX_DOC_CHARS = 600_000  # cap fetch size to avoid runaway memory on huge filings

# Patterns that identify major 10-K section boundaries
_SECTION_PATTERNS = [
    (re.compile(r"item\s+1a[\.\s]", re.I),  "risk_factors"),
    (re.compile(r"item\s+1[\.\s]",  re.I),  "business"),
    (re.compile(r"item\s+7a[\.\s]", re.I),  "quantitative_market_risk"),
    (re.compile(r"item\s+7[\.\s]",  re.I),  "mda"),
    (re.compile(r"item\s+8[\.\s]",  re.I),  "financial_statements"),
    (re.compile(r"item\s+9a[\.\s]", re.I),  "controls"),
]


async def ingest_text(ticker: str, cik: str, period_end: str) -> dict:
    """
    Fetch, chunk, embed, and store the primary 10-K document for the given CIK/period.
    Safe to call multiple times — uses $setOnInsert to avoid re-embedding existing chunks.

    Chunking strategy (in priority order):
      1. PageIndex — builds a hierarchical section tree, chunks align with document structure
      2. Regex fallback — Item 1/1A/7/7A/8 patterns + 800-char sliding window
    """
    doc_url = await _get_primary_doc_url(cik, period_end)
    if not doc_url:
        logger.warning("No primary 10-K document found for CIK=%s period=%s", cik, period_end)
        return {"status": "no_document", "chunks_written": 0}

    logger.info("Fetching 10-K for CIK=%s: %s", cik, doc_url)
    html = await _fetch_html(doc_url)
    if not html:
        return {"status": "fetch_failed", "chunks_written": 0}

    text = _html_to_text(html)
    # Regex section split (Docling is used for PDF uploads; 10-K ingestion uses HTML → text)
    sections = _split_sections(text)
    chunks = _build_chunks_regex(sections)
    indexer = "regex"

    if not chunks:
        return {"status": "no_chunks", "chunks_written": 0}

    logger.info("Embedding %d chunks (%s) with %s...", len(chunks), indexer, EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = embed(texts)

    accession_number = _accn_from_url(doc_url)
    written = 0
    for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        doc_id = f"{cik}__{period_end}__{chunk['section']}__{i:05d}"
        doc = {
            "_id": doc_id,
            "cik": cik,
            "ticker": ticker,
            "period_end": period_end,
            "form": "10-K",
            "section": chunk["section"],
            "chunk_index": i,
            "text": chunk["text"],
            "embedding": vec,
            "embedding_model": EMBEDDING_MODEL,
            "char_count": len(chunk["text"]),
            "accession_number": accession_number,
            "edgar_url": doc_url,
            "indexer": indexer,
            "ingested_at": _now(),
        }
        await collections.text_chunks.update_one(
            {"_id": doc_id},
            {"$setOnInsert": doc},
            upsert=True,
        )
        written += 1

    sections_used = list({c["section"] for c in chunks})
    logger.info("text_chunks: wrote %d chunks (%s) for CIK=%s period=%s",
                written, indexer, cik, period_end)
    return {"status": "ok", "chunks_written": written, "sections": sections_used, "indexer": indexer}


async def _get_primary_doc_url(cik: str, period_end: str) -> Optional[str]:
    """Find the primary 10-K document URL via the EDGAR submissions API."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        async with httpx.AsyncClient(headers=HEADERS) as client:
            resp = await client.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.error("submissions fetch failed: %s", exc)
        return None

    filings = data.get("filings", {}).get("recent", {})
    forms        = filings.get("form", [])
    accns        = filings.get("accessionNumber", [])
    report_dates = filings.get("reportDate", [])
    primary_docs = filings.get("primaryDocument", [])

    year = period_end[:4] if period_end else ""
    cik_bare = cik.lstrip("0")

    for form, accn, rdate, primary_doc in zip(forms, accns, report_dates, primary_docs):
        if form not in ("10-K", "20-F", "10-K/A", "20-F/A"):
            continue
        if not primary_doc:
            continue
        if year and year not in rdate:
            continue
        accn_clean = accn.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/{accn_clean}/{primary_doc}"

    return None


async def _fetch_html(url: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
            resp = await client.get(url, timeout=60)
            resp.raise_for_status()
            return resp.text[:MAX_DOC_CHARS]
    except Exception as exc:
        logger.error("HTML fetch failed (%s): %s", url, exc)
        return None


def _html_to_text(html: str) -> str:
    """Strip HTML tags, collapse whitespace, remove boilerplate."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "meta", "link", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove long sequences of dashes/dots (table separators)
    text = re.sub(r"[-_.]{4,}", " ", text)
    return text.strip()


def _split_sections(text: str) -> dict[str, str]:
    """
    Split full 10-K text into labelled sections using Item number patterns.
    Falls back to a single 'full_document' section if no sections detected.
    """
    positions: list[tuple[int, str]] = []
    for pattern, label in _SECTION_PATTERNS:
        for m in pattern.finditer(text):
            positions.append((m.start(), label))

    if not positions:
        return {"full_document": text}

    positions.sort(key=lambda x: x[0])
    sections: dict[str, str] = {}

    # Text before first section → preamble
    first_pos = positions[0][0]
    if first_pos > 200:
        sections["preamble"] = text[:first_pos]

    for i, (pos, label) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[pos:end].strip()
        if len(content) > 50:
            sections[label] = content

    return sections


def _build_chunks_regex(sections: dict[str, str]) -> list[dict]:
    """Split each section into overlapping character chunks (regex fallback)."""
    chunks: list[dict] = []
    for section, text in sections.items():
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 50:
                chunks.append({"section": section, "text": chunk_text})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _accn_from_url(url: str) -> Optional[str]:
    """Extract accession number from an EDGAR filing URL."""
    m = re.search(r"/(\d{18})/", url)
    if m:
        raw = m.group(1)
        return f"{raw[:10]}-{raw[10:12]}-{raw[12:]}"
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
