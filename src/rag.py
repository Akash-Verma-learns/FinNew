from __future__ import annotations

"""
RAG retrieval and verification.

Flow:
  embed(query) → Atlas Vector Search on text_chunks → top-K chunks → LLM verify

Atlas Vector Search index must be created in the Atlas UI before this works:
  Collection : text_chunks
  Index name : text_chunks_vector
  Field      : embedding
  Dimensions : 384
  Similarity : cosine
  Filter     : add "cik" as a filter field (pre-filter by company)
"""

import logging
from typing import Optional

from .embeddings import embed_one
from .groq_client import chat, parse_json
from .models import Citation, ValidationResult, ValidationStatus
from .xbrl_lookup import _normalize_ticker, _parse_value, _compare, _format_value

logger = logging.getLogger(__name__)

RAG_PROMPT = """You are a financial analyst verifying a specific claim against excerpts from an SEC 10-K filing.

Company: {company} ({ticker})
Claim: {raw_text}
Stated value: {value}
Period: {period}

Filing excerpts (from {company}'s actual 10-K):
{context}

Instructions:
- Only use data from the excerpts above. Do not use outside knowledge.
- If the excerpts are from a different company, return UNVERIFIABLE.
- If you find the exact figure, compare it to the stated value.
- Return JSON only:
  {{
    "status": "VERIFIED" | "PARTIALLY_VERIFIED" | "UNVERIFIABLE" | "CONTRADICTED",
    "confidence": 0.0-1.0,
    "actual_value": "<value found in filing or null>",
    "reasoning": "<one sentence>"
  }}"""


async def rag_verify(claim) -> Optional[ValidationResult]:
    """
    Retrieve relevant 10-K chunks via Atlas Vector Search and verify the claim.
    Returns None if DB is unavailable, index not set up, or no chunks found.
    """
    try:
        from .db import collections, is_connected
        from .xbrl_lookup import _is_non_us_ticker
        if not is_connected() or not claim.ticker:
            return None

        # RAG only works for companies whose 10-K was ingested via /api/ingest-text
        # Non-US exchange tickers (WSE:, LSE:, etc.) are never on SEC EDGAR, skip immediately
        if _is_non_us_ticker(claim.ticker):
            logger.info("  [rag] skipping — non-US ticker %r has no EDGAR 10-K chunks", claim.ticker)
            return None

        ticker = _normalize_ticker(claim.ticker)
        cik_doc = await collections.ticker_lookup.find_one({"_id": ticker})
        if not cik_doc:
            logger.info("  [rag] ticker %r not in ticker_lookup — run /api/ingest first", ticker)
            return None
        cik = cik_doc["cik"]

        # Check if text chunks exist for this company at all
        count = await collections.text_chunks.count_documents({"cik": cik}, limit=1)
        if count == 0:
            logger.info("  [rag] no text_chunks for CIK=%s — run /api/ingest-text %s first", cik, ticker)
            return None

        # Embed the query
        query = f"{claim.company or ''} {claim.metric or ''} {claim.value or ''} {claim.period or ''}".strip()
        query_vec = embed_one(query)

        # Atlas Vector Search — pre-filtered by company CIK
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "text_chunks_vector",
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": 80,
                    "limit": 5,
                    "filter": {"cik": cik},
                }
            },
            {
                "$project": {
                    "text": 1,
                    "section": 1,
                    "period_end": 1,
                    "edgar_url": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        chunks = []
        async for doc in collections.text_chunks.aggregate(pipeline):
            if doc.get("score", 0) >= 0.5:  # discard low-relevance chunks
                chunks.append(doc)

        if not chunks:
            logger.info("  [rag] no relevant chunks (score < 0.5) for claim=%s — skipping", claim.id)
            return None

        logger.info("  [rag] %d chunks retrieved | scores: %s | sections: %s | source: %s",
                    len(chunks),
                    ", ".join(f"{c.get('score', 0):.2f}" for c in chunks),
                    ", ".join(c.get("section", "?") for c in chunks),
                    chunks[0].get("edgar_url", "unknown"))

        context = "\n\n---\n\n".join(
            f"[Section: {c['section']}]\n{c['text']}" for c in chunks
        )

        prompt = RAG_PROMPT.format(
            company=claim.company or "unknown",
            ticker=claim.ticker or "unknown",
            raw_text=claim.raw_text,
            value=claim.value or "not stated",
            period=claim.period or "not stated",
            context=context,
        )

        raw = await chat(prompt)
        data = parse_json(raw)

        raw_status = data.get("status", "UNVERIFIABLE")
        try:
            status = ValidationStatus(raw_status)
        except ValueError:
            status = ValidationStatus.UNVERIFIABLE

        result = ValidationResult(claim_id=claim.id)
        result.status = status
        result.confidence = float(data.get("confidence", 0.5))
        result.actual_value = data.get("actual_value")
        result.reasoning = data.get("reasoning", "")
        result.evidence = context[:400]
        result.filing_source = f"10-K text chunks (RAG) — {chunks[0].get('section', 'unknown')} section"
        result.cik = cik
        result.edgar_url = chunks[0].get("edgar_url")
        unique_urls = list({c.get("edgar_url") for c in chunks if c.get("edgar_url")})
        result.citations = unique_urls
        result.structured_citations = [
            Citation(
                source="10K_TEXT",
                label=f"10-K {c.get('section', 'unknown')} (RAG)",
                url=c.get("edgar_url"),
                ticker=ticker,
                filing=f"10-K {c.get('period_end', '')}",
                section=c.get("section"),
                excerpt=c.get("text", "")[:200],
                period=c.get("period_end"),
            )
            for c in chunks
            if c.get("edgar_url") or c.get("section")
        ]

        logger.info("  [rag] LLM verdict: claim=%s status=%s confidence=%.2f",
                    claim.id, status.value, result.confidence)
        return result

    except Exception as exc:
        # Atlas Vector Search index not created yet → falls through gracefully
        logger.debug("RAG skipped (%s)", exc)
        return None
