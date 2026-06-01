from __future__ import annotations

"""
PageIndex integration for FinValidator.

Wraps VectifyAI/PageIndex to build hierarchical section trees from PDFs and
markdown documents. Used in two places:
  1. /api/analyze-pdf  — section-aware multi-pass claim extraction (no truncation)
  2. text_ingestion.py — section-aligned 10-K chunks for RAG (replaces 800-char flat cuts)

LLM backend: routes through LiteLLM → Groq using the existing GROQ_API_KEY.
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

WORKSPACE = Path(__file__).parent.parent / "pageindex_workspace"
_VENDOR = Path(__file__).parent.parent / "vendor" / "PageIndex"
_client = None


def _ensure_vendor_path() -> None:
    """Force the vendored PageIndex onto sys.path and evict any conflicting cached import."""
    import sys
    if not _VENDOR.exists():
        return
    vendor_str = str(_VENDOR)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
    # Evict any pip-installed version so re-import picks up the vendor copy
    stale = [k for k in sys.modules if k == "pageindex" or k.startswith("pageindex.")]
    for k in stale:
        del sys.modules[k]


def _get_client():
    global _client
    if _client is not None:
        return _client
    _ensure_vendor_path()
    try:
        from pageindex import PageIndexClient
    except ImportError:
        raise RuntimeError(
            "PageIndex not found — expected at vendor/PageIndex. "
            "Run: git clone https://github.com/VectifyAI/PageIndex.git vendor/PageIndex"
        )
    WORKSPACE.mkdir(exist_ok=True)
    ollama_base = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    if ollama_base:
        # Route PageIndex through local Ollama — no rate limits, no API key needed
        # LiteLLM reads OLLAMA_API_BASE to find the local server
        os.environ.setdefault("OLLAMA_API_BASE", ollama_base)
        pi_model = f"ollama/{ollama_model}"
        api_key = None
        logger.info("[pageindex] using Ollama backend — model=%s base=%s", ollama_model, ollama_base)
    else:
        # Fall back to Groq (llama-3.1-8b-instant has 30k TPM vs 12k for 70B)
        pi_model = f"groq/{os.getenv('PAGEINDEX_GROQ_MODEL', 'llama-3.1-8b-instant')}"
        api_key = os.getenv("PAGEINDEX_API_KEY") or os.getenv("OPENAI_API_KEY")
        logger.info("[pageindex] using Groq backend — model=%s", pi_model)

    try:
        _client = PageIndexClient(
            api_key=api_key,
            model=pi_model,
            retrieve_model=pi_model,
            workspace=str(WORKSPACE),
        )
        logger.info("[pageindex] client initialised — model=%s workspace=%s", pi_model, WORKSPACE)
    except TypeError:
        _client = PageIndexClient(api_key=api_key, workspace=str(WORKSPACE))
        logger.info("[pageindex] client initialised (no model override) — workspace=%s", WORKSPACE)
    return _client


def index_pdf(pdf_bytes: bytes, name: str = "document") -> tuple[str, list]:
    """
    Index a PDF from raw bytes. Returns (doc_id, tree_nodes).
    The tree reflects the document's natural section hierarchy.
    """
    client = _get_client()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix=f"fi_{name}_") as f:
        f.write(pdf_bytes)
        tmp_path = f.name
    try:
        logger.info("[pageindex] indexing PDF %s (%d bytes)", name, len(pdf_bytes))
        doc_id = client.index(tmp_path)
        raw = client.get_document_structure(doc_id)
        tree = _parse_structure(raw)
        logger.info("[pageindex] indexed %s → doc_id=%s  nodes=%d", name, doc_id, len(tree))
        return doc_id, tree
    finally:
        os.unlink(tmp_path)


def index_markdown(md_text: str, name: str = "document") -> tuple[str, list]:
    """
    Index markdown text (e.g. 10-K HTML converted to plain text).
    Returns (doc_id, tree_nodes).
    """
    client = _get_client()
    with tempfile.NamedTemporaryFile(
        suffix=".md", mode="w", encoding="utf-8", delete=False, prefix=f"fi_{name}_"
    ) as f:
        f.write(md_text)
        tmp_path = f.name
    try:
        logger.info("[pageindex] indexing markdown %s (%d chars)", name, len(md_text))
        doc_id = client.index(tmp_path)
        raw = client.get_document_structure(doc_id)
        tree = _parse_structure(raw)
        logger.info("[pageindex] indexed %s → doc_id=%s  nodes=%d", name, doc_id, len(tree))
        return doc_id, tree
    finally:
        os.unlink(tmp_path)


def get_section_texts(doc_id: str, tree: list) -> dict[str, str]:
    """
    Walk the PageIndex tree and fetch text for each section.
    Returns {section_title: text}.  Sections without page data are skipped.
    """
    client = _get_client()
    sections: dict[str, str] = {}

    def _walk(node: dict, depth: int = 0) -> None:
        title = node.get("title") or f"section_{depth}"
        start = node.get("start_index")
        end = node.get("end_index")

        if start is not None and end is not None:
            pages = f"{start}-{end}" if start != end else str(start)
            try:
                raw = client.get_page_content(doc_id, pages)
                pages_data = json.loads(raw) if isinstance(raw, str) else raw
                text = "\n".join(p.get("content", "") for p in (pages_data or []))
                if text.strip():
                    key = title[:80]  # truncate long headings
                    # merge duplicate section names
                    if key in sections:
                        sections[key] += "\n\n" + text
                    else:
                        sections[key] = text
                    logger.info("[pageindex] section=%r pages=%s chars=%d", key, pages, len(text))
            except Exception as exc:
                logger.debug("[pageindex] page fetch failed for %r pages=%s: %s", title, pages, exc)

        for child in node.get("nodes", []):
            _walk(child, depth + 1)

    for node in tree:
        _walk(node)

    return sections


def get_flat_chunks(doc_id: str, tree: list, max_chars: int = 2000) -> list[dict]:
    """
    Convert the PageIndex tree into a flat list of chunks, each bounded by a
    natural section boundary rather than an arbitrary character count.

    Each chunk: {"section": str, "text": str, "pages": str}
    """
    sections = get_section_texts(doc_id, tree)
    chunks: list[dict] = []

    for section, text in sections.items():
        # Split long sections at paragraph boundaries (keeps chunks coherent)
        if len(text) <= max_chars:
            chunks.append({"section": section, "text": text.strip(), "pages": ""})
        else:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            current: list[str] = []
            current_len = 0
            for para in paragraphs:
                if current_len + len(para) > max_chars and current:
                    chunks.append({
                        "section": section,
                        "text": "\n\n".join(current),
                        "pages": "",
                    })
                    current, current_len = [], 0
                current.append(para)
                current_len += len(para)
            if current:
                chunks.append({
                    "section": section,
                    "text": "\n\n".join(current),
                    "pages": "",
                })

    logger.info("[pageindex] %d section-aligned chunks from %d sections", len(chunks), len(sections))
    return chunks


def _parse_structure(raw) -> list:
    """Normalise whatever get_document_structure returns into a plain list of nodes."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return parsed.get("structure", [parsed])
        except json.JSONDecodeError:
            pass
    if isinstance(raw, dict):
        return raw.get("structure", [raw])
    return []
