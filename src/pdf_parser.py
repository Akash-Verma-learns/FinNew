from __future__ import annotations

import io
import logging
import re

logger = logging.getLogger(__name__)

MAX_CHARS = 20_000
HEAD_CHARS = 15_000
TAIL_CHARS = 5_000


def extract_text_from_pdf(data: bytes) -> tuple[str, int]:
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n".join(pages), len(pages)


def prepare_text(full_text: str, max_chars: int = MAX_CHARS) -> tuple[str, bool, str | None]:
    """Return (prepared_text, was_truncated, warning_message).

    Preserves the executive summary (first 15k chars) and appends financially
    significant sentences from the remainder (numbers, %, $) up to 5k more chars.
    """
    if len(full_text) <= max_chars:
        return full_text, False, None

    head = full_text[:HEAD_CHARS]
    remainder = full_text[HEAD_CHARS:]

    financial_sentences: list[str] = []
    current_len = 0
    for sentence in re.split(r"(?<=[.!?])\s+", remainder):
        if re.search(r"[\d$%]", sentence) and current_len + len(sentence) <= TAIL_CHARS:
            financial_sentences.append(sentence)
            current_len += len(sentence)

    text = head + "\n\n[...document truncated...]\n\n" + " ".join(financial_sentences)
    warning = (
        f"Document truncated from {len(full_text):,} to ~{len(text):,} characters. "
        "Segment breakdowns, risk factor details, and deeper MD&A discussion may have been omitted. "
        "For complete analysis, consider splitting the document into sections."
    )
    return text, True, warning
