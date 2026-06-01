from __future__ import annotations

"""
Docling-based PDF ingestion pipeline.

Replaces PageIndex. Uses IBM Docling for:
  - Layout-aware PDF → Markdown conversion (no LLM needed for structure detection)
  - TableFormer (ACCURATE mode) for financial table extraction
  - VLM (qwen3-vl:2b via Ollama) for chart/image descriptions

Architecture matches the reference implementation exactly.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VLM_MODEL = os.getenv("DOCLING_VLM_MODEL", "qwen3-vl:2b")
VLM_PROMPT = (
    "Act as a senior financial analyst. Explain what you see in the image in 1 sentence. "
    "Focus on trends, tickers, and key financial metrics visible in charts or tables."
)

PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"
IMAGE_DESCRIPTION_START = "<image_description>"
IMAGE_DESCRIPTION_END = "</image_description>"


def _create_picture_description_options():
    from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
    return PictureDescriptionApiOptions(
        url=f"{OLLAMA_URL}/v1/chat/completions",
        params=dict(
            model=VLM_MODEL,
            think=False,
            seed=42,
            max_completion_tokens=256,
        ),
        prompt=VLM_PROMPT,
        timeout=90,
    )


def _create_pdf_pipeline_options(describe_images: bool = True):
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableStructureOptions,
        AcceleratorOptions,
        AcceleratorDevice,
    )
    try:
        from docling.models.table_structure_model import TableFormerMode
        table_mode = TableFormerMode.ACCURATE
    except ImportError:
        table_mode = None

    table_opts = TableStructureOptions(mode=table_mode) if table_mode else TableStructureOptions()

    # Use GPU if available, fall back to CPU automatically
    accelerator = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.AUTO,
    )

    kwargs: dict[str, Any] = dict(
        enable_remote_services=True,
        do_ocr=False,
        do_table_structure=True,
        generate_picture_images=describe_images,
        do_picture_description=describe_images,
        table_structure_options=table_opts,
        accelerator_options=accelerator,
    )
    if describe_images:
        kwargs["picture_description_options"] = _create_picture_description_options()

    return PdfPipelineOptions(**kwargs)


def process_pdf_to_markdown(pdf_path: str, describe_images: bool = True) -> str:
    """
    Convert a PDF file to structured Markdown using Docling.

    Returns the full markdown string with:
    - Page-break placeholders (<!-- page_break -->)
    - Image descriptions wrapped in <image_description>...</image_description>
    - Tables preserved as markdown tables
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import ImageRefMode
    from docling.datamodel.base_models import InputFormat
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

    logger.info("[docling] converting %s (describe_images=%s, vlm=%s)", pdf_path, describe_images, VLM_MODEL)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=_create_pdf_pipeline_options(describe_images),
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

    result = converter.convert(pdf_path)
    doc = result.document

    content = doc.export_to_markdown(
        image_mode=ImageRefMode.PLACEHOLDER,
        image_placeholder="",
        page_break_placeholder=PAGE_BREAK_PLACEHOLDER,
        include_annotations=True,
        mark_annotations=True,
    )

    # Normalise docling annotation markers
    content = content.replace('<!--<annotation kind="description">-->', IMAGE_DESCRIPTION_START)
    content = content.replace("<!--<annotation/>-->", IMAGE_DESCRIPTION_END)

    logger.info("[docling] conversion complete — %d chars", len(content))

    # Release PyTorch CUDA allocator cache so Ollama can reclaim VRAM for inference.
    # Without this, PyTorch holds the memory even though all tensors are done,
    # which crashes Ollama when it tries to reload qwen3:8b after Docling finishes.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("[docling] CUDA cache cleared — VRAM released for Ollama")
    except Exception:
        pass

    return content


def process_pdf_bytes(pdf_bytes: bytes, name: str = "document", describe_images: bool | None = None) -> str:
    """Convert raw PDF bytes to markdown. Handles temp-file lifecycle."""
    if describe_images is None:
        # Env override: DOCLING_DESCRIBE_IMAGES=false skips VLM calls (~200s saved on table-heavy reports)
        describe_images = os.getenv("DOCLING_DESCRIBE_IMAGES", "true").lower() != "false"
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix=f"dl_{name}_") as f:
        f.write(pdf_bytes)
        tmp_path = f.name
    try:
        return process_pdf_to_markdown(tmp_path, describe_images=describe_images)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def split_sections(markdown: str) -> dict[str, str]:
    """
    Split Docling markdown into sections by heading (# / ## / ###).
    Returns {heading_title: section_text}.
    Falls back to {'full_document': markdown} if no headings found.
    """
    import re
    lines = markdown.splitlines()
    sections: dict[str, str] = {}
    current_title = "preamble"
    current_lines: list[str] = []

    for line in lines:
        m = re.match(r"^(#{1,3})\s+(.+)", line)
        if m:
            text = "\n".join(current_lines).strip()
            if text:
                sections[current_title] = text
            current_title = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    text = "\n".join(current_lines).strip()
    if text:
        sections[current_title] = text

    if not sections:
        return {"full_document": markdown}

    return sections
