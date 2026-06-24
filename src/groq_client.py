from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Backend selection — Ollama takes priority over Groq when configured
# ---------------------------------------------------------------------------
_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Local Ollama: no rate limit, no API key required
USE_OLLAMA = bool(_OLLAMA_BASE)

# Groq free-tier: 6000 TPM ≈ 4s gap at ~400 tokens/call
MIN_GAP_SECONDS = 4.0

# Ollama: GPU is single-threaded per model load — more than 4 concurrent
# requests just queue inside Ollama with no throughput gain and high memory cost.
_OLLAMA_CONCURRENCY = int(os.getenv("OLLAMA_CONCURRENCY", "4"))

# Shared state (Groq path only — Ollama is unlimited)
_groq_client = None
_ollama_client = None
_rate_lock: asyncio.Lock | None = None
_ollama_sem: asyncio.Semaphore | None = None
_last_call_time: float = 0.0

ACTIVE_MODEL = _OLLAMA_MODEL if USE_OLLAMA else _GROQ_MODEL
ACTIVE_BACKEND = "ollama" if USE_OLLAMA else "groq"


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import AsyncGroq
        _groq_client = AsyncGroq()
    return _groq_client


def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        from openai import AsyncOpenAI
        _ollama_client = AsyncOpenAI(
            base_url=f"{_OLLAMA_BASE}/v1",
            api_key="ollama",  # Ollama doesn't enforce API keys
        )
    return _ollama_client


def _get_lock() -> asyncio.Lock:
    global _rate_lock
    if _rate_lock is None:
        _rate_lock = asyncio.Lock()
    return _rate_lock


def _get_ollama_sem() -> asyncio.Semaphore:
    global _ollama_sem
    if _ollama_sem is None:
        _ollama_sem = asyncio.Semaphore(_OLLAMA_CONCURRENCY)
    return _ollama_sem


async def _call_ollama(content: str) -> str:
    """
    Call Ollama via its native /api/chat endpoint.

    The OpenAI-compatible wrapper doesn't support the top-level "think" field.
    Using the native API directly lets us pass think=false, which reliably
    suppresses qwen3's chain-of-thought mode (saves 30-60s per call).

    Gated by a semaphore (_OLLAMA_CONCURRENCY, default 4) so the GPU isn't
    flooded with queued requests — beyond ~4 concurrent calls the model just
    serialises internally with no throughput gain but high VRAM overhead.
    """
    import httpx
    url = f"{_OLLAMA_BASE}/api/chat"
    payload = {
        "model": _OLLAMA_MODEL,
        "messages": [{"role": "user", "content": content}],
        "think": False,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2000},
    }
    logger.info("  [ollama] POST %s/api/chat model=%s prompt=%d chars", _OLLAMA_BASE, _OLLAMA_MODEL, len(content))
    async with _get_ollama_sem():
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    reply = data.get("message", {}).get("content", "")
    logger.info("  [ollama] response=%d chars", len(reply))
    return reply


async def _call_groq(content: str) -> str:
    """Call Groq with rate-limit enforcement (free-tier: 6000 TPM → ~4s gap)."""
    global _last_call_time
    async with _get_lock():
        elapsed = time.monotonic() - _last_call_time
        wait = MIN_GAP_SECONDS - elapsed
        if wait > 0:
            logger.info("  [groq] rate-limit wait %.1fs", wait)
            await asyncio.sleep(wait)
        logger.info("  [groq] calling %s (prompt=%d chars)", _GROQ_MODEL, len(content))
        _last_call_time = time.monotonic()
        response = await _get_groq_client().chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=1500,
        )
    reply = response.choices[0].message.content or ""
    logger.info("  [groq] response=%d chars", len(reply))
    return reply


async def chat(content: str) -> str:
    """Send a message to the configured LLM backend (Ollama or Groq)."""
    if USE_OLLAMA:
        return await _call_ollama(content)
    return await _call_groq(content)


def _extract_json(text: str) -> str:
    """
    Extract the first complete JSON array or object from text.

    Uses bracket-walking instead of a greedy regex so trailing model commentary
    is ignored and the JSON stops exactly at the matching closing bracket.
    Also strips qwen3 <think>...</think> blocks which appear before the JSON
    and contain bracket characters that confuse naive extraction.
    """
    # Strip qwen3/deepseek thinking blocks
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    # Strip code fences
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
    return text


def parse_json(text: str) -> Any:
    return json.loads(_extract_json(text))


async def chat_json(prompt: str, schema: Type[T], max_retries: int = 3) -> T:
    """Call the LLM and validate the JSON response against a Pydantic schema."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = await chat(prompt)
            data = parse_json(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError, Exception) as exc:
            last_error = exc
            logger.warning("chat_json attempt %d/%d failed: %s", attempt, max_retries, exc)
    raise RuntimeError(f"chat_json failed after {max_retries} retries: {last_error}") from last_error
