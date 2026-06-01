from __future__ import annotations

"""
Local embedding model — sentence-transformers all-MiniLM-L6-v2.

384-dimensional, CPU-only, ~88 MB download on first use.
Loaded once at server startup via `load_model()` so the first request
doesn't pay the ~2s cold-start cost.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384

_model = None


def load_model() -> None:
    global _model
    if _model is not None:
        return
    logger.info("Loading embedding model %s (first load ~2s, ~88 MB)...", EMBEDDING_MODEL)
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded — dims=%d", EMBEDDING_DIMS)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns list of 384-float vectors."""
    if _model is None:
        load_model()
    vecs = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
