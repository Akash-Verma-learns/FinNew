"""
src/stock_scorer.py

Composite Stock Score: Score = 0.4F + 0.25T + 0.2Q + 0.15M
  F — Fundamental (EDGAR XBRL: growth, margins, ROE, debt)
  T — Technical   (yfinance: trend strength + RSI-14)
  Q — Qualitative (lazy: latest CredibilitySnapshot from MongoDB)
  M — Macro       (placeholder 5.0 — sector data source pending)

For educational purposes only. Not investment advice.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from .models import CredibilitySnapshot, StockScore

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "For educational purposes only. This is not investment advice. "
    "Past patterns do not guarantee future results. "
    "Brokerage services provided by MARV Capital Inc."
)

WEIGHTS: dict = {"F": 0.40, "T": 0.25, "Q": 0.20, "M": 0.15}


def _signal_label(score: float) -> str:
    if score >= 7.5:
        return "HIGH historical alignment"
    if score >= 6.0:
        return "MODERATE historical alignment"
    if score >= 5.0:
        return "MIXED signals"
    return "LOW historical alignment"


def _q_from_snapshot(snap: CredibilitySnapshot) -> dict:
    """
    Derive Q sub-score (0–10) from a CredibilitySnapshot.
    Q blends raw credibility, internal consistency, and analyst bias balance.
    """
    total = snap.total_claims or 1
    verified_rate      = snap.verified_count / total
    contradiction_rate = snap.contradicted_count / total

    # Balanced reports carry higher epistemic quality
    bias_factor = 1.0 if snap.analyst_bias == "BALANCED" else 0.7

    base        = (snap.overall_score / 100) * 10
    consistency = ((verified_rate * 0.5) + ((1 - contradiction_rate) * 0.5)) * 10
    q           = base * 0.6 + consistency * 0.3 + bias_factor * 10 * 0.1

    return {
        "score":              round(min(10.0, max(0.0, q)), 2),
        "source":             "credibility_snapshot",
        "snapshot_date":      snap.timestamp.isoformat(),
        "credibility_rating": snap.credibility_rating,
        "verified_rate":      round(verified_rate, 3),
        "contradiction_rate": round(contradiction_rate, 3),
        "analyst_bias":       snap.analyst_bias,
    }


async def compute_stock_score(ticker: str) -> StockScore:
    """
    Compute the composite StockScore for a ticker.
    F, price context, and latest snapshot are fetched concurrently;
    T is computed after price context resolves.
    """
    from .fundamental_scorer import compute_fundamental_score
    from .historical_claims import get_ticker_history
    from .price_fetcher import fetch_price_context
    from .technical_scorer import compute_technical_score

    ticker = ticker.upper()
    logger.info("[stock_score] computing for %s", ticker)

    f_result, price_ctx, history = await asyncio.gather(
        compute_fundamental_score(ticker),
        fetch_price_context(ticker),
        get_ticker_history(ticker, limit=1),
    )

    t_result = await compute_technical_score(ticker, price_ctx=price_ctx)

    # Q: lazy — latest credibility snapshot, else neutral 5.0
    if history:
        q_data         = _q_from_snapshot(history[0])
        q_score        = q_data["score"]
        q_from_snapshot = True
    else:
        q_data = {
            "score":  5.0,
            "source": "default_neutral",
            "note":   "No prior validation runs recorded — analyze a report first to improve Q accuracy",
        }
        q_score        = 5.0
        q_from_snapshot = False

    # M: sector/macro placeholder
    m_score = 5.0

    f_score = f_result["score"]
    t_score = t_result["score"]

    # Re-normalize over available components so a missing F or T doesn't silently
    # drag the composite toward 5.0 (the neutral placeholder).
    _component_scores = {"F": f_score, "T": t_score, "Q": q_score, "M": m_score}
    _active_weights = {
        k: WEIGHTS[k]
        for k, available in [
            ("F", f_result["available"]),
            ("T", t_result["available"]),
            ("Q", True),
            ("M", True),
        ]
        if available
    }
    _total_w = sum(_active_weights.values())
    composite = round(
        sum(_component_scores[k] * (_active_weights[k] / _total_w) for k in _active_weights),
        2,
    )

    logger.info(
        "[stock_score] %s → F=%.1f T=%.1f Q=%.1f M=%.1f composite=%.2f [%s]",
        ticker, f_score, t_score, q_score, m_score, composite, _signal_label(composite),
    )

    return StockScore(
        ticker=ticker,
        generated_at=datetime.utcnow(),
        fundamental_score=f_score if f_result["available"] else None,
        technical_score=t_score if t_result["available"] else None,
        qualitative_score=q_score,
        macro_score=m_score,
        composite_score=composite,
        signal_label=_signal_label(composite),
        fundamental_available=f_result["available"],
        technical_available=t_result["available"],
        qualitative_from_snapshot=q_from_snapshot,
        weights=WEIGHTS,
        fundamental_breakdown=f_result.get("breakdown"),
        technical_breakdown=t_result.get("breakdown"),
        qualitative_breakdown=q_data,
        disclaimer=DISCLAIMER,
    )
