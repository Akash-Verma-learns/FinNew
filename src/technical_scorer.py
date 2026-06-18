"""
src/technical_scorer.py

Technical Score (T) from yfinance price history.
Formula: T = 0.40(Trend) + 0.25(RSI) + 0.20(12M Momentum) + 0.15(MA Cross)
For educational purposes only — historical price patterns, not predictions.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _norm(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 5.0
    return max(0.0, min(10.0, (value - lo) / (hi - lo) * 10))


def _wilder_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d,  0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period])  / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i])  / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def _ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2.0 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


async def compute_technical_score(ticker: str, price_ctx: Optional[dict] = None) -> dict:
    """
    Returns {"score": float, "available": bool, "breakdown": dict}.
    Pass an already-fetched price_ctx to avoid a duplicate yfinance call.
    """
    if price_ctx is None:
        from .price_fetcher import fetch_price_context
        price_ctx = await fetch_price_context(ticker)

    if not price_ctx.get("price_available"):
        return {"score": 5.0, "available": False, "breakdown": {}, "reason": "price_unavailable"}

    breakdown: dict = {}
    scores: dict[str, float] = {}

    # ── 30d / 90d trend strength ───────────────────────────────────────────────
    chg30 = price_ctx.get("price_change_30d_pct")
    chg90 = price_ctx.get("price_change_90d_pct")

    if chg30 is not None and chg90 is not None:
        s30 = _norm(chg30, -20, 40)
        s90 = _norm(chg90, -20, 40)
        trend = round(0.6 * s30 + 0.4 * s90, 2)
        scores["trend"] = trend
        breakdown["price_change_30d_pct"] = chg30
        breakdown["price_change_90d_pct"] = chg90
        breakdown["trend_score"] = trend
    elif chg30 is not None:
        s = round(_norm(chg30, -20, 40), 2)
        scores["trend"] = s
        breakdown["price_change_30d_pct"] = chg30
        breakdown["trend_score"] = s

    # ── RSI-14 ─────────────────────────────────────────────────────────────────
    history = price_ctx.get("_price_history", [])
    closes = [row[1] for row in history]   # handles both 2-tuple and 3-tuple rows

    if len(closes) >= 15:
        rsi = _wilder_rsi(closes)
        if rsi is not None:
            rsi_score = round(_norm(100 - rsi, 30, 70), 2)
            scores["rsi"] = rsi_score
            breakdown["rsi_14"] = round(rsi, 1)
            breakdown["rsi_score"] = rsi_score

    # ── 12-month momentum ──────────────────────────────────────────────────────
    chg12 = price_ctx.get("price_change_12m_pct")
    if chg12 is not None:
        s12 = round(_norm(chg12, -30, 80), 2)
        scores["momentum_12m"] = s12
        breakdown["price_change_12m_pct"] = chg12
        breakdown["momentum_12m_score"] = s12

    # ── 50 / 200 SMA cross ────────────────────────────────────────────────────
    sma50  = price_ctx.get("sma_50")
    sma200 = price_ctx.get("sma_200")
    if sma50 is not None and sma200 is not None:
        ma_trend = "Bullish" if sma50 > sma200 else "Bearish"
        ma_score = round(_norm(sma50 - sma200, -30, 30), 2)
        scores["ma_cross"] = ma_score
        breakdown["ma_trend"]  = ma_trend
        breakdown["sma_50"]    = sma50
        breakdown["sma_200"]   = sma200
        breakdown["ma_score"]  = ma_score

    # ── MACD (EMA12 − EMA26, signal EMA9) ─────────────────────────────────────
    if len(closes) >= 35:
        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        macd_line = [e12 - e26 for e12, e26 in zip(ema12, ema26)]
        if len(macd_line) >= 9:
            signal_line = _ema(macd_line, 9)
            macd_val    = macd_line[-1]
            signal_val  = signal_line[-1]
            breakdown["macd_signal"] = "Positive" if macd_val > signal_val else "Negative"
            breakdown["macd_value"]  = round(macd_val, 4)
            breakdown["macd_signal_value"] = round(signal_val, 4)

    # ── Volume signal ──────────────────────────────────────────────────────────
    vol3m = price_ctx.get("volume_3m_avg")
    volcur = price_ctx.get("volume_current")
    if vol3m and volcur and vol3m > 0:
        vol_ratio = volcur / vol3m
        breakdown["volume_ratio"] = round(vol_ratio, 2)
        if   vol_ratio > 1.30: breakdown["volume_signal"] = "Above Avg"
        elif vol_ratio > 0.70: breakdown["volume_signal"] = "Average"
        else:                  breakdown["volume_signal"] = "Below Avg"

    if not scores:
        return {"score": 5.0, "available": False, "breakdown": breakdown, "reason": "insufficient_data"}

    # ── Composite technical score ──────────────────────────────────────────────
    w = {
        "trend":       0.35,
        "rsi":         0.25,
        "momentum_12m": 0.25,
        "ma_cross":    0.15,
    }
    total_w = sum(w[k] for k in scores if k in w)
    if total_w > 0:
        t_score = round(sum(scores[k] * (w[k] / total_w) for k in scores if k in w), 2)
    else:
        t_score = round(sum(scores.values()) / len(scores), 2)

    return {"score": t_score, "available": True, "breakdown": breakdown}
