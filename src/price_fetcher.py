"""
src/price_fetcher.py

Fetches historical share price data for a ticker using yfinance.
Returns structured price context for the trend engine and technical scorer.
All outputs are for educational/informational use only.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def fetch_price_context(ticker: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_sync, ticker)


def _ema(values: list[float], period: int) -> list[float]:
    """Exponential moving average."""
    if not values:
        return []
    k = 2.0 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _fetch_sync(ticker: str) -> dict:
    base: dict = {
        "price_available": False,
        "current_price": None,
        "price_52w_high": None,
        "price_52w_low": None,
        "price_change_30d_pct": None,
        "price_change_90d_pct": None,
        "price_change_12m_pct": None,
        "sma_50": None,
        "sma_200": None,
        "volume_3m_avg": None,
        "volume_current": None,
        "price_volatility_note": None,
        "credibility_price_pattern": None,
        "_price_history": [],        # [(date_str, close)] — internal, used for RSI
        "_price_history_vol": [],    # [(date_str, close, volume)] — public API
    }
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        info = t.info or {}

        current = info.get("currentPrice") or info.get("regularMarketPrice")
        high_52w = info.get("fiftyTwoWeekHigh")
        low_52w  = info.get("fiftyTwoWeekLow")

        if not current:
            logger.info("  [price] no currentPrice for %s — unavailable", ticker)
            return base

        hist = t.history(period="1y", interval="1d")
        if hist.empty:
            logger.info("  [price] empty history for %s — unavailable", ticker)
            return base

        closes  = hist["Close"].dropna()
        volumes = hist["Volume"].dropna() if "Volume" in hist.columns else None

        price_history     = [(str(d.date()), float(p)) for d, p in closes.items()]
        price_history_vol = [
            (str(d.date()), float(p), int(volumes.loc[d]) if volumes is not None and d in volumes.index else 0)
            for d, p in closes.items()
        ]

        # 30d / 90d price change
        change_30d: Optional[float] = None
        change_90d: Optional[float] = None
        if len(closes) >= 22:
            p_30 = float(closes.iloc[-22])
            change_30d = round((current - p_30) / p_30 * 100, 2)
        if len(closes) >= 63:
            p_90 = float(closes.iloc[-63])
            change_90d = round((current - p_90) / p_90 * 100, 2)

        # 12-month momentum (full period in history)
        change_12m: Optional[float] = None
        if len(closes) > 5:
            p_start = float(closes.iloc[0])
            if p_start:
                change_12m = round((current - p_start) / p_start * 100, 2)

        # 50 / 200 day SMA
        sma_50  = round(float(closes.iloc[-50:].mean()),  2) if len(closes) >= 50  else None
        sma_200 = round(float(closes.iloc[-200:].mean()), 2) if len(closes) >= 200 else None

        # Volume metrics
        volume_3m_avg: Optional[float] = None
        volume_current: Optional[float] = None
        if volumes is not None and len(volumes) > 0:
            volume_3m_avg  = round(float(volumes.iloc[-63:].mean()), 0) if len(volumes) >= 10 else None
            volume_current = round(float(volumes.iloc[-1]), 0)

        # Volatility note
        vol_note: Optional[str] = None
        if change_30d is not None:
            d30 = "up" if change_30d > 0 else "down"
            vol_note = f"Price has moved {d30} {abs(change_30d):.1f}% over the past 30 days"
            if change_90d is not None:
                d90 = "up" if change_90d > 0 else "down"
                vol_note += f" and {d90} {abs(change_90d):.1f}% over the past 90 days"

        logger.info(
            "  [price] %s: cur=%.2f 30d=%s 90d=%s 12m=%s sma50=%s sma200=%s history=%d pts",
            ticker, current, change_30d, change_90d, change_12m, sma_50, sma_200, len(price_history),
        )
        return {
            "price_available": True,
            "current_price": round(current, 4),
            "price_52w_high": round(high_52w, 4) if high_52w else None,
            "price_52w_low":  round(low_52w,  4) if low_52w  else None,
            "price_change_30d_pct": change_30d,
            "price_change_90d_pct": change_90d,
            "price_change_12m_pct": change_12m,
            "sma_50":  sma_50,
            "sma_200": sma_200,
            "volume_3m_avg":  volume_3m_avg,
            "volume_current": volume_current,
            "price_volatility_note":    vol_note,
            "credibility_price_pattern": None,
            "_price_history":     price_history,
            "_price_history_vol": price_history_vol,
        }

    except Exception as exc:
        logger.warning("Price fetch failed for %s: %s", ticker, exc)
        return base
