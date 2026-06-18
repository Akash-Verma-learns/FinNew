"""
src/recommendation_engine.py

Derives Short / Medium / Long-term recommendations from scorer outputs.
All scores are on a 0-10 scale internally; the API endpoint multiplies ×10.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Tagging helpers — return [{name, value, tag, color}]
# tag: "positive" | "neutral" | "negative"
# ──────────────────────────────────────────────────────────────────────────────

_TAG_COLOR = {"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"}


def _tag(name: str, value, tag: str) -> dict:
    return {"name": name, "value": value, "tag": tag, "color": _TAG_COLOR[tag]}


def _score_tag(score: float) -> str:
    if score >= 7: return "positive"
    if score >= 4: return "neutral"
    return "negative"


def tag_fundamental(bd: dict) -> list[dict]:
    rows = []
    if "revenue_growth_pct" in bd:
        v = bd["revenue_growth_pct"]
        rows.append(_tag("Revenue Growth", f"{v:+.1f}%", "positive" if v > 5 else "neutral" if v > 0 else "negative"))
    if "eps_growth_pct" in bd:
        v = bd["eps_growth_pct"]
        rows.append(_tag("EPS Growth", f"{v:+.1f}%", "positive" if v > 5 else "neutral" if v > -5 else "negative"))
    if "roe_pct" in bd:
        v = bd["roe_pct"]
        rows.append(_tag("ROE", f"{v:.1f}%", "positive" if v > 15 else "neutral" if v > 5 else "negative"))
    if "net_margin_pct" in bd:
        v = bd["net_margin_pct"]
        rows.append(_tag("Net Margin", f"{v:.1f}%", "positive" if v > 10 else "neutral" if v > 0 else "negative"))
    if "debt_to_equity" in bd:
        v = bd["debt_to_equity"]
        rows.append(_tag("Debt/Equity", f"{v:.2f}x", "positive" if v < 0.5 else "neutral" if v < 1.5 else "negative"))
    if "fcf_margin_pct" in bd:
        v = bd["fcf_margin_pct"]
        rows.append(_tag("FCF Margin", f"{v:.1f}%", "positive" if v > 10 else "neutral" if v > 0 else "negative"))
    if "roic_pct" in bd:
        v = bd["roic_pct"]
        rows.append(_tag("ROIC", f"{v:.1f}%", "positive" if v > 12 else "neutral" if v > 5 else "negative"))
    return rows


def tag_technical(bd: dict, price_ctx: Optional[dict] = None) -> list[dict]:
    rows = []
    if "rsi_14" in bd:
        v = bd["rsi_14"]
        tag = "positive" if 40 <= v <= 60 else "neutral" if 30 <= v <= 70 else "negative"
        rows.append(_tag("RSI-14", f"{v:.0f}", tag))
    if "ma_trend" in bd:
        t = bd["ma_trend"]
        rows.append(_tag("MA Trend", t, "positive" if t == "Bullish" else "negative"))
    if "macd_signal" in bd:
        t = bd["macd_signal"]
        rows.append(_tag("MACD", t, "positive" if t == "Positive" else "negative"))
    if "price_change_30d_pct" in bd:
        v = bd["price_change_30d_pct"]
        rows.append(_tag("30D Return", f"{v:+.1f}%", "positive" if v > 3 else "neutral" if v > -3 else "negative"))
    if "price_change_12m_pct" in bd:
        v = bd["price_change_12m_pct"]
        rows.append(_tag("12M Return", f"{v:+.1f}%", "positive" if v > 10 else "neutral" if v > -10 else "negative"))
    if "volume_signal" in bd:
        t = bd["volume_signal"]
        rows.append(_tag("Volume", t, "positive" if t == "Above Avg" else "neutral" if t == "Average" else "negative"))
    return rows


def tag_quality(qbd: dict) -> list[dict]:
    rows = []
    for key in ("factual_accuracy", "coherence", "disclosure_completeness"):
        if key in qbd:
            v = qbd[key]
            label = key.replace("_", " ").title()
            try:
                score = float(v)
                rows.append(_tag(label, f"{score:.1f}/10", _score_tag(score)))
            except Exception:
                rows.append(_tag(label, str(v), "neutral"))
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Recommendation labels (0-100 scale)
# ──────────────────────────────────────────────────────────────────────────────

def _label_st(s: float) -> str:
    if s >= 75: return "STRONG BUY"
    if s >= 60: return "BUY"
    if s >= 45: return "HOLD"
    if s >= 30: return "REDUCE"
    return "SELL"


def _label_mt(s: float) -> str:
    if s >= 75: return "STRONG BUY"
    if s >= 60: return "BUY"
    if s >= 45: return "HOLD"
    if s >= 30: return "REDUCE"
    return "SELL"


def _label_lt(s: float) -> str:
    if s >= 82: return "STRONG BUY"
    if s >= 70: return "BUY & HOLD"
    if s >= 55: return "BUY"
    if s >= 40: return "HOLD"
    if s >= 25: return "REDUCE"
    return "SELL"


def overall_rating(composite_100: float) -> str:
    if composite_100 >= 82: return "STRONG BUY"
    if composite_100 >= 70: return "BUY"
    if composite_100 >= 55: return "HOLD"
    if composite_100 >= 40: return "REDUCE"
    return "SELL"


def _confidence(score_100: float) -> int:
    if score_100 >= 80: return 5
    if score_100 >= 70: return 4
    if score_100 >= 55: return 3
    if score_100 >= 40: return 2
    return 1


# ──────────────────────────────────────────────────────────────────────────────
# Bullet builders
# ──────────────────────────────────────────────────────────────────────────────

def _st_bullets(tech_bd: dict, composite_100: float) -> list[str]:
    bullets = []
    if tech_bd.get("macd_signal") == "Positive":
        bullets.append("MACD above signal line — bullish momentum")
    elif tech_bd.get("macd_signal") == "Negative":
        bullets.append("MACD below signal line — bearish momentum")
    if "rsi_14" in tech_bd:
        r = tech_bd["rsi_14"]
        if r > 70: bullets.append(f"RSI {r:.0f} — overbought territory")
        elif r < 30: bullets.append(f"RSI {r:.0f} — oversold, potential reversal")
        else: bullets.append(f"RSI {r:.0f} — neutral zone")
    if tech_bd.get("ma_trend") == "Bullish":
        bullets.append("50-day MA above 200-day MA (golden cross)")
    elif tech_bd.get("ma_trend") == "Bearish":
        bullets.append("50-day MA below 200-day MA (death cross)")
    if "price_change_30d_pct" in tech_bd:
        v = tech_bd["price_change_30d_pct"]
        bullets.append(f"30-day price change: {v:+.1f}%")
    return bullets[:4]


def _mt_bullets(fund_bd: dict, tech_bd: dict) -> list[str]:
    bullets = []
    if "revenue_growth_pct" in fund_bd:
        v = fund_bd["revenue_growth_pct"]
        bullets.append(f"Revenue growth: {v:+.1f}% YoY")
    if "eps_growth_pct" in fund_bd:
        v = fund_bd["eps_growth_pct"]
        bullets.append(f"EPS growth: {v:+.1f}% YoY")
    if "net_margin_pct" in fund_bd:
        v = fund_bd["net_margin_pct"]
        bullets.append(f"Net margin: {v:.1f}%")
    if tech_bd.get("price_change_12m_pct") is not None:
        v = tech_bd["price_change_12m_pct"]
        bullets.append(f"12-month price momentum: {v:+.1f}%")
    return bullets[:4]


def _lt_bullets(fund_bd: dict, quality_bd: dict) -> list[str]:
    bullets = []
    if "roe_pct" in fund_bd:
        v = fund_bd["roe_pct"]
        bullets.append(f"Return on equity: {v:.1f}%")
    if "roic_pct" in fund_bd:
        v = fund_bd["roic_pct"]
        bullets.append(f"ROIC: {v:.1f}%")
    if "fcf_margin_pct" in fund_bd:
        v = fund_bd["fcf_margin_pct"]
        bullets.append(f"Free cash flow margin: {v:.1f}%")
    if "debt_to_equity" in fund_bd:
        v = fund_bd["debt_to_equity"]
        bullets.append(f"Debt/equity ratio: {v:.2f}x")
    return bullets[:4]


# ──────────────────────────────────────────────────────────────────────────────
# Main computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_recommendations(stock_score: dict) -> dict:
    """
    stock_score: the full dict returned by scorer.compute_stock_score().
    Returns {short_term, medium_term, long_term} each as:
      {score: 0-100, label, confidence: 1-5, bullets: [str]}
    """
    f_res = stock_score.get("fundamental", {})
    t_res = stock_score.get("technical", {})
    q_res = stock_score.get("quality", {})

    F = f_res.get("score", 5.0) if f_res.get("available") else None
    T = t_res.get("score", 5.0) if t_res.get("available") else None
    C = stock_score.get("composite", 5.0)
    Q = q_res.get("score", stock_score.get("quality_score", 5.0))

    fund_bd = f_res.get("breakdown", {})
    tech_bd = t_res.get("breakdown", {})
    qual_bd = q_res.get("breakdown", {}) if isinstance(q_res, dict) else {}

    # ── Short-term (T-dominant) ────────────────────────────────────────────────
    if T is not None:
        st_raw = T * 0.60 + C * 0.30 + Q * 0.10
    else:
        st_raw = C * 0.85 + Q * 0.15
    st_100 = round(min(100, max(0, st_raw * 10)), 1)

    # ── Medium-term ────────────────────────────────────────────────────────────
    if F is not None and T is not None:
        mt_raw = F * 0.40 + T * 0.35 + Q * 0.25
    elif F is not None:
        mt_raw = F * 0.55 + Q * 0.30 + C * 0.15
    elif T is not None:
        mt_raw = T * 0.55 + Q * 0.30 + C * 0.15
    else:
        mt_raw = C * 0.70 + Q * 0.30
    mt_100 = round(min(100, max(0, mt_raw * 10)), 1)

    # ── Long-term (F-dominant) ─────────────────────────────────────────────────
    M_raw = stock_score.get("macro_score", 5.0)
    if F is not None:
        lt_raw = F * 0.55 + Q * 0.30 + M_raw * 0.10 + C * 0.05
    else:
        lt_raw = Q * 0.40 + C * 0.45 + M_raw * 0.15
    lt_100 = round(min(100, max(0, lt_raw * 10)), 1)

    return {
        "short_term": {
            "score": st_100,
            "label": _label_st(st_100),
            "confidence": _confidence(st_100),
            "horizon": "1-4 weeks",
            "driver": "Technical momentum",
            "bullets": _st_bullets(tech_bd, st_100),
        },
        "medium_term": {
            "score": mt_100,
            "label": _label_mt(mt_100),
            "confidence": _confidence(mt_100),
            "horizon": "1-6 months",
            "driver": "Fundamental + Technical blend",
            "bullets": _mt_bullets(fund_bd, tech_bd),
        },
        "long_term": {
            "score": lt_100,
            "label": _label_lt(lt_100),
            "confidence": _confidence(lt_100),
            "horizon": "1-3 years",
            "driver": "Fundamental quality",
            "bullets": _lt_bullets(fund_bd, qual_bd),
        },
    }
