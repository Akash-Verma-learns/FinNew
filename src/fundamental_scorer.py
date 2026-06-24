"""
src/fundamental_scorer.py

Fundamental Score (F) from EDGAR XBRL data.
Formula: F = weighted average of ROE, Net Margin, Revenue Growth, EPS Growth,
             Debt/Equity, FCF Margin, ROIC (re-normalized over available components).
Each component normalized to 0-10. Missing components are excluded from the weighted average.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_RANGES = {
    "revenue_growth": (-20, 50),
    "eps_growth":     (-30, 50),
    "roe":            (-10, 30),
    "net_margin":     (-5,  30),
    "debt_equity":    (-3,   0),   # inverted before normalizing
    "fcf_margin":     (-10, 40),
    "roic":           (-5,  35),
}

_WEIGHTS = {
    "revenue_growth": 0.25,
    "eps_growth":     0.20,
    "roe":            0.20,
    "net_margin":     0.15,
    "debt_equity":    0.10,
    "fcf_margin":     0.05,
    "roic":           0.05,
}


def _norm(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 5.0
    return max(0.0, min(10.0, (value - lo) / (hi - lo) * 10))


async def compute_fundamental_score(ticker: str) -> dict:
    """
    Returns {"score": float, "available": bool, "breakdown": dict, ...}.
    Degrades gracefully when DB is unavailable or ticker has no XBRL data.
    """
    from .db import collections, is_connected
    from .xbrl_lookup import get_cik

    if not is_connected():
        return {"score": 5.0, "available": False, "breakdown": {}, "reason": "db_unavailable"}

    cik = await get_cik(ticker)
    if not cik:
        return {"score": 5.0, "available": False, "breakdown": {}, "reason": "ticker_not_found"}

    breakdown: dict = {}
    component_scores: dict[str, float] = {}

    # ── Derived ratios from derivation_log (pre-computed by FormulaGraph) ──────
    try:
        from pymongo import DESCENDING
        cursor = collections.derivation_log.find(
            {"cik": cik, "concept": {"$in": [
                "ROE", "NetMarginPct", "DebtToEquity", "FCFMarginPct", "ROIC"
            ]}},
            sort=[("period_end", DESCENDING)],
        )
        latest: dict[str, float] = {}
        async for doc in cursor:
            c = doc["concept"]
            if c not in latest and doc.get("value") is not None:
                latest[c] = doc["value"]

        if "ROE" in latest:
            s = _norm(latest["ROE"], *_RANGES["roe"])
            component_scores["roe"] = s
            breakdown["roe_pct"] = round(latest["ROE"], 2)
            breakdown["roe_score"] = round(s, 2)

        if "NetMarginPct" in latest:
            s = _norm(latest["NetMarginPct"], *_RANGES["net_margin"])
            component_scores["net_margin"] = s
            breakdown["net_margin_pct"] = round(latest["NetMarginPct"], 2)
            breakdown["net_margin_score"] = round(s, 2)

        if "DebtToEquity" in latest:
            s = _norm(-latest["DebtToEquity"], *_RANGES["debt_equity"])
            component_scores["debt_equity"] = s
            breakdown["debt_to_equity"] = round(latest["DebtToEquity"], 2)
            breakdown["debt_equity_score"] = round(s, 2)

        if "FCFMarginPct" in latest:
            s = _norm(latest["FCFMarginPct"], *_RANGES["fcf_margin"])
            component_scores["fcf_margin"] = s
            breakdown["fcf_margin_pct"] = round(latest["FCFMarginPct"], 2)
            breakdown["fcf_margin_score"] = round(s, 2)

        if "ROIC" in latest:
            s = _norm(latest["ROIC"], *_RANGES["roic"])
            component_scores["roic"] = s
            breakdown["roic_pct"] = round(latest["ROIC"], 2)
            breakdown["roic_score"] = round(s, 2)

    except Exception as exc:
        logger.warning("[fundamental] derivation_log query failed for %s: %s", ticker, exc)

    # ── YoY growth from xbrl_facts (revenue and EPS) ──────────────────────────
    try:
        from pymongo import DESCENDING as D
        for alias, key, lo, hi in [
            ("revenue", "revenue_growth", -20, 50),
            ("eps",     "eps_growth",     -30, 50),
        ]:
            docs = await collections.xbrl_facts.find(
                {"cik": cik, "metric_alias": alias},
                sort=[("period_end", D)],
            ).to_list(length=3)
            if len(docs) >= 2:
                curr = docs[0].get("value") or 0
                prev = docs[1].get("value") or 0
                if prev:
                    growth_pct = (curr - prev) / abs(prev) * 100
                    s = _norm(growth_pct, lo, hi)
                    component_scores[key] = s
                    breakdown[f"{key}_pct"] = round(growth_pct, 2)
                    breakdown[f"{key}_score"] = round(s, 2)

    except Exception as exc:
        logger.warning("[fundamental] xbrl_facts growth query failed for %s: %s", ticker, exc)

    # ── FCF Margin fallback: compute from FreeCashFlow + Revenue if not in derivation_log ──
    if "fcf_margin" not in component_scores:
        try:
            from pymongo import DESCENDING as D3
            fcf_doc = await collections.derivation_log.find_one(
                {"cik": cik, "concept": "FreeCashFlow"},
                sort=[("period_end", D3)],
            )
            rev_docs = await collections.xbrl_facts.find(
                {"cik": cik, "metric_alias": "revenue"},
                sort=[("period_end", D3)],
            ).to_list(length=1)
            if fcf_doc and rev_docs and rev_docs[0].get("value"):
                fcf = fcf_doc.get("value")
                rev = rev_docs[0]["value"]
                if fcf is not None and rev:
                    fcf_margin = fcf / rev * 100
                    s = _norm(fcf_margin, *_RANGES["fcf_margin"])
                    component_scores["fcf_margin"] = s
                    breakdown["fcf_margin_pct"] = round(fcf_margin, 2)
                    breakdown["fcf_margin_score"] = round(s, 2)
        except Exception as exc:
            logger.debug("[fundamental] FCF margin fallback failed for %s: %s", ticker, exc)

    # ── ROIC fallback: compute from OperatingIncome + Debt + Equity ───────────
    if "roic" not in component_scores:
        try:
            from pymongo import DESCENDING as D4
            roic_inputs: dict[str, float] = {}
            for concept in ["OperatingIncomeLoss", "LongTermDebt", "StockholdersEquity"]:
                doc = await collections.derivation_log.find_one(
                    {"cik": cik, "concept": concept},
                    sort=[("period_end", D4)],
                )
                if not doc:
                    doc = await collections.xbrl_facts.find_one(
                        {"cik": cik, "metric_alias": concept.lower()},
                        sort=[("period_end", D4)],
                    )
                if doc and doc.get("value") is not None:
                    roic_inputs[concept] = doc["value"]

            if all(k in roic_inputs for k in ["OperatingIncomeLoss", "LongTermDebt", "StockholdersEquity"]):
                invested = abs(roic_inputs["LongTermDebt"] + roic_inputs["StockholdersEquity"])
                if invested > 0:
                    roic_val = roic_inputs["OperatingIncomeLoss"] * 0.80 / invested * 100
                    s = _norm(roic_val, *_RANGES["roic"])
                    component_scores["roic"] = s
                    breakdown["roic_pct"] = round(roic_val, 2)
                    breakdown["roic_score"] = round(s, 2)
        except Exception as exc:
            logger.debug("[fundamental] ROIC fallback failed for %s: %s", ticker, exc)

    if not component_scores:
        return {"score": 5.0, "available": False, "breakdown": breakdown, "reason": "no_data"}

    total_w = sum(_WEIGHTS[k] for k in component_scores if k in _WEIGHTS)
    if not total_w:
        return {"score": 5.0, "available": False, "breakdown": breakdown, "reason": "no_weights"}

    f_score = sum(
        component_scores[k] * (_WEIGHTS[k] / total_w)
        for k in component_scores
        if k in _WEIGHTS
    )

    return {
        "score": round(min(10.0, max(0.0, f_score)), 2),
        "available": True,
        "breakdown": breakdown,
        "components_used": sorted(component_scores.keys()),
    }
