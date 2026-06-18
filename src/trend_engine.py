"""
src/trend_engine.py

Fuses credibility history, price context, and current report validation
into an educational TrendInsight for the UNBOUNDX platform.

COMPLIANCE: Never use the words 'buy', 'sell', 'invest', 'recommend', or 'should'.
Frame all findings as historical observations. No price targets or return projections.
"""
from __future__ import annotations

import logging
from datetime import datetime
from statistics import mean, stdev
from typing import Optional

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "For educational purposes only. This is not investment advice. "
    "Past patterns do not guarantee future results. "
    "Brokerage services provided by MARV Capital Inc."
)
FWD_NOTE = "Forward-looking statements are unverifiable and carry inherent uncertainty."


async def build_trend_insight(
    ticker: str,
    validations: list[dict],
    claims: list,
    scoring: dict,
) -> "TrendInsight":
    """
    Main entry point. Fetches all three data sources and synthesizes
    an educational TrendInsight for the given ticker.
    """
    from .models import TrendInsight
    from .price_fetcher import fetch_price_context
    from .historical_claims import get_ticker_history

    ticker_upper = ticker.upper()

    # Gather all sources concurrently
    import asyncio
    price_ctx, snapshots = await asyncio.gather(
        fetch_price_context(ticker_upper),
        get_ticker_history(ticker_upper, limit=10),
    )

    history_ctx = _analyze_history(snapshots)
    current_ctx = _analyze_current(validations, claims, scoring)

    corr = _correlate_price_credibility(price_ctx, snapshots)
    price_ctx = {**price_ctx, "credibility_price_pattern": corr}

    has_fwd = any(
        c.type.value == "FORWARD_PROJECTION"
        for c in claims
        if hasattr(c, "type")
    )

    return TrendInsight(
        ticker=ticker_upper,
        generated_at=datetime.utcnow(),
        # history
        history_available=history_ctx["history_available"],
        snapshot_count=history_ctx["snapshot_count"],
        avg_credibility_score=history_ctx.get("avg_credibility_score"),
        credibility_trend=history_ctx.get("credibility_trend"),
        credibility_trend_detail=history_ctx.get("credibility_trend_detail"),
        bias_pattern=history_ctx.get("bias_pattern"),
        past_contradiction_rate=history_ctx.get("past_contradiction_rate"),
        # price
        price_available=price_ctx["price_available"],
        current_price=price_ctx.get("current_price"),
        price_52w_high=price_ctx.get("price_52w_high"),
        price_52w_low=price_ctx.get("price_52w_low"),
        price_change_30d_pct=price_ctx.get("price_change_30d_pct"),
        price_change_90d_pct=price_ctx.get("price_change_90d_pct"),
        price_volatility_note=price_ctx.get("price_volatility_note"),
        credibility_price_pattern=price_ctx.get("credibility_price_pattern"),
        # current report
        current_score=current_ctx["current_score"],
        current_rating=current_ctx["current_rating"],
        current_bias=current_ctx["current_bias"],
        current_verified_count=current_ctx["current_verified_count"],
        current_contradicted_count=current_ctx["current_contradicted_count"],
        current_high_flags=current_ctx["current_high_flags"],
        notable_verified_claims=current_ctx["notable_verified_claims"],
        notable_contradicted_claims=current_ctx["notable_contradicted_claims"],
        # synthesis
        summary_headline=_build_headline(history_ctx, current_ctx),
        pattern_observations=_build_observations(history_ctx, price_ctx, current_ctx),
        data_gaps=_build_data_gaps(history_ctx, price_ctx),
        disclaimer=DISCLAIMER,
        forward_looking_note=FWD_NOTE if has_fwd else None,
    )


# ---------------------------------------------------------------------------
# Source analysis helpers
# ---------------------------------------------------------------------------

def _analyze_history(snapshots: list) -> dict:
    """Extract trend signals from past CredibilitySnapshots."""
    if not snapshots:
        return {"history_available": False, "snapshot_count": 0}

    scores = [s.overall_score for s in snapshots]
    avg_score = round(mean(scores), 2)

    trend: Optional[str] = None
    trend_detail: Optional[str] = None
    if len(scores) >= 3:
        recent_avg = mean(scores[:3])
        older_avg = mean(scores[3:]) if len(scores) > 3 else mean(scores)
        delta = recent_avg - older_avg
        if abs(delta) < 3:
            trend = "STABLE"
            trend_detail = (
                f"Credibility scores have historically been stable "
                f"(avg {avg_score:.1f} across {len(snapshots)} reports)"
            )
        elif delta > 0:
            trend = "IMPROVING"
            trend_detail = (
                f"Recent reports have historically scored higher than earlier ones "
                f"(+{delta:.1f} pts shift observed)"
            )
        else:
            trend = "DECLINING"
            trend_detail = (
                f"Recent reports have historically scored lower than earlier ones "
                f"({delta:.1f} pts shift observed)"
            )
    elif len(scores) == 2:
        delta = scores[0] - scores[1]
        trend = "IMPROVING" if delta > 0 else ("DECLINING" if delta < 0 else "STABLE")
        trend_detail = f"Two reports available; score moved {delta:+.1f} pts between them."

    # Bias pattern
    biases = [s.analyst_bias for s in snapshots]
    bias_counter: dict[str, int] = {}
    for b in biases:
        bias_counter[b] = bias_counter.get(b, 0) + 1
    dominant_bias, dom_count = max(bias_counter.items(), key=lambda x: x[1])
    bias_pattern: Optional[str] = None
    if dom_count / len(biases) >= 0.6:
        bias_pattern = (
            f"Historically {dominant_bias.lower()} bias has been detected in "
            f"{dom_count} of {len(biases)} reports analyzed"
        )

    # Contradiction rate
    total_claims = sum(s.total_claims for s in snapshots)
    total_contradicted = sum(s.contradicted_count for s in snapshots)
    past_contradiction_rate = (
        round(total_contradicted / total_claims * 100, 1)
        if total_claims > 0 else None
    )

    return {
        "history_available": True,
        "snapshot_count": len(snapshots),
        "avg_credibility_score": avg_score,
        "credibility_trend": trend,
        "credibility_trend_detail": trend_detail,
        "bias_pattern": bias_pattern,
        "past_contradiction_rate": past_contradiction_rate,
    }


def _analyze_current(validations: list[dict], claims: list, scoring: dict) -> dict:
    """Summarize the current pipeline run for TrendInsight fields."""
    from .models import ValidationStatus

    verified_count = sum(
        1 for v in validations if v.get("status") == ValidationStatus.VERIFIED.value
    )
    contradicted_count = sum(
        1 for v in validations if v.get("status") == ValidationStatus.CONTRADICTED.value
    )
    high_flags = [
        f["message"] for f in scoring.get("red_flags", [])
        if f.get("severity") == "HIGH"
    ]

    # Build claim text lookup
    claim_text: dict[str, str] = {}
    for c in claims:
        if hasattr(c, "id") and hasattr(c, "raw_text"):
            claim_text[c.id] = c.raw_text
        elif isinstance(c, dict):
            claim_text[c.get("id", "")] = c.get("raw_text", "")

    notable_verified: list[str] = []
    notable_contradicted: list[str] = []
    for v in validations:
        cid = v.get("claim_id", "")
        raw = claim_text.get(cid, "")
        if not raw:
            continue
        short = raw[:120] + ("…" if len(raw) > 120 else "")
        if v.get("status") == ValidationStatus.VERIFIED.value and len(notable_verified) < 3:
            notable_verified.append(short)
        elif v.get("status") == ValidationStatus.CONTRADICTED.value and len(notable_contradicted) < 3:
            notable_contradicted.append(short)

    return {
        "current_score": scoring.get("overall_score", 0.0),
        "current_rating": scoring.get("credibility_rating", "UNKNOWN"),
        "current_bias": scoring.get("analyst_bias", "BALANCED"),
        "current_verified_count": verified_count,
        "current_contradicted_count": contradicted_count,
        "current_high_flags": high_flags,
        "notable_verified_claims": notable_verified,
        "notable_contradicted_claims": notable_contradicted,
    }


def _correlate_price_credibility(price_ctx: dict, snapshots: list) -> Optional[str]:
    """
    Compute an educational observation about whether price movements and
    credibility score changes have historically coincided. Returns None if
    insufficient data.
    """
    if not price_ctx.get("price_available") or len(snapshots) < 3:
        return None

    price_history = price_ctx.get("_price_history", [])
    if not price_history or len(price_history) < 2:
        return None

    # Compare credibility trend direction to 90-day price direction
    change_90d = price_ctx.get("price_change_90d_pct")
    if change_90d is None:
        return None

    scores = [s.overall_score for s in snapshots]
    score_delta = scores[0] - scores[-1]

    if score_delta > 5 and change_90d > 5:
        return (
            "Historically, periods of rising credibility scores have coincided with "
            "positive price movement for this ticker (educational observation only)"
        )
    elif score_delta < -5 and change_90d < -5:
        return (
            "Historically, periods of declining credibility scores have coincided with "
            "negative price movement for this ticker (educational observation only)"
        )
    elif abs(score_delta) > 5 and abs(change_90d) < 3:
        return (
            "Credibility score changes have historically not strongly coincided with "
            "short-term price movements for this ticker (educational observation only)"
        )
    return None


def _build_headline(history_ctx: dict, current_ctx: dict) -> str:
    """Single educational headline summarizing the current report in historical context."""
    rating = current_ctx["current_rating"]
    score = current_ctx["current_score"]
    ticker_phrase = f"Current report scored {score:.1f} ({rating})"

    if history_ctx["history_available"]:
        avg = history_ctx.get("avg_credibility_score")
        trend = history_ctx.get("credibility_trend")
        hist_phrase = f"; historical average is {avg:.1f}" if avg else ""
        trend_phrase = f" with a {trend.lower()} pattern" if trend else ""
        return f"{ticker_phrase}{hist_phrase}{trend_phrase}."
    return f"{ticker_phrase}. No prior history available for comparison."


def _build_observations(
    history_ctx: dict,
    price_ctx: dict,
    current_ctx: dict,
) -> list[str]:
    """Build educational pattern observation bullets."""
    obs: list[str] = []

    # Current report observations
    contradicted = current_ctx["current_contradicted_count"]
    verified = current_ctx["current_verified_count"]
    if contradicted > 0:
        obs.append(
            f"The data shows {contradicted} claim(s) in this report contradicted EDGAR XBRL filings."
        )
    if verified > 0:
        obs.append(
            f"The data shows {verified} claim(s) in this report verified against SEC filings."
        )

    high_flags = current_ctx["current_high_flags"]
    for flag in high_flags[:2]:
        obs.append(f"High-severity flag observed: {flag}")

    # Historical observations
    if history_ctx["history_available"]:
        trend_detail = history_ctx.get("credibility_trend_detail")
        if trend_detail:
            obs.append(trend_detail)
        bias = history_ctx.get("bias_pattern")
        if bias:
            obs.append(bias)
        contra_rate = history_ctx.get("past_contradiction_rate")
        if contra_rate is not None:
            obs.append(
                f"Historically, {contra_rate:.1f}% of verifiable claims across past reports "
                "have coincided with SEC EDGAR contradictions."
            )

    # Price observations
    if price_ctx.get("price_available"):
        vol_note = price_ctx.get("price_volatility_note")
        if vol_note:
            obs.append(vol_note + " (educational context only).")
        corr = price_ctx.get("credibility_price_pattern")
        if corr:
            obs.append(corr)

        high_52w = price_ctx.get("price_52w_high")
        low_52w = price_ctx.get("price_52w_low")
        current_p = price_ctx.get("current_price")
        if high_52w and low_52w and current_p:
            range_pct = (current_p - low_52w) / (high_52w - low_52w) * 100 if (high_52w - low_52w) > 0 else None
            if range_pct is not None:
                obs.append(
                    f"Price is currently at {range_pct:.0f}% of its 52-week range "
                    f"(52w low: {low_52w:.2f}, 52w high: {high_52w:.2f}) — educational context only."
                )

    return obs


def _build_data_gaps(history_ctx: dict, price_ctx: dict) -> list[str]:
    """List data sources that were unavailable for this analysis."""
    gaps: list[str] = []
    if not history_ctx["history_available"]:
        gaps.append(
            "No prior credibility history found for this ticker — "
            "historical comparison is unavailable."
        )
    elif history_ctx["snapshot_count"] < 3:
        gaps.append(
            f"Only {history_ctx['snapshot_count']} prior report(s) found — "
            "trend analysis is limited with fewer than 3 data points."
        )
    if not price_ctx.get("price_available"):
        gaps.append(
            "Price data unavailable for this ticker — "
            "price context could not be included in this analysis."
        )
    return gaps
