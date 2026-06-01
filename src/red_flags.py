from __future__ import annotations

import re

from .models import Claim, ClaimType, RedFlag, ValidationResult, ValidationStatus

# NOTE: keyword matching is intentionally simple for now.
# "risk" in "risk-adjusted return" would count as bearish; "growth" in "growth has slowed"
# would count as bullish. Context-aware NLP is a planned improvement.
BULLISH_WORDS = frozenset({"buy", "upgrade", "outperform", "strong", "growth", "opportunity", "upside", "beat", "exceed"})
BEARISH_WORDS = frozenset({"sell", "downgrade", "underperform", "risk", "headwind", "concern", "decline", "miss", "below"})


def _extract_numeric(text: str) -> float | None:
    m = re.search(r"([\d.]+)\s*%?", text or "")
    return float(m.group(1)) if m else None


def check_red_flags(
    claims: list[Claim],
    validations: dict[str, ValidationResult],
) -> list[RedFlag]:
    flags: list[RedFlag] = []
    by_type: dict[str, list[Claim]] = {}
    for c in claims:
        by_type.setdefault(c.type, []).append(c)

    _default = lambda cid: ValidationResult(claim_id=cid)

    contradicted_facts = [
        c for c in by_type.get(ClaimType.DIRECT_FACT, [])
        if validations.get(c.id, _default(c.id)).status == ValidationStatus.CONTRADICTED
    ]
    if len(contradicted_facts) >= 2:
        flags.append(RedFlag(
            severity="HIGH",
            message=f"{len(contradicted_facts)} direct facts contradicted by SEC filings — foundational data unreliable",
        ))

    for c in by_type.get(ClaimType.ACCOUNTING_POLICY, []):
        if validations.get(c.id, _default(c.id)).status == ValidationStatus.CONTRADICTED:
            flags.append(RedFlag(severity="HIGH", message="Accounting policy claim contradicted by filing evidence"))

    for c in by_type.get(ClaimType.MODEL_ASSUMPTION, []):
        metric = (c.metric or "").lower()
        val = _extract_numeric(c.value or "")
        if val is None:
            continue
        if "wacc" in metric and val < 7:
            flags.append(RedFlag(severity="MEDIUM", message=f"Aggressive WACC assumption: {val}% (below 7% threshold)"))
        if "terminal" in metric and "growth" in metric and val > 4:
            flags.append(RedFlag(severity="MEDIUM", message=f"Aggressive terminal growth rate: {val}% (above 4% threshold)"))
        if ("p/e" in metric or "pe" in metric or ("price" in metric and "earnings" in metric)) and val > 35:
            flags.append(RedFlag(severity="MEDIUM", message=f"Excessive P/E multiple: {val}x (above 35x threshold)"))

    recs = by_type.get(ClaimType.RECOMMENDATION, [])
    if recs:
        bullish_recs = [
            c for c in recs
            if any(w in c.raw_text.lower() for w in ("buy", "upgrade", "outperform", "strong buy"))
        ]
        if len(bullish_recs) == len(recs):
            flags.append(RedFlag(
                severity="LOW",
                message="All recommendations are bullish — no bearish counterbalance in report",
            ))

    for c in by_type.get(ClaimType.ACCOUNTING_POLICY, []):
        if validations.get(c.id, _default(c.id)).status == ValidationStatus.UNVERIFIABLE:
            flags.append(RedFlag(severity="LOW", message="Accounting policy claim could not be verified against filing"))

    return flags


def detect_bias(claims: list[Claim]) -> str:
    bullish_count = 0
    bearish_count = 0
    for c in claims:
        words = set(re.findall(r"\b\w+\b", c.raw_text.lower()))
        bullish_count += len(words & BULLISH_WORDS)
        bearish_count += len(words & BEARISH_WORDS)
    if bullish_count > bearish_count * 2:
        return "BULLISH"
    if bearish_count > bullish_count * 2:
        return "BEARISH"
    return "BALANCED"
