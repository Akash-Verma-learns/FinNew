from __future__ import annotations

from .models import CascadeInfo, Claim, ClaimType, RedFlag, ValidationResult, ValidationStatus
from .red_flags import check_red_flags, detect_bias

CLAIM_WEIGHTS: dict[str, float] = {
    ClaimType.DIRECT_FACT: 1.0,
    ClaimType.DERIVED_METRIC: 0.9,
    ClaimType.ACCOUNTING_POLICY: 0.85,
    ClaimType.MODEL_ASSUMPTION: 0.7,
    ClaimType.FORWARD_PROJECTION: 0.6,
    ClaimType.RECOMMENDATION: 0.4,
    ClaimType.QUALITATIVE: 0.25,
}

STATUS_SCORES: dict[ValidationStatus, float] = {
    ValidationStatus.VERIFIED: 1.0,
    ValidationStatus.PARTIALLY_VERIFIED: 0.75,
    ValidationStatus.UNVERIFIABLE: 0.70,   # "couldn't check" ≠ "probably wrong"
    ValidationStatus.CONTRADICTED: 0.0,
    ValidationStatus.NOT_APPLICABLE: 0.70,
    ValidationStatus.ERROR: 0.5,
}

CASCADE_DEPS: dict[str, list[str]] = {
    ClaimType.ACCOUNTING_POLICY: [ClaimType.DERIVED_METRIC, ClaimType.FORWARD_PROJECTION],
    ClaimType.DIRECT_FACT: [ClaimType.DERIVED_METRIC, ClaimType.FORWARD_PROJECTION, ClaimType.MODEL_ASSUMPTION],
    ClaimType.DERIVED_METRIC: [ClaimType.FORWARD_PROJECTION, ClaimType.MODEL_ASSUMPTION],
    ClaimType.FORWARD_PROJECTION: [ClaimType.RECOMMENDATION],
}

SEVERITY_PENALTIES = {"HIGH": 10, "MEDIUM": 5, "LOW": 2}


def score_report(
    claims: list[Claim],
    validations: dict[str, ValidationResult],
) -> dict:
    total_weight = 0.0
    weighted_sum = 0.0
    type_scores: dict[str, list[float]] = {}

    for claim in claims:
        v = validations.get(claim.id)
        if not v:
            continue
        # Non-checkable claims have no numeric value to verify — they default to
        # UNVERIFIABLE and add no signal.  Counting them drags the weighted average
        # toward 0.25 (UNVERIFIABLE × min_confidence floor) even when all numeric
        # facts are correct.  Skip them from the score; they still appear in the UI.
        if not claim.checkable:
            continue
        weight = CLAIM_WEIGHTS.get(claim.type, 0.5)
        status_score = STATUS_SCORES.get(v.status, 0.5)
        confidence = max(v.confidence, 0.65) if v.status == ValidationStatus.UNVERIFIABLE else v.confidence
        contribution = weight * status_score * confidence
        weighted_sum += contribution
        total_weight += weight
        type_scores.setdefault(claim.type, []).append(status_score * confidence * 100)

    raw_score = (weighted_sum / total_weight * 100) if total_weight > 0 else 50.0

    claim_by_type: dict[str, list[Claim]] = {}
    for c in claims:
        claim_by_type.setdefault(c.type, []).append(c)

    cascades: list[CascadeInfo] = []
    cascade_penalty = 0.0
    # Cap per claim-type: once we know a given root type has contradicted claims,
    # the full cascade signal is captured by a single penalty entry.  Adding N×15
    # for every contradicted root claim in a large filing (e.g. 44 DIRECT_FACTs
    # from a "(in millions)" table → 660 penalty) would overwhelm the raw score
    # (max ~100) even on a report with no real data problems.
    for root_type, dep_types in CASCADE_DEPS.items():
        any_contradicted_for_type = False
        for root_claim in claim_by_type.get(root_type, []):
            v = validations.get(root_claim.id)
            if v and v.status == ValidationStatus.CONTRADICTED:
                affected = [c.id for dt in dep_types for c in claim_by_type.get(dt, [])]
                if affected:
                    cascades.append(CascadeInfo(
                        root_claim_id=root_claim.id,
                        affected_ids=affected,
                        message=f"Contradicted {root_type} invalidates downstream: {', '.join(dep_types)}",
                    ))
                    if not any_contradicted_for_type:
                        cascade_penalty += 3 * len(dep_types)
                        any_contradicted_for_type = True

    red_flags: list[RedFlag] = check_red_flags(claims, validations)
    flag_penalty = sum(SEVERITY_PENALTIES.get(f.severity, 0) for f in red_flags)

    final_score = max(0.0, min(100.0, raw_score - cascade_penalty - flag_penalty))
    breakdown = {t: round(sum(scores) / len(scores), 1) for t, scores in type_scores.items()}

    verified = sum(1 for v in validations.values() if v.status == ValidationStatus.VERIFIED)
    contradicted = sum(1 for v in validations.values() if v.status == ValidationStatus.CONTRADICTED)

    # Determine what fraction of checkable claims are inherently unverifiable types
    # (QUALITATIVE = ESG/product metrics, FORWARD_PROJECTION = guidance/targets).
    # When these dominate the document, the score reflects verifiability of claim types,
    # not document accuracy — adjust thresholds and surface an explanatory note.
    checkable_claims = [c for c in claims if c.checkable]
    unverifiable_types = {ClaimType.QUALITATIVE, ClaimType.FORWARD_PROJECTION}
    unverifiable_count = sum(1 for c in checkable_claims if c.type in unverifiable_types)
    unverifiable_fraction = unverifiable_count / len(checkable_claims) if checkable_claims else 0.0

    # For documents dominated by unverifiable claims, use relaxed rating thresholds:
    # the evidence infrastructure (XBRL, SEC filings) simply cannot confirm ESG targets
    # or forward projections — a "LOW" rating on such a document is misleading.
    if unverifiable_fraction >= 0.7:
        if final_score >= 65:
            rating = "HIGH"
        elif final_score >= 48:
            rating = "MODERATE"
        elif final_score >= 32:
            rating = "LOW"
        else:
            rating = "VERY LOW"
    else:
        if final_score >= 80:
            rating = "HIGH"
        elif final_score >= 60:
            rating = "MODERATE"
        elif final_score >= 40:
            rating = "LOW"
        else:
            rating = "VERY LOW"

    verifiability_note: str | None = None
    if unverifiable_fraction >= 0.7:
        verifiability_note = (
            f"{unverifiable_count} of {len(checkable_claims)} checkable claims are "
            f"inherently unverifiable types (ESG targets, forward projections, product metrics). "
            f"Score reflects claim verifiability, not document accuracy — "
            f"no evidence infrastructure exists to confirm or deny these claims."
        )

    return {
        "overall_score": round(final_score, 1),
        "credibility_rating": rating,
        "breakdown": breakdown,
        "cascades": [c.model_dump() for c in cascades],
        "red_flags": [f.model_dump() for f in red_flags],
        "analyst_bias": detect_bias(claims),
        "verified_count": verified,
        "contradicted_count": contradicted,
        "claim_count": len(claims),
        "verifiability_note": verifiability_note,
    }
