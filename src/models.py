from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, RootModel


class ClaimType(str, Enum):
    DIRECT_FACT = "DIRECT_FACT"
    DERIVED_METRIC = "DERIVED_METRIC"
    ACCOUNTING_POLICY = "ACCOUNTING_POLICY"
    MODEL_ASSUMPTION = "MODEL_ASSUMPTION"
    FORWARD_PROJECTION = "FORWARD_PROJECTION"
    RECOMMENDATION = "RECOMMENDATION"
    QUALITATIVE = "QUALITATIVE"


class ValidationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIABLE = "UNVERIFIABLE"
    CONTRADICTED = "CONTRADICTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    ERROR = "ERROR"


class Claim(BaseModel):
    id: str
    raw_text: str
    type: ClaimType
    company: Optional[str] = None
    ticker: Optional[str] = None
    metric: Optional[str] = None
    value: Optional[str] = None
    period: Optional[str] = None
    checkable: bool = False


class ClaimList(RootModel[list[Claim]]):
    pass


class FormulaDefinition(BaseModel):
    source: str        # "company_10k" | "fasb_linkbase" | "textbook"
    value: float
    label: str = ""    # human-readable source label


class ValidationResult(BaseModel):
    claim_id: str
    status: ValidationStatus = ValidationStatus.UNVERIFIABLE
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    actual_value: Optional[str] = None
    discrepancy: Optional[str] = None
    filing_source: Optional[str] = None
    evidence: Optional[str] = None
    citations: list[str] = []
    reasoning: str = ""
    flag: Optional[str] = None
    # EDGAR provenance — populated when value comes from XBRL lookup
    cik: Optional[str] = None
    accession_number: Optional[str] = None
    filing_date: Optional[str] = None
    edgar_url: Optional[str] = None
    # FormulaGraph multi-resolution — populated for derived metrics
    formula_concept: Optional[str] = None      # e.g. "GrossMarginPct"
    formula_source: Optional[str] = None       # "company_10k" | "fasb_linkbase" | "textbook"
    formula_inputs: dict[str, float] = {}      # {"GrossProfit": 180683e6, "Revenues": 391035e6}
    all_definitions: list[FormulaDefinition] = []  # all sources that produced a value
    definition_delta: Optional[float] = None   # max - min across all definitions
    # Structured citations (richer than the URL-only citations list)
    structured_citations: list[Citation] = []
    # Conflicts detected between sources (e.g. XBRL GAAP vs 8-K non-GAAP)
    source_conflicts: list[SourceConflict] = []


class Citation(BaseModel):
    source: str          # EDGAR_XBRL | 10K_TEXT | 8K_NONGAAP | SEGMENT | WEB | FORMULA
    label: str           # Human-readable source name
    url: Optional[str] = None
    ticker: Optional[str] = None
    filing: Optional[str] = None       # "10-K 2024", "8-K Q4 2024"
    accession: Optional[str] = None
    field: Optional[str] = None        # XBRL concept or metric name
    value: Optional[str] = None        # formatted value found in source
    section: Optional[str] = None      # document section or note
    excerpt: Optional[str] = None      # short text quote from filing
    period: Optional[str] = None


class SourceConflict(BaseModel):
    source_a: str
    source_b: str
    value_a: str
    value_b: str
    difference_pct: Optional[float] = None
    message: str


class RedFlag(BaseModel):
    severity: str
    message: str


class CascadeInfo(BaseModel):
    root_claim_id: str
    affected_ids: list[str]
    message: str


class AuditLog(BaseModel):
    validation_id: str
    created_at: datetime
    input_summary: str          # first 400 chars of input text
    input_type: str             # "text" | "pdf"
    ticker: Optional[str] = None
    llm_model: str
    llm_backend: str            # "ollama" | "groq"
    embedding_model: str
    claims_count: int
    verified_count: int
    contradicted_count: int
    partially_verified_count: int
    unverifiable_count: int
    red_flag_count: int
    overall_score: float
    credibility_rating: str
    analyst_bias: str
    extraction_seconds: float
    validation_seconds: float
    scoring_seconds: float
    total_seconds: float
    claims: list[dict]          # full claim dicts
    validations: list[dict]     # full validation result dicts
    red_flags: list[dict]


class FeedbackSignal(BaseModel):
    feedback_id: str
    validation_id: str
    claim_id: Optional[str] = None   # None = overall report feedback
    feedback_type: str               # "evidence_relevance" | "claim_quality" | "contradiction_accuracy" | "overall"
    feedback_value: str              # "positive" | "negative" | "1"-"5"
    notes: Optional[str] = None
    created_at: datetime


class CredibilitySnapshot(BaseModel):
    ticker: str
    run_id: str
    timestamp: datetime
    overall_score: float
    credibility_rating: str
    analyst_bias: str
    report_type: str
    total_claims: int
    verified_count: int
    contradicted_count: int
    partially_verified_count: int
    unverifiable_count: int
    red_flag_count: int
    high_severity_flags: list[str]
    source_label: str


class StockScore(BaseModel):
    ticker: str
    generated_at: datetime

    # Component sub-scores (0–10 scale; None when data unavailable)
    fundamental_score: Optional[float] = None
    technical_score: Optional[float] = None
    qualitative_score: float                    # from credibility snapshot, or 5.0 neutral
    macro_score: float                          # placeholder 5.0 pending sector data

    # Composite output
    composite_score: float
    signal_label: str                           # "HIGH/MODERATE/MIXED/LOW historical alignment"

    # Data availability
    fundamental_available: bool
    technical_available: bool
    qualitative_from_snapshot: bool             # False → no prior runs, Q defaulted to 5.0

    # Weights used
    weights: dict

    # Per-component breakdowns
    fundamental_breakdown: Optional[dict] = None
    technical_breakdown: Optional[dict] = None
    qualitative_breakdown: Optional[dict] = None

    disclaimer: str


class TrendInsight(BaseModel):
    ticker: str
    generated_at: datetime

    # Source 1: credibility history
    history_available: bool
    snapshot_count: int
    avg_credibility_score: Optional[float] = None
    credibility_trend: Optional[str] = None
    credibility_trend_detail: Optional[str] = None
    bias_pattern: Optional[str] = None
    past_contradiction_rate: Optional[float] = None

    # Source 2: price context
    price_available: bool
    current_price: Optional[float] = None
    price_52w_high: Optional[float] = None
    price_52w_low: Optional[float] = None
    price_change_30d_pct: Optional[float] = None
    price_change_90d_pct: Optional[float] = None
    price_volatility_note: Optional[str] = None
    credibility_price_pattern: Optional[str] = None

    # Source 3: current report
    current_score: float
    current_rating: str
    current_bias: str
    current_verified_count: int
    current_contradicted_count: int
    current_high_flags: list[str]
    notable_verified_claims: list[str]
    notable_contradicted_claims: list[str]

    # Synthesized educational summary
    summary_headline: str
    pattern_observations: list[str]
    data_gaps: list[str]
    disclaimer: str
    forward_looking_note: Optional[str] = None
