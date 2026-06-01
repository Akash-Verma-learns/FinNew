from __future__ import annotations

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


class RedFlag(BaseModel):
    severity: str
    message: str


class CascadeInfo(BaseModel):
    root_claim_id: str
    affected_ids: list[str]
    message: str
