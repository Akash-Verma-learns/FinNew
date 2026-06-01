# Architecture

## What the system does

The Financial Report Validator takes a research report (plain text or PDF) and checks whether its key claims are accurate. It does this by extracting specific factual claims, looking them up in authoritative sources, and producing a credibility score with an explanation.

---

## Pipeline overview

```
Input (text or PDF)
        │
        ▼
1. Text extraction (PDF only)
        │
        ▼
2. Claim extraction   ──── Groq LLM (llama-3.3-70b)
        │
        ▼
3. Claim validation   ──── SEC EDGAR XBRL API  (DIRECT_FACT)
        │                   Tavily web search   (all other types)
        │                   Groq LLM            (reasoning)
        ▼
4. Scoring            ──── scorer.js (pure JS, no LLM)
        │
        ▼
JSON response: claims + validations + score
```

---

## Step-by-step

### Step 1 — Text extraction (PDF path only)

`src/pdfParser.js` uses `pdf-parse` to extract raw text from the PDF buffer.

For large documents (up to ~200 pages), the full text can be 100,000+ characters. The
`prepareTextForAnalysis` function keeps the first 15,000 characters (which covers the
executive summary and key findings in most research reports) and appends up to 5,000
characters of financially significant sentences from the remainder (identified by the
presence of numbers, percentages, and dollar signs).

### Step 2 — Claim extraction

`src/claimExtractor.js` sends the prepared text to the Groq LLM with a structured prompt
asking it to identify 6–8 verifiable claims.

Each claim has:
- `type` — one of 7 categories (DIRECT_FACT, DERIVED_METRIC, ACCOUNTING_POLICY, MODEL_ASSUMPTION, FORWARD_PROJECTION, QUALITATIVE, RECOMMENDATION)
- `company` and `ticker` — used to route to the right data source
- `metric` — what is being measured (e.g. "total revenue")
- `value` — the figure as stated in the report
- `period` — the fiscal year (e.g. "FY2024")
- `checkable` — false for pure opinion claims

### Step 3 — Claim validation

`src/validator.js` applies a different strategy per claim type:

| Claim type | Strategy |
|---|---|
| DIRECT_FACT (with ticker) | SEC EDGAR XBRL lookup first; falls back to Tavily + Groq if not found |
| ACCOUNTING_POLICY | Retrieves latest 10-K filing URL from EDGAR, then searches SEC.gov via Tavily, then Groq |
| All others | Targeted Tavily web search + Groq reasoning |

Each validation returns: `status`, `confidence`, `actual_value`, `discrepancy`, `filing_source`, `evidence`, `reasoning`, `citations`, and `flag`.

**Validation statuses:**
- `VERIFIED` — confirmed with high confidence
- `PARTIALLY_VERIFIED` — directionally correct but with minor discrepancy
- `CONTRADICTED` — directly contradicts an authoritative source
- `UNVERIFIABLE` — no usable evidence found

### Step 4 — Scoring

`src/scorer.js` and `src/redFlags.js` compute the final score in pure JavaScript with no LLM involved.

**Weighted score:** each claim's validation status and confidence are combined using `CLAIM_WEIGHTS` (DIRECT_FACT = 1.0, down to QUALITATIVE = 0.25). The raw score is the weighted average × 100.

**Penalties applied:**
- **Cascade penalty:** if a foundational claim (e.g. an ACCOUNTING_POLICY) is contradicted, all downstream claim types (DERIVED_METRIC, FORWARD_PROJECTION, RECOMMENDATION) are flagged and a 5-point penalty is applied per cascade.
- **Red flag penalty:** rule-based checks fire deterministic penalties (HIGH = –10 pts, MEDIUM = –5 pts, LOW = –2 pts).

**Red flag rules** (see `src/redFlags.js`):
- 2+ contradicted direct facts
- Contradicted accounting policy
- WACC below 7%
- Terminal growth rate above 4%
- P/E multiple above 35×
- All recommendations bullish (no bearish balance)
- Unverifiable accounting policy

**Analyst bias detector** scans claim text for bullish/bearish language and returns BULLISH, BEARISH, or BALANCED.

---

## External APIs

| Service | What it's used for | Auth |
|---|---|---|
| Groq (llama-3.3-70b) | Claim extraction + validation reasoning | `GROQ_API_KEY` env var |
| Tavily | Web search for evidence | `TAVILY_API_KEY` env var |
| SEC EDGAR XBRL API | Deterministic fact lookup for public companies | None (free, public) |

**Rate limiting:** Groq's free tier allows ~12,000 tokens/minute. The `groqClient.js` enforces a 22-second minimum gap between calls. A full 8-claim report takes ~3 minutes.

---

## File map

```
src/
  server.js          Express API — routes and request/response handling
  claimExtractor.js  LLM prompt for extracting claims from report text
  validator.js       Validation strategies (EDGAR, search, LLM reasoning)
  xbrlLookup.js      SEC EDGAR XBRL API client (CIK lookup, fact retrieval)
  evidenceSearch.js  Tavily search client + search query builder per claim type
  scorer.js          Weighted scoring, cascade penalty, credibility rating
  redFlags.js        Rule-based red flag checks and analyst bias detector
  groqClient.js      Groq SDK wrapper with rate-limit enforcement
  pdfParser.js       PDF text extraction and preparation for large documents

tests/
  claimExtractor.test.js   Checks extraction output shape (mocked LLM)
  xbrlLookup.test.js       Checks EDGAR lookup against live API (Apple FY2024)
  pipeline.test.js         End-to-end API test (all external calls mocked)

examples/
  sampleReport.txt   Apple FY2024 research report — use this to try the API

docs/
  architecture.md    This file
```

---

## Data flow diagram

```
POST /api/analyze
  { report: "..." }
        │
        ├─ extractClaims(text)
        │       └─ chat(prompt) → Groq → JSON array of claims
        │
        └─ for each claim:
                ├─ DIRECT_FACT + ticker?
                │       └─ lookupDirectFact(claim)
                │               ├─ getCIK(ticker) → SEC /company_tickers.json
                │               ├─ getCompanyFacts(cik) → SEC /xbrl/companyfacts/
                │               └─ compareValues(stated, actual) → status/confidence
                │
                ├─ ACCOUNTING_POLICY?
                │       ├─ getLatest10K(ticker) → SEC /submissions/
                │       ├─ tavilySearch(query, ["sec.gov"])
                │       └─ chat(prompt + evidence) → Groq → validation JSON
                │
                └─ all others
                        ├─ buildSearchQuery(claim) → domain-specific query
                        ├─ tavilySearch(query, domains)
                        └─ chat(prompt + evidence) → Groq → validation JSON
        │
        └─ scoreReport(claims, validations)
                ├─ weighted average of status × confidence
                ├─ findCascades() → penalty for contradicted root claims
                ├─ checkRedFlags() → rule-based penalties
                └─ detectBias() → BULLISH / BEARISH / BALANCED
```
