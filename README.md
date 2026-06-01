# Financial Report Validator

Paste in a financial research report (or upload a PDF up to ~200 pages) and get back a
credibility score, a breakdown of every factual claim, and citations to the actual SEC
filings that prove or disprove each one.

It extracts 6–8 key claims using an LLM, checks direct facts against SEC EDGAR's public
XBRL database, verifies everything else with targeted web search, and scores the report
on a 0–100 scale with red-flag rules baked in.

---

## Prerequisites

You need three things before you start:

1. **Node.js v18 or higher** — check with `node --version`. Download from https://nodejs.org if needed.
2. **A Groq API key** (free) — sign up at https://console.groq.com, then go to **API Keys** and create one.
3. **A Tavily API key** (free, 1 000 searches/month) — sign up at https://app.tavily.com, then go to **API Keys** and create one.

SEC EDGAR is queried directly (free, no key needed).

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/akash-verma-learns/finvalidunboundx.git
cd finvalidunboundx

# 2. Install all dependencies
npm install

# 3. Set up your environment variables
cp .env.example .env
```

Open the `.env` file you just created and fill in your two keys:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Save the file. Do not commit it — `.env` is already in `.gitignore`.

---

## Running the server

```bash
npm start
```

You should see this output:

```
Financial Report Validator v3
──────────────────────────────────────
LLM    : llama-3.3-70b-versatile (Groq — free)
Search : Tavily (free 1000/mo)
Facts  : SEC EDGAR XBRL API (free, no key)
PDF    : pdf-parse (up to ~200 pages)
...
http://localhost:8000
```

The server is now running at `http://localhost:8000`.

---

## Sending a report

### Option A — Plain text (via curl)

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "report": "Apple Inc. reported total net sales of $391.035 billion for fiscal year 2024, representing a 2% year-over-year increase. The company achieved a gross margin of 46.2%. Apple maintained a WACC of 8.5% in its DCF model, with a terminal growth rate of 3.0%, yielding a price target of $210."
  }'
```

A ready-made example report is in `examples/sampleReport.txt`:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d "{\"report\": $(cat examples/sampleReport.txt | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}"
```

### Option B — PDF upload

```bash
curl -X POST http://localhost:8000/api/analyze-pdf \
  -F "file=@/path/to/your/report.pdf"
```

The PDF can be up to ~200 pages (30 MB). The server extracts the text automatically.

---

## What the input looks like

**Plain text:**
```json
{
  "report": "Any financial research report as a plain string. Can be a few sentences or several pages."
}
```

**PDF:** multipart form upload with field name `file`.

---

## What the output looks like

```json
{
  "claims": [
    {
      "id": "CLM-001",
      "raw_text": "Apple Inc. reported total net sales of $391.035 billion for fiscal year 2024.",
      "type": "DIRECT_FACT",
      "company": "Apple Inc.",
      "ticker": "AAPL",
      "metric": "total revenue",
      "value": "$391.035 billion",
      "period": "FY2024",
      "checkable": true
    }
  ],
  "validations": {
    "CLM-001": {
      "status": "VERIFIED",
      "confidence": 0.99,
      "actual_value": "$391.035B",
      "discrepancy": "",
      "filing_source": "SEC EDGAR — Apple Inc. Form 10-K (filed 2024-11-01), XBRL: Revenues",
      "evidence": "SEC EDGAR XBRL database confirms Revenues = $391.035B for FY2024, filed 2024-11-01. Exact match (within 0.5% rounding).",
      "citations": [
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/",
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K"
      ],
      "reasoning": "Step 1: Resolved ticker AAPL → CIK 0000320193 via SEC EDGAR\nStep 2: Retrieved XBRL company facts (Form 10-K)\nStep 3: Located concept \"Revenues\" = $391.035B (filed 2024-11-01)\nStep 4: Exact match (within 0.5% rounding)",
      "flag": ""
    }
  },
  "scoring": {
    "overall_score": 84,
    "credibility_rating": "HIGH",
    "breakdown": {
      "DIRECT_FACT": 99,
      "MODEL_ASSUMPTION": 60
    },
    "cascades": [],
    "red_flags": [],
    "analyst_bias": "BALANCED",
    "verified_count": 3,
    "contradicted_count": 0,
    "claim_count": 6
  }
}
```

**`credibility_rating`** maps to score ranges:
- `HIGH` — 80–100
- `MODERATE` — 60–79
- `LOW` — 40–59
- `VERY LOW` — 0–39

---

## Running the tests

```bash
npm test
```

Three test suites run:
- `claimExtractor.test.js` — checks extraction output shape (LLM is mocked, runs instantly)
- `xbrlLookup.test.js` — looks up Apple FY2024 revenue against the live SEC EDGAR API
- `pipeline.test.js` — end-to-end API test with all external calls mocked

---

## Other endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analyze` | Validate a plain-text report |
| `POST` | `/api/analyze-pdf` | Validate a PDF report (field: `file`) |
| `POST` | `/api/extract` | Extract claims only (no validation) |
| `POST` | `/api/validate` | Validate a single pre-extracted claim |
| `GET`  | `/api/health` | Check the server is up |

---

## Rate limits and timing

Groq's free tier allows ~12 000 tokens per minute. The server automatically inserts a
22-second pause between LLM calls. A full 8-claim report takes roughly **3 minutes**.
Tavily's free tier gives you 1 000 searches per month.

---

## Project structure

```
src/
  server.js           API server — routes and request handling
  claimExtractor.js   Extracts claims from report text using the LLM
  validator.js        Validation strategies (EDGAR, search, LLM)
  xbrlLookup.js       SEC EDGAR XBRL API client
  evidenceSearch.js   Tavily web search client
  scorer.js           Weighted scoring and cascade penalties
  redFlags.js         Rule-based red flag checks
  groqClient.js       Groq SDK wrapper with rate limiting
  pdfParser.js        PDF text extraction for large documents

tests/               Automated tests (run with npm test)
examples/            Sample report to try immediately
docs/
  architecture.md    How the pipeline works in detail
```

See `docs/architecture.md` for a full explanation of how each step works.
