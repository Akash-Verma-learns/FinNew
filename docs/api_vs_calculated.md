# API vs. Calculated: What We Can Fetch vs. What We Must Build

This document maps every metric we care about to either a direct API source or a calculation we need to implement ourselves. The goal is a clear "we can fetch this" vs. "we have to build this" split before we scale the pipeline.

---

## Part 1: SEC EDGAR APIs (Free, No Key Required)

All endpoints are under `https://data.sec.gov`. Rate limit: 10 requests/second. Required header: `User-Agent: <app-name> <contact-email>`.

### 1.1 Company Tickers
```
GET https://data.sec.gov/files/company_tickers.json
```
**Returns**: Full map of ticker → CIK (company identifier) for all SEC registrants.
**Use for**: Resolving any ticker to a CIK before other API calls.
**Limitation**: No fuzzy matching — must be exact ticker symbol.

---

### 1.2 Company Facts (XBRL) — Our Primary Source
```
GET https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
```
**Returns**: All XBRL-tagged facts ever reported by the company, organized by concept and period.
**Covers**: Every income statement, balance sheet, and cash flow line item tagged in 10-K and 10-Q filings since ~2009.

**What we can fetch directly from this endpoint**:
- Revenue (multiple concept names depending on company)
- Gross Profit, Operating Income, Net Income
- EPS (basic and diluted)
- Cash & Equivalents, Short-Term Investments
- Total Assets, Total Liabilities, Stockholders' Equity
- Long-Term Debt, Short-Term Debt
- Operating Cash Flow, CapEx
- Shares Outstanding (basic and diluted)
- R&D Expense, SG&A Expense
- D&A (from cash flow section)
- Interest Expense, Tax Expense
- Goodwill, Intangible Assets
- Accounts Receivable, Accounts Payable, Inventory

**Limitation**: Data is annual (10-K) or quarterly (10-Q). No intra-quarter data. XBRL concept names vary by company — requires mapping logic.

---

### 1.3 Submissions (Filing History)
```
GET https://data.sec.gov/submissions/CIK{cik}.json
```
**Returns**: All filings ever made by the company — form type, date, accession number.
**Use for**: 
- Finding the most recent 10-K URL
- Getting filing dates for period alignment
- Detecting auditor changes (auditor listed in 10-K header)
- Building a timeline of when data was filed vs. reported

---

### 1.4 XBRL Frames API — Cross-Company Comparison
```
GET https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/{unit}/{period}.json
```
Example:
```
GET https://data.sec.gov/api/xbrl/frames/us-gaap/Revenues/USD/CY2023Q4I.json
```
**Returns**: The value of a specific XBRL concept for ALL companies that reported it in a given period.
**Use for**:
- Industry benchmarking — "what is the median gross margin across all S&P 500 companies?"
- Peer comparison tables
- Detecting outliers in a metric across a sector

**Period format**: `CY{year}` for annual, `CY{year}Q{n}I` for quarterly instant, `CY{year}Q{n}` for quarterly duration.

---

### 1.5 Full-Text Search API (EFTS)
```
GET https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt={date}&enddt={date}&forms={form-type}
```
Example — find all 10-Ks mentioning "supply chain concentration":
```
GET https://efts.sec.gov/LATEST/search-index?q=%22supply+chain+concentration%22&forms=10-K&dateRange=custom&startdt=2023-01-01&enddt=2024-01-01
```
**Returns**: Filing documents matching the full-text query with links to the actual filing.
**Use for**:
- Finding filings where a specific risk factor or topic is discussed
- Tracking when a company first mentioned a topic (e.g., "generative AI", "tariff risk")
- Verifying accounting policy disclosures
- Detecting new language in risk factors across a portfolio of companies

**Limitation**: Returns document links, not extracted text — we still need to fetch and parse the actual filing.

---

### 1.6 Direct Filing Documents
```
https://www.sec.gov/Archives/edgar/data/{cik}/{accession-number-no-dashes}/
```
**Use for**: Downloading the actual 10-K HTML/text to extract semi-structured and unstructured content (risk factors, MD&A, segment tables).

---

## Part 2: External APIs

### 2.1 Free / Low-Cost

| Source | What It Provides | Limitations |
|---|---|---|
| **FRED (Federal Reserve)** | Macro data: GDP, CPI, interest rates, unemployment, treasury yields | No company-level data |
| **Yahoo Finance (unofficial)** | Stock prices, earnings estimates, analyst ratings, historical data | Unofficial — fragile, no SLA, can break |
| **Alpha Vantage** | Prices, basic financials, some earnings data | 25 API calls/day on free tier; limited history |
| **Quandl (Nasdaq Data Link)** | Some financial datasets free | Most useful datasets are paid |

### 2.2 Paid (Worth Knowing About)

| Source | What It Provides | Notes |
|---|---|---|
| **Polygon.io** | Real-time + historical prices, options, fundamentals | Reasonable pricing; good for market data |
| **Intrinio** | Fundamentals, XBRL data, news sentiment | Good alternative to EDGAR for structured financials |
| **FactSet / Bloomberg / Refinitiv** | Comprehensive — prices, estimates, transcripts, ownership | Enterprise pricing; overkill for now |
| **Calcbench** | Parsed XBRL with entity normalization | Addresses the concept-name-variation problem |

---

## Part 3: Metrics We Must Calculate

These are not available as raw data from any API — they require computation from fetched raw values.

### 3.1 Profitability Margins
```
Gross Margin        = Gross Profit / Revenue × 100
Operating Margin    = Operating Income / Revenue × 100
Net Margin          = Net Income / Revenue × 100
EBITDA Margin       = EBITDA / Revenue × 100
```

### 3.2 EBITDA (not directly in XBRL for most companies)
```
EBITDA = Operating Income + Depreciation & Amortization
```
D&A is available in cash flow XBRL (`DepreciationDepletionAndAmortization`), so this is computable.
Note: Adjusted EBITDA (non-GAAP) requires company-specific add-backs — cannot be standardized.

### 3.3 Growth Rates
```
YoY Revenue Growth  = (Revenue_t / Revenue_{t-1} - 1) × 100
CAGR (n years)      = (Revenue_t / Revenue_{t-n})^(1/n) - 1
```
Both periods must be fetched first; growth is then derived.

### 3.4 Liquidity Ratios
```
Current Ratio   = Current Assets / Current Liabilities
Quick Ratio     = (Cash + Short-Term Investments + Receivables) / Current Liabilities
Cash Ratio      = Cash / Current Liabilities
Working Capital = Current Assets - Current Liabilities
```

### 3.5 Leverage / Debt Ratios
```
Debt/Equity Ratio   = Total Debt / Stockholders' Equity
Debt/EBITDA         = Total Debt / EBITDA
Net Debt            = Total Debt - Cash & Equivalents
Net Debt/EBITDA     = Net Debt / EBITDA
Interest Coverage   = Operating Income (EBIT) / Interest Expense
```

### 3.6 Return Metrics
```
ROE     = Net Income / Average Stockholders' Equity × 100
ROA     = Net Income / Average Total Assets × 100
ROIC    = NOPAT / Invested Capital × 100
         where NOPAT = Operating Income × (1 - Tax Rate)
         and Invested Capital = Total Assets - Non-Interest-Bearing Current Liabilities
```
Average values require fetching two periods (beginning + end of year).

### 3.7 Free Cash Flow
```
FCF = Operating Cash Flow - CapEx
```
Both components are available from XBRL — FCF itself is not.

### 3.8 Enterprise Value and Multiples
```
Enterprise Value (EV) = Market Cap + Total Debt - Cash & Equivalents
Market Cap            = Stock Price × Shares Outstanding

P/E   = Stock Price / Diluted EPS
P/S   = Market Cap / Revenue
P/B   = Market Cap / Book Value of Equity
EV/EBITDA = Enterprise Value / EBITDA
EV/Revenue = Enterprise Value / Revenue
```
**Stock price is NOT in SEC EDGAR.** Must come from a market data API (Yahoo Finance, Polygon.io, etc.). Without price, we cannot compute EV or any price-based multiple.

### 3.9 Segment Metrics
```
Segment Operating Margin = Segment Operating Income / Segment Revenue × 100
Segment Revenue Mix      = Segment Revenue / Total Revenue × 100
Segment Revenue Growth   = (Segment Revenue_t / Segment Revenue_{t-1} - 1) × 100
```
Segment revenue and operating income must be parsed from the 10-K notes (ASC 280 disclosures) — not XBRL-tagged.

### 3.10 Non-GAAP Metrics
Non-GAAP metrics (Adjusted EPS, Adjusted EBITDA, Free Cash Flow as defined by the company) are **company-specific** and require:
1. Parsing the reconciliation table from the earnings release (8-K exhibit)
2. Identifying each add-back/exclusion item
3. Applying the adjustment to the GAAP base

These cannot be standardized across companies — each needs its own reconciliation parser.

---

## Summary Table

| Metric | Source | Notes |
|---|---|---|
| Revenue, Gross Profit, Net Income, EPS | SEC EDGAR XBRL | Fetch directly |
| Cash, Debt, Assets, Equity | SEC EDGAR XBRL | Fetch directly |
| Operating Cash Flow, CapEx | SEC EDGAR XBRL | Fetch directly |
| Shares Outstanding | SEC EDGAR XBRL | Fetch directly |
| Filing dates, 10-K URL | SEC EDGAR Submissions | Fetch directly |
| Cross-company benchmarks | SEC EDGAR XBRL Frames | Fetch directly |
| Risk factors, MD&A narrative | SEC EDGAR Full-Text Search + filing download | Fetch + parse |
| Stock Price | Yahoo Finance / Polygon.io | External API |
| Analyst Estimates / Ratings | Yahoo Finance / paid APIs | External API |
| Macro data (rates, GDP) | FRED | External API |
| Gross / Operating / Net Margin | Derived | Profit / Revenue |
| EBITDA | Derived | Op. Income + D&A |
| Revenue Growth (YoY, CAGR) | Derived | Two periods required |
| Current / Quick Ratio | Derived | Balance sheet items |
| Debt/EBITDA, Interest Coverage | Derived | Requires EBITDA calculation |
| ROE, ROA, ROIC | Derived | Requires average balance sheet values |
| Free Cash Flow | Derived | Op. CF - CapEx |
| Enterprise Value | Derived | Requires stock price |
| P/E, EV/EBITDA, P/S, P/B | Derived | Requires stock price + derived EBITDA |
| Segment margins, mix | Derived from parsed tables | Not in XBRL |
| Non-GAAP Adjusted metrics | Derived from reconciliation tables | Company-specific |
