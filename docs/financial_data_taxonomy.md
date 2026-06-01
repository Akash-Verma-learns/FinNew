# Financial Data Taxonomy
### Structured and Unstructured Data Across SEC Filings, Earnings Calls, and Analyst Reports

---

## 1. Machine-Readable (XBRL-Tagged)

These fields are directly extractable from SEC EDGAR's XBRL APIs with no parsing required. Every 10-K and 10-Q filed since ~2009 must include XBRL tagging for these items.

### Income Statement
| Field | XBRL Concept(s) | Notes |
|---|---|---|
| Revenue | `RevenueFromContractWithCustomerExcludingAssessedTax`, `Revenues`, `SalesRevenueNet` | Concept varies by company |
| Cost of Revenue | `CostOfRevenue`, `CostOfGoodsSoldAndServicesSold` | |
| Gross Profit | `GrossProfit` | |
| R&D Expense | `ResearchAndDevelopmentExpense` | |
| SG&A Expense | `SellingGeneralAndAdministrativeExpense` | |
| Operating Income | `OperatingIncomeLoss` | |
| Interest Expense | `InterestExpense` | |
| Income Tax Expense | `IncomeTaxExpensesBenefit` | |
| Net Income | `NetIncomeLoss` | |
| EPS (Basic) | `EarningsPerShareBasic` | |
| EPS (Diluted) | `EarningsPerShareDiluted` | |
| D&A | `DepreciationDepletionAndAmortization` | Often in cash flow or notes |

### Balance Sheet
| Field | XBRL Concept(s) | Notes |
|---|---|---|
| Cash & Equivalents | `CashAndCashEquivalentsAtCarryingValue` | |
| Short-Term Investments | `ShortTermInvestments` | |
| Accounts Receivable | `AccountsReceivableNetCurrent` | |
| Inventory | `InventoryNet` | |
| Total Current Assets | `AssetsCurrent` | |
| PP&E (net) | `PropertyPlantAndEquipmentNet` | |
| Goodwill | `Goodwill` | |
| Intangible Assets | `FiniteLivedIntangibleAssetsNet` | |
| Total Assets | `Assets` | |
| Accounts Payable | `AccountsPayableCurrent` | |
| Short-Term Debt | `ShortTermBorrowings`, `LongTermDebtCurrent` | |
| Total Current Liabilities | `LiabilitiesCurrent` | |
| Long-Term Debt | `LongTermDebt` | |
| Total Liabilities | `Liabilities` | |
| Stockholders' Equity | `StockholdersEquity` | |

### Cash Flow Statement
| Field | XBRL Concept(s) | Notes |
|---|---|---|
| Operating Cash Flow | `NetCashProvidedByUsedInOperatingActivities` | |
| CapEx | `PaymentsToAcquirePropertyPlantAndEquipment` | |
| Acquisitions | `PaymentsToAcquireBusinessesNetOfCashAcquired` | |
| Share Repurchases | `PaymentsForRepurchaseOfCommonStock` | |
| Dividends Paid | `PaymentsOfDividends` | |
| Net Change in Cash | `CashAndCashEquivalentsPeriodIncreaseDecrease` | |

### Share Data
| Field | XBRL Concept(s) | Notes |
|---|---|---|
| Shares Outstanding | `CommonStockSharesOutstanding` | |
| Weighted Avg Shares (Basic) | `WeightedAverageNumberOfSharesOutstandingBasic` | |
| Weighted Avg Shares (Diluted) | `WeightedAverageNumberOfDilutedSharesOutstanding` | |

---

## 2. Semi-Structured (Parseable but Requires Extraction Logic)

These are present in filings but not fully XBRL-tagged. They require table parsing, regex, or NLP to extract reliably.

### Segment Breakdowns (in 10-K/10-Q MD&A and Notes)
- **By geography**: Americas, Europe, Asia-Pacific, etc. — revenue and sometimes margin per region
- **By product/service line**: e.g., Apple's iPhone, Mac, iPad, Wearables, Services — each with revenue
- **By business unit**: e.g., Google's Search, YouTube, Cloud, Other Bets
- Often appear as tables in Note disclosures (ASC 280) — parseable with PDF table extraction

### Non-GAAP Reconciliation Tables
- Adjusted EBITDA, Adjusted EPS, Adjusted Operating Income
- Companies reconcile from GAAP to non-GAAP in earnings releases (8-K) and sometimes in 10-K
- Format varies significantly by company — no standard structure
- **Critical**: these are the numbers analysts and media quote; often differ materially from GAAP

### Guidance Figures
- Revenue guidance range (e.g., "$X billion to $Y billion")
- EPS guidance, gross margin guidance
- Usually disclosed in earnings call (transcript) or earnings release (8-K exhibit)
- No standard XBRL tag — must be extracted from text or tables

### Executive Compensation Tables (DEF 14A / Proxy Statement)
- Base salary, annual bonus, stock awards, option awards, total compensation
- XBRL-tagged in DEF 14A since 2018 but format is inconsistent
- Named executive officers (NEOs) only — CEO, CFO, and top 3 earners

### Remaining Performance Obligations / Backlog
- Contracted revenue not yet recognized (ASC 606)
- Tagged as `RevenueRemainingPerformanceObligation` in XBRL, but timing breakdown is in notes only

### Debt Maturity Schedules
- Annual principal payments due in each of next 5 years + thereafter
- In notes to financial statements — table format, parseable

### Goodwill by Segment
- Disclosed in notes (ASC 350) — goodwill allocated to each reportable segment
- Relevant for tracking acquisition history and impairment risk by business area

---

## 3. Raw Text (Unstructured — Requires NLP)

These are narrative disclosures with no structured format. They contain some of the most analytically important information but require NLP to extract insights.

### Risk Factors (Item 1A of 10-K)
- Ordered list of material risks as assessed by management
- **Order matters**: risks that changed position may signal shifting management priorities
- New risk factors that didn't appear in the prior filing signal something new and material
- Removed risk factors may mean the risk was resolved — or de-emphasized for strategic reasons
- Examples: regulatory risks, cybersecurity risks, supply chain concentration, key customer dependency

### MD&A — Management Commentary (Item 7 of 10-K)
- Explains what drove revenue, margin, and cash flow changes
- Discusses competitive dynamics, market share, pricing environment
- Contains qualitative forward-looking language that precedes quantitative guidance
- Comparison between consecutive filings reveals shifts in tone and focus areas

### Business Description (Item 1 of 10-K)
- Products, services, markets, customers, suppliers, competition
- Changes between filings can reveal strategic pivots or new competitive threats
- Rarely changes dramatically but diffs are meaningful when they do

### Forward-Looking Statements / Outlook Language
- Hedged language like "we expect," "we anticipate," "subject to"
- Found in MD&A, earnings calls, and 8-K press releases
- Key source for qualitative guidance when quantitative guidance is not provided

### Earnings Call Transcripts (not SEC filings, but highly valuable)
- **Prepared remarks**: scripted, management-controlled narrative
- **Q&A session**: unscripted — analysts probe for specific metrics, management tone under pressure
- Tone analysis (confidence, hesitation) and topic frequency tracking are high-signal
- Often the first place non-GAAP metrics and guidance appear

### Analyst Reports
- Investment thesis and price target rationale
- Qualitative assessments of competitive positioning, management quality, industry dynamics
- Model assumptions (WACC, terminal growth, revenue growth rates)
- Not filed with SEC — sourced from brokerages, Bloomberg, FactSet

### Legal Proceedings (Item 3 of 10-K)
- Material litigation and regulatory investigations
- New proceedings are a flag; settled proceedings may no longer appear
- Often vague — details come from legal filings and press releases

---

## 4. What Changes Between Filings

This is where the analytical signal is densest. Static snapshots matter less than deltas.

| Change Type | Where to Look | What It Signals |
|---|---|---|
| New risk factor | Item 1A diff | Emerging threat management is now legally required to disclose |
| Removed risk factor | Item 1A diff | Risk resolved — or company stopped worrying about it |
| Risk factor moved up in ordering | Item 1A diff | Management considers it higher priority |
| New segment created | Notes (ASC 280) | Business has grown enough to report separately; or strategic refocus |
| Segment eliminated/merged | Notes (ASC 280) | Business being wound down or combined for optics |
| Accounting policy change | Note 1 (Summary of Significant Accounting Policies) | ASC adoption (e.g., ASC 842 for leases), restatement risk |
| Auditor change | Form 8-K within 4 days of change | Disagreement with auditor; independence issues |
| Revenue recognition method change | Note 1 | Can materially shift reported timing of revenue |
| MD&A tone shift | Text diff between quarters | Management becoming more/less confident about a product line or market |
| New KPI introduced | MD&A tables | Company wants investors to track a metric where they are doing well |
| KPI removed or buried | MD&A comparison | Metric is deteriorating and management is de-emphasizing it |
| Guidance format changed | 8-K / earnings call | Company becoming less specific (risk increasing) or more specific (confidence increasing) |
| Non-GAAP definition changed | Earnings release | Goalposts moved; prior period comparisons break |

---

## Filing Types Quick Reference

| Filing | Frequency | Key Content |
|---|---|---|
| 10-K | Annual | Full audited financials, full risk factors, MD&A, all notes |
| 10-Q | Quarterly | Unaudited financials, condensed notes, updated MD&A |
| 8-K | Event-driven | Earnings releases, material events, auditor changes, guidance updates |
| DEF 14A (Proxy) | Annual | Executive compensation, board composition, shareholder votes |
| S-1 / S-3 | Offering | Registration statements — first look at financials for new issuers |
| Form 4 | Per transaction | Insider buying/selling — executives and directors |
| SC 13D/G | Ownership changes | Large shareholders (>5%) acquiring or reducing positions |
