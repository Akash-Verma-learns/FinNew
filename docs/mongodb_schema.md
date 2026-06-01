# MongoDB Schema — FinValidator / Zenith Data Plane

All collections live on the shared Zenith MongoDB instance. CIK is the canonical company identifier throughout — tickers are resolved to CIK at ingestion time and stored in a lookup collection. Services never join on ticker strings internally.

---

## Collection: `ticker_lookup`

Maps every known identifier for a company to its CIK.

```json
{
  "_id": "AAPL",
  "cik": "0000320193",
  "canonical_ticker": "AAPL",
  "aliases": ["AAPL", "NASDAQ:AAPL", "$AAPL"],
  "company_name": "Apple Inc.",
  "exchange": "NASDAQ",
  "filing_type": "10-K"
}
```

Index: `{ "_id": 1 }`, `{ "cik": 1 }`, `{ "aliases": 1 }`

**Rule:** All other collections key on `cik`, never on ticker. Resolve ticker → CIK at the pipeline boundary using this collection.

---

## Collection: `xbrl_facts`

One document per (company, period, XBRL concept). Raw facts as filed — immutable once written, safe to cache indefinitely.

```json
{
  "_id": "0000320193__2024-09-28__Revenues",
  "cik": "0000320193",
  "ticker": "AAPL",
  "period_end": "2024-09-28",
  "period_type": "annual",
  "xbrl_concept": "RevenueFromContractWithCustomerExcludingAssessedTax",
  "metric_alias": "revenue",
  "value": 391035000000,
  "unit": "USD",
  "form": "10-K",
  "accession_number": "0000320193-24-000123",
  "filing_date": "2024-11-01",
  "edgar_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/",
  "ingested_at": "2026-05-23T00:00:00Z"
}
```

Index: `{ "cik": 1, "period_end": -1, "metric_alias": 1 }` (compound, covers most queries)

**Cache policy:** Immutable — never invalidate. New filings add new documents; existing ones are never overwritten.

---

## Collection: `derivation_log`

One document per computed metric per (company, period, formula source). Written by the calculation engine after resolving from `xbrl_facts` via `formula_graph.py`.

```json
{
  "_id": "0000320193__2024-09-28__GrossMarginPct__fasb_linkbase",
  "cik": "0000320193",
  "period_end": "2024-09-28",
  "concept": "GrossMarginPct",
  "formula_source": "fasb_linkbase",
  "value": 46.21,
  "inputs_used": {
    "GrossProfit": 180683000000,
    "Revenues": 391035000000
  },
  "all_definitions": [
    { "source": "company_10k", "value": 46.21 },
    { "source": "fasb_linkbase", "value": 46.21 },
    { "source": "textbook", "value": 46.21 }
  ],
  "delta": 0.0,
  "computed_at": "2026-05-23T00:00:00Z"
}
```

Index: `{ "cik": 1, "period_end": -1, "concept": 1 }`

**Cache policy:** Invalidate all derivations for a `cik` whenever a new filing is ingested for that ticker. Key: `(cik, period_end, formula_source)`. On new 10-K or 10-Q: `db.derivation_log.deleteMany({ "cik": cik })` then recompute.

---

## Collection: `formula_definitions`

Stores the parsed FASB calculation linkbase. Loaded into memory at service startup — no runtime queries.

```json
{
  "_id": "GrossMarginPct__fasb_linkbase",
  "concept": "GrossMarginPct",
  "source": "fasb_linkbase",
  "inputs": ["GrossProfit", "Revenues"],
  "formula_description": "GrossProfit / Revenues * 100",
  "taxonomy_version": "2024",
  "updated_at": "2024-01-15T00:00:00Z"
}
```

**Cache policy:** Permanent in memory. Refresh once per year when FASB updates the taxonomy.

---

## Collection: `text_chunks`

Chunked and embedded text from 10-K sections, 10-Q MD&A, and earnings call transcripts. Supports Atlas Vector Search for RAG queries.

```json
{
  "_id": "0000320193__10K__2024__risk_factors__chunk_042",
  "cik": "0000320193",
  "ticker": "AAPL",
  "form": "10-K",
  "period_end": "2024-09-28",
  "filing_date": "2024-11-01",
  "accession_number": "0000320193-24-000123",
  "edgar_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/",
  "section": "risk_factors",
  "chunk_index": 42,
  "text": "Our business could be harmed by events outside our control including...",
  "embedding": [0.021, -0.043, 0.118, "...1536 floats..."],
  "embedding_model": "text-embedding-3-small",
  "char_count": 412,
  "ingested_at": "2026-05-23T00:00:00Z"
}
```

Atlas Vector Search index on `embedding` field (cosine similarity, dimensions: 1536).
Standard index: `{ "cik": 1, "form": 1, "section": 1, "period_end": -1 }`

**Cache policy:** Immutable per (accession_number, chunk_index). New filings add new chunks; old ones are retained for historical diff queries (e.g. "what new risks appeared since last quarter?").

---

## Collection: `filing_index`

One document per SEC filing. Source of truth for what has been ingested.

```json
{
  "_id": "0000320193-24-000123",
  "cik": "0000320193",
  "ticker": "AAPL",
  "form": "10-K",
  "period_end": "2024-09-28",
  "filing_date": "2024-11-01",
  "accession_number": "0000320193-24-000123",
  "edgar_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/",
  "xbrl_ingested": true,
  "text_ingested": true,
  "derived_computed": false,
  "ingested_at": "2026-05-23T00:00:00Z"
}
```

Index: `{ "cik": 1, "form": 1, "period_end": -1 }`

Used by the ingestion pipeline to avoid re-processing and to trigger cache invalidation on new filings.

---

## Query Patterns

**"What was Apple's gross margin in FY2024?"**
```js
// Try derivation_log first (pre-computed)
db.derivation_log.findOne({ cik: "0000320193", period_end: "2024-09-28", concept: "GrossMarginPct" })
// If missing, fetch GrossProfit + Revenues from xbrl_facts and compute
```

**"What risks did Apple disclose about China in FY2024?"**
```js
// Atlas Vector Search on text_chunks
// { cik: "0000320193", section: "risk_factors", period_end: "2024-09-28" }
// vector: embed("Apple China supply chain risk")
```

**"What new risks appeared since last quarter?"**
```js
// Pull text_chunks for two consecutive periods, diff by section + chunk similarity
```

**"Resolve ticker BRK.A to CIK"**
```js
db.ticker_lookup.findOne({ aliases: "BRK-A" })  // normalized before query
```
