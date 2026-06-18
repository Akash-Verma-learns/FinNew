from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

# Tavily's free tier allows ~20 RPS; cap at 10 concurrent to stay safe and
# avoid burst errors that would silently degrade claim results to UNVERIFIABLE.
_TAVILY_SEM = asyncio.Semaphore(10)

TAVILY_URL = "https://api.tavily.com/search"

DOMAIN_MAP: dict[str, list[str]] = {
    # businesswire/prnewswire carry earnings press releases with exact segment figures
    # (Azure, Intelligent Cloud, etc.) that sec.gov XBRL doesn't expose separately.
    "DIRECT_FACT": ["sec.gov", "businesswire.com", "prnewswire.com", "finance.yahoo.com"],
    "DERIVED_METRIC": ["businesswire.com", "prnewswire.com", "finance.yahoo.com", "macrotrends.net"],
    "ACCOUNTING_POLICY": ["sec.gov"],
    "MODEL_ASSUMPTION": ["finance.yahoo.com", "wsj.com", "reuters.com", "marketwatch.com"],
    "FORWARD_PROJECTION": ["businesswire.com", "prnewswire.com", "finance.yahoo.com", "wsj.com"],
    "RECOMMENDATION": ["finance.yahoo.com", "wsj.com", "reuters.com", "marketwatch.com"],
    # finance.yahoo.com surfaces product metrics from earnings calls/press releases
    "QUALITATIVE": ["businesswire.com", "prnewswire.com", "finance.yahoo.com", "reuters.com"],
}

# Domains for non-US companies — avoid sec.gov, use broader financial data sources
DOMAIN_MAP_INTL: dict[str, list[str]] = {
    "DIRECT_FACT": ["finance.yahoo.com", "stockanalysis.com", "macrotrends.net", "reuters.com"],
    "DERIVED_METRIC": ["finance.yahoo.com", "stockanalysis.com", "macrotrends.net", "finbox.com"],
    "ACCOUNTING_POLICY": ["reuters.com", "wsj.com", "businesswire.com"],
    "MODEL_ASSUMPTION": ["finance.yahoo.com", "wsj.com", "reuters.com"],
    "FORWARD_PROJECTION": ["finance.yahoo.com", "wsj.com", "reuters.com"],
    "RECOMMENDATION": ["finance.yahoo.com", "wsj.com", "reuters.com"],
    "QUALITATIVE": ["reuters.com", "wsj.com", "marketwatch.com", "businesswire.com"],
}

# Per-exchange-prefix domain overrides for better data quality on specific markets
_EXCHANGE_DOMAINS: dict[str, dict[str, list[str]]] = {
    "WSE": {
        "DIRECT_FACT": ["stockanalysis.com", "finbox.com", "biznesradar.pl", "stooq.pl"],
        "DERIVED_METRIC": ["stockanalysis.com", "finbox.com", "macrotrends.net"],
        "FORWARD_PROJECTION": ["finance.yahoo.com", "reuters.com", "businesswire.com"],
        "RECOMMENDATION": ["finance.yahoo.com", "reuters.com", "businesswire.com"],
    },
    "GPW": {
        "DIRECT_FACT": ["stockanalysis.com", "finbox.com", "biznesradar.pl", "stooq.pl"],
        "DERIVED_METRIC": ["stockanalysis.com", "finbox.com", "macrotrends.net"],
    },
    "LSE": {
        "DIRECT_FACT": ["stockanalysis.com", "macrotrends.net", "finance.yahoo.com"],
        "DERIVED_METRIC": ["macrotrends.net", "stockanalysis.com", "finance.yahoo.com"],
    },
    "TSX": {
        "DIRECT_FACT": ["sedarplus.ca", "stockanalysis.com", "macrotrends.net", "finance.yahoo.com"],
        "DERIVED_METRIC": ["sedarplus.ca", "stockanalysis.com", "macrotrends.net", "finance.yahoo.com"],
        "ACCOUNTING_POLICY": ["sedarplus.ca", "reuters.com"],
        "FORWARD_PROJECTION": ["finance.yahoo.com", "reuters.com", "marketwatch.com"],
        "RECOMMENDATION": ["finance.yahoo.com", "reuters.com", "marketwatch.com"],
        "QUALITATIVE": ["reuters.com", "marketwatch.com", "businesswire.com"],
        "MODEL_ASSUMPTION": ["finance.yahoo.com", "macrotrends.net", "reuters.com"],
    },
}

QUERY_TEMPLATES: dict[str, str] = {
    # earnings press releases use "quarterly results" and report exact segment figures
    "DIRECT_FACT": "{company} {ticker} {metric} {period} quarterly results earnings",
    "DERIVED_METRIC": "{company} {ticker} {metric} {period} quarterly results earnings",
    "ACCOUNTING_POLICY": "{company} {ticker} accounting policy {metric} 10-K SEC",
    "MODEL_ASSUMPTION": "{company} {ticker} {metric} consensus analyst estimate {period}",
    "FORWARD_PROJECTION": "{company} {ticker} {metric} announcement {period}",
    "RECOMMENDATION": "{company} {ticker} analyst rating price target",
    "QUALITATIVE": "{company} {metric} {period}",
}

# Versions without the "10-K SEC" framing — used for non-US companies
QUERY_TEMPLATES_INTL: dict[str, str] = {
    "DIRECT_FACT": "{company} {ticker} {metric} {period} annual report financial results",
    "DERIVED_METRIC": "{company} {ticker} {metric} {period} annual",
    "ACCOUNTING_POLICY": "{company} {ticker} accounting policy {metric} annual report",
    "MODEL_ASSUMPTION": "{company} {ticker} {metric} consensus analyst estimate {period}",
    "FORWARD_PROJECTION": "{company} {ticker} {metric} guidance forecast {period}",
    "RECOMMENDATION": "{company} {ticker} analyst rating price target",
    "QUALITATIVE": "{company} {metric}",
}

# Non-US exchange prefixes for query routing
_NON_US_PREFIXES = frozenset({
    "WSE", "GPW", "LSE", "LON", "TSX", "TSE", "ASX", "HKG", "HKEX",
    "TYO", "TKS", "SHE", "SHG", "NSE", "BSE", "KRX", "EURONEXT",
    "EPA", "AMS", "STO", "OSL", "CPH", "FRA", "ETR", "BIT", "JSE",
})


# Well-known non-US tickers that LLMs frequently extract without exchange prefix.
# Prevents sec.gov domain routing for companies that don't file with the SEC.
_KNOWN_NON_US: dict[str, str] = {
    # TSX (Canada)
    "CJT": "TSX", "RY": "TSX", "TD": "TSX", "BNS": "TSX", "BMO": "TSX",
    "CNR": "TSX", "CP": "TSX", "SU": "TSX", "ENB": "TSX", "TRP": "TSX",
    "BCE": "TSX", "T": "TSX", "MFC": "TSX", "SLF": "TSX", "POW": "TSX",
    # LSE (UK)
    "HSBA": "LSE", "BP": "LSE", "SHEL": "LSE", "AZN": "LSE", "GSK": "LSE",
    "ULVR": "LSE", "RIO": "LSE", "AAL": "LSE", "BT": "LSE", "VOD": "LSE",
    # ASX (Australia)
    "BHP": "ASX", "CBA": "ASX", "WBC": "ASX", "ANZ": "ASX", "NAB": "ASX",
    "CSL": "ASX", "WES": "ASX", "WOW": "ASX", "MQG": "ASX", "FMG": "ASX",
}


def _is_non_us(ticker: str | None) -> bool:
    if not ticker:
        return False
    t = ticker.strip().upper()
    if ":" in t:
        return t.split(":")[0] in _NON_US_PREFIXES
    # Fallback: known non-US tickers extracted without exchange prefix
    return t in _KNOWN_NON_US


def _resolve_ticker(ticker: str) -> str:
    """Return ticker with exchange prefix, inferring it from _KNOWN_NON_US if missing."""
    t = ticker.strip().upper()
    if ":" not in t and t in _KNOWN_NON_US:
        return f"{_KNOWN_NON_US[t]}:{t}"
    return ticker


def _exchange_ticker_suffix(ticker: str) -> str:
    """Convert exchange-prefixed ticker to the local market suffix format used by Yahoo Finance etc.
    TSX:CJT → CJT.TO, LSE:VOD → VOD.L, ASX:BHP → BHP.AX, others → bare ticker."""
    _SUFFIXES = {"TSX": ".TO", "LSE": ".L", "LON": ".L", "ASX": ".AX", "TSE": ".T", "TYO": ".T"}
    if ":" in ticker:
        prefix, sym = ticker.upper().split(":", 1)
        return sym + _SUFFIXES.get(prefix, "")
    return ticker


def build_search_query(claim) -> str:
    resolved = _resolve_ticker(claim.ticker) if claim.ticker else ""
    intl = _is_non_us(claim.ticker)
    templates = QUERY_TEMPLATES_INTL if intl else QUERY_TEMPLATES
    template = templates.get(claim.type, "{company} {metric}")
    # For non-US tickers, use the local market suffix (CJT.TO, VOD.L) so
    # Yahoo Finance / stockanalysis results match the right company.
    ticker_fmt = _exchange_ticker_suffix(resolved) if intl and resolved else (claim.ticker or "")
    query = template.format(
        company=claim.company or "",
        ticker=ticker_fmt,
        metric=claim.metric or "",
        period=claim.period or "",
    ).strip()
    if query:
        return query
    # QUALITATIVE claims have no literal text in their template ("{company} {metric}"),
    # so a claim with neither field set (common for narrative statements without a
    # named company/metric) would otherwise produce an empty string — Tavily rejects
    # empty queries with a 400. Fall back to the claim's own text.
    return " ".join((claim.raw_text or "").split())[:150]


def get_domains(claim) -> list[str]:
    resolved = _resolve_ticker(claim.ticker) if claim.ticker else (claim.ticker or "")
    if ":" in resolved:
        prefix = resolved.strip().upper().split(":")[0]
        exchange_overrides = _EXCHANGE_DOMAINS.get(prefix, {})
        if claim.type in exchange_overrides:
            return exchange_overrides[claim.type]
        if prefix in _NON_US_PREFIXES:
            return DOMAIN_MAP_INTL.get(claim.type, [])
    return DOMAIN_MAP.get(claim.type, [])


async def tavily_search(query: str, domains: list[str] | None = None) -> dict:
    payload: dict = {
        "api_key": os.getenv("TAVILY_API_KEY", ""),
        "query": query,
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": True,
    }
    if domains:
        payload["include_domains"] = domains

    logger.info("  [tavily] POST %s query=%r domains=%s", TAVILY_URL, query, domains)
    async with _TAVILY_SEM:
        async with httpx.AsyncClient() as client:
            resp = await client.post(TAVILY_URL, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()

    results = data.get("results", [])
    citations = [r["url"] for r in results if r.get("url")]
    context_parts: list[str] = []
    if data.get("answer"):
        context_parts.append(data["answer"])
    for r in results[:5]:
        if r.get("content"):
            context_parts.append(r["content"][:1000])

    logger.info("  [tavily] returned %d results: %s", len(results), citations)
    return {"context": "\n\n".join(context_parts), "citations": citations}
