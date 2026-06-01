import requests, time, sys

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "test_report.pdf"
print(f"Testing with: {pdf_path}")

start = time.time()
with open(pdf_path, "rb") as f:
    fname = pdf_path.split("\\")[-1].split("/")[-1]
    r = requests.post(
        "http://localhost:8000/api/analyze-pdf",
        files={"file": (fname, f, "application/pdf")},
        timeout=600,
    )
elapsed = round(time.time() - start, 1)

if r.status_code != 200:
    print(f"ERROR {r.status_code}: {r.text[:500]}")
    raise SystemExit(1)

d = r.json()
print(f"\n{'='*60}")
print(f"Time:       {elapsed}s")
print(f"Indexer:    {d.get('indexer')}")
secs = d.get("sections_found", [])
print(f"Sections:   {len(secs)}")
for s in secs[:10]:
    print(f"  • {s}")
if len(secs) > 10:
    print(f"  ... and {len(secs)-10} more")
print(f"Claims:     {d.get('claim_count')}  (checkable: {sum(1 for c in d.get('claims',[]) if c.get('checkable'))})")
print(f"Score:      {d.get('overall_score')} / 100  ({d.get('credibility_rating')})")
print(f"Verified:   {d.get('verified_count')}")
print(f"Contradict: {d.get('contradicted_count')}")
flags = d.get("red_flags", [])
if flags:
    print(f"Red flags:  {len(flags)}")
    for f in flags:
        print(f"  [{f['severity']}] {f['message']}")
print(f"\n--- Claims & Validation ---")
for c in d.get("claims", []):
    v = d.get("validations", {}).get(c["id"], {})
    ticker = f"({c.get('ticker')})" if c.get("ticker") else ""
    period = c.get("period") or ""
    print(f"  [{c['type'][:12]}] {c.get('metric','?')} {ticker} = {c.get('value','?')} {period}")
    print(f"           → {v.get('status')} conf={v.get('confidence',0):.2f}  {v.get('actual_value') or ''}")
