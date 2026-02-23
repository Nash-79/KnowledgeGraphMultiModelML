import argparse
import json
import pathlib
import datetime

FORMS_K = {"10-K"}
FORMS_Q = {"10-Q"}
FORMS_K_AMEND = {"10-K/A"}
FORMS_Q_AMEND = {"10-Q/A"}

def parse_filing_date(s: str) -> datetime.date:
    """
    SEC 'filingDate' often appears as 'YYYYMMDD' or 'YYYY-MM-DD'.
    """
    s = (s or "").strip()
    if not s:
        # Fallback to very old date so it will be filtered out.
        return datetime.date(1900, 1, 1)
    s2 = s.replace("-", "")
    try:
        return datetime.datetime.strptime(s2, "%Y%m%d").date()
    except Exception:
        try:
            return datetime.date.fromisoformat(s[:10])
        except Exception:
            return datetime.date(1900, 1, 1)

def within_years(d: datetime.date, years: int) -> bool:
    cutoff = datetime.date.today() - datetime.timedelta(days=365 * years)
    return d >= cutoff

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/raw/sec_edgar/index.json",
                    help="JSON mapping CIK -> {'submissions_json': path}")
    ap.add_argument("--out", default="data/raw/sec_edgar/selected.json")
    ap.add_argument("--years", type=int, default=4,
                    help="Look-back window (years) for 10-K/10-Q selection")
    ap.add_argument("--include_amends", action="store_true",
                    help="Include 10-K/A and 10-Q/A")
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional per-form cap after year filtering (0 = no cap)")
    args = ap.parse_args()

    idx_path = pathlib.Path(args.index)
    if not idx_path.exists():
        raise SystemExit(f"Index file not found: {idx_path}")

    idx = json.loads(idx_path.read_text())
    ok = idx.get("ok", {})
    selected = {}
    total_cik = 0
    kept_k = kept_q = 0

    for cik, meta in ok.items():
        sub_path = pathlib.Path(meta.get("submissions_json", ""))
        if not sub_path.exists():
            print(f"Warning: missing submissions JSON for CIK {cik}: {sub_path}")
            continue

        j = json.loads(sub_path.read_text())
        recent = j.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accns = recent.get("accessionNumber", [])
        primdocs = recent.get("primaryDocument", [])
        dates = recent.get("filingDate", [])

        # Collect candidates with parsed dates
        rows = []
        for f, a, d, p in zip(forms, accns, dates, primdocs):
            fd = parse_filing_date(d)
            rows.append({
                "form": f,
                "accession": a,
                "filingDate": fd,
                "doc": p,
            })

        # Allowed form sets
        allow_k = set(FORMS_K)
        allow_q = set(FORMS_Q)
        if args.include_amends:
            allow_k |= FORMS_K_AMEND
            allow_q |= FORMS_Q_AMEND

        # Filter by form + years, then sort newest-first
        k_rows = [r for r in rows if r["form"] in allow_k and within_years(r["filingDate"], args.years)]
        q_rows = [r for r in rows if r["form"] in allow_q and within_years(r["filingDate"], args.years)]
        k_rows.sort(key=lambda r: r["filingDate"], reverse=True)
        q_rows.sort(key=lambda r: r["filingDate"], reverse=True)

        # Per-form cap after year-filter (optional)
        if args.limit and args.limit > 0:
            k_rows = k_rows[:args.limit]
            q_rows = q_rows[:args.limit]

        # Serialise back to strings (ISO)
        out_k = [{"accession": r["accession"], "doc": r["doc"], "filingDate": r["filingDate"].isoformat()} for r in k_rows]
        out_q = [{"accession": r["accession"], "doc": r["doc"], "filingDate": r["filingDate"].isoformat()} for r in q_rows]

        selected[cik] = {"10-K": out_k, "10-Q": out_q}
        total_cik += 1
        kept_k += len(out_k)
        kept_q += len(out_q)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selected, indent=2))
    print(f"Saved {out_path}  (CIKs={total_cik}, 10-K kept={kept_k}, 10-Q kept={kept_q}, years={args.years}, cap={args.limit}, amends={args.include_amends})")

if __name__ == "__main__":
    main()
