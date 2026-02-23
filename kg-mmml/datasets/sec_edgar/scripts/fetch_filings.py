# datasets/sec_edgar/scripts/fetch_filings.py
import argparse, os, json, time, pathlib, requests, sys
from dotenv import load_dotenv

UA = "Your Name Contact@domain.com"  # Replace with your real name and email per SEC requirements
load_dotenv()
UA = os.getenv("SEC_USER_AGENT", "Your Name Contact@domain.com")  # fallback if not set
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

def zfill_cik(x: str) -> str:
    x = "".join(ch for ch in x.strip() if ch.isdigit())
    return x.zfill(10)

def fetch_json(url, sleep=0.2):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    time.sleep(sleep)  # be nice to the API
    return r.json()

def load_ciks_from_file(path):
    ciks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ciks.append(zfill_cik(s))
    # de-dup while preserving order
    return list(dict.fromkeys(ciks))

def fetch_ticker_map(outdir):
    data = fetch_json(TICKERS_URL)
    if not data:
        return {}
    # Input format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    mapping = {}
    for _, row in data.items():
        cik = str(row["cik_str"]).zfill(10)
        mapping[row["ticker"].upper()] = {"cik": cik, "title": row["title"]}
    out = pathlib.Path(outdir) / "company_tickers.json"
    out.write_text(json.dumps(mapping, indent=2))
    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ciks", nargs="*", help="List of CIKs (any format; will be zero-padded to 10)")
    ap.add_argument("--ciks_file", help="File with one CIK per line")
    ap.add_argument("--tickers_file", help="Optional: file with one TICKER per line to resolve to CIKs")
    ap.add_argument("--out", default="data/raw/sec_edgar")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build candidate CIK list
    ciks = []
    if args.ciks_file:
        ciks.extend(load_ciks_from_file(args.ciks_file))
    if args.ciks:
        ciks.extend(zfill_cik(x) for x in args.ciks)

    # Optional: resolve tickers → CIKs
    if args.tickers_file:
        mapping = fetch_ticker_map(outdir)
        with open(args.tickers_file, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip().upper()
                if t and t in mapping:
                    ciks.append(mapping[t]["cik"])

    # de-dup preserving order
    ciks = list(dict.fromkeys(ciks))

    if not ciks:
        print("No CIKs provided. Use --ciks, --ciks_file, or --tickers_file.", file=sys.stderr)
        sys.exit(1)

    index = {"ok": {}, "missing": []}
    for cik in ciks:
        url = SUBMISSIONS.format(cik=cik)
        data = fetch_json(url, sleep=args.sleep)
        if data is None:  # 404
            index["missing"].append(cik)
            continue
        path = outdir / f"submissions_{cik}.json"
        path.write_text(json.dumps(data, indent=2))
        index["ok"][cik] = {"submissions_json": str(path)}

    (outdir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Saved index with {len(index['ok'])} OK and {len(index['missing'])} missing → {outdir/'index.json'}")

if __name__ == "__main__":
    main()
