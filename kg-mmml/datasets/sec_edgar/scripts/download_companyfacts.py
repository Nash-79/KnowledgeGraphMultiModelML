# datasets/sec_edgar/scripts/download_companyfacts.py
import argparse, time, pathlib, requests

URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

def zpad(c): return "".join(ch for ch in str(c) if ch.isdigit()).zfill(10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ciks_file", required=True)
    ap.add_argument("--out", default="data/processed/sec_edgar/companyfacts")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    ua = os.environ.get("SEC_USER_AGENT", "y2r03@students.keele.ac.uk CSC40098MScProject/1.0")
    headers = {"User-Agent": ua}
    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    ciks = [zpad(l.strip()) for l in pathlib.Path(args.ciks_file).read_text().splitlines() if l.strip()]
    for i, cik in enumerate(ciks, 1):
        url = URL_TMPL.format(cik=cik)
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                (outdir / f"{cik}.json").write_bytes(r.content)
                print(f"[{i}/{len(ciks)}] {cik} ok")
            else:
                print(f"[{i}/{len(ciks)}] {cik} HTTP {r.status_code}")
        except Exception as e:
            print(f"[{i}/{len(ciks)}] {cik} error: {e}")
        time.sleep(args.sleep)

if __name__ == "__main__":
    import os
    main()
