# datasets/sec_edgar/scripts/augment_ciks.py
import argparse, pathlib

CURATED = [
  # Mega/large caps & high reporters (zero-padded CIKs)
  "0000320193","0000789019","0001652044","0001318605","0001067983","0000200406",
  "0000021344","0000093410","0001018724","0001326801","0001045810","0000034088",
  "0000104169","0000019617","0000070858","0000831001","0000050863","0000051143",
  "0000066740","0000078003","0000080424","0000354950","0000012927","0000006201",
  "0000277135","0000318154","0000215457","0000073309","0000200406","0000096021",
  "0000732717","0000200406","00000059478"  # add/adjust freely; script dedupes
]

def zpad(x):
    s = "".join([c for c in str(x) if c.isdigit()])
    return s.zfill(10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file",  default="datasets/sec_edgar/ciks.txt")
    ap.add_argument("--out_file", default="datasets/sec_edgar/ciks.txt")
    args = ap.parse_args()

    p = pathlib.Path(args.in_file)
    existing = []
    if p.exists():
        existing = [zpad(l.strip()) for l in p.read_text().splitlines() if l.strip()]
    merged = sorted(set(existing) | set(zpad(c) for c in CURATED if c))
    pathlib.Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_file).write_text("\n".join(merged) + "\n")
    print(f"[augment_ciks] wrote {args.out_file} total={len(merged)} (added={len(merged)-len(existing)})")

if __name__ == "__main__":
    main()
