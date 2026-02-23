import argparse, json, pathlib, re, time, requests

UA = "NareshMepani-MScProject/1.0 (your.email@example.com)"
ARCH_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/"

def strip0(cik): return str(int(cik))  # remove leading zeros
def nodash(accn): return accn.replace("-", "")

def fetch_index(url, sleep=0.2):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status(); time.sleep(sleep)
    return r.text

def download(url, out):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", default="data/raw/sec_edgar/selected.json")
    ap.add_argument("--outdir", default="data/raw/sec_edgar/xbrl")
    args = ap.parse_args()

    sel = json.loads(pathlib.Path(args.selected).read_text())
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for cik, forms in sel.items():
        for form, items in forms.items():
            for it in items:
                accn = it["accession"]
                base = ARCH_BASE.format(cik=strip0(cik), accn=nodash(accn))
                idx = fetch_index(base)
                # naive pick: any .xml that looks like instance; refine as you learn
                candidates = re.findall(r'href="([^"]+\.xml)"', idx, re.I)
                if not candidates: 
                    print("No XML for", cik, accn); continue
                # pick first xml file (or prefer those with 'ins'/'cal' heuristics)
                xfile = [c for c in candidates if "ins" in c.lower()] or candidates
                xfile = xfile[0]
                if xfile.startswith("/Archives/"):
                    url = "https://www.sec.gov" + xfile
                else:
                    url = base.rstrip("/") + "/" + xfile.lstrip("/")
                out = outdir / f"{cik}_{nodash(accn)}_{xfile.split('/')[-1]}"
                print("Downloading", url, "->", out)
                download(url, out)

if __name__ == "__main__":
    main()
