"""Parse XBRL instance docs into concept facts (stub)."""
import argparse, os, json
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infile', required=True)
    ap.add_argument('--outdir', default='parsed')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, 'facts.jsonl')
    with open(out, 'w') as f:
        f.write(json.dumps({'concept':'us-gaap:Assets','value':0,'unit':'USD','period':'2024'})+'\n')
    print('Wrote', out)
if __name__ == '__main__':
    main()

