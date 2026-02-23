"""
Generate auto-taxonomy edges from regex pattern rules.

Applies pattern-based rules to infer parent-child relationships from observed concepts.
Conservative: first matching rule per concept wins.
"""
import argparse, json, pathlib, re
import pandas as pd

def load_patterns(yaml_path):
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parents = cfg.get("parents", {}) or {}
    compiled = []
    for parent, pats in parents.items():
        for pat in (pats or []):
            compiled.append((parent.strip(), re.compile(pat)))
    return compiled

def iter_concepts(facts_path):
    seen = set()
    with open(facts_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            ns = (rec.get("ns") or "").strip()
            c  = (rec.get("concept") or "").strip()
            if not c: continue
            child = f"{ns}:{c}" if ns and not c.startswith(ns + ":") else c
            if child and child not in seen:
                seen.add(child)
                yield child

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="data/processed/sec_edgar/facts.jsonl")
    ap.add_argument("--rules", default="datasets/sec_edgar/taxonomy/pattern_rules.yaml")
    ap.add_argument("--out",   default="datasets/sec_edgar/taxonomy/usgaap_auto.csv")
    args = ap.parse_args()

    patterns = load_patterns(args.rules)
    rows = []
    matched = 0

    for child in iter_concepts(args.facts):
        for parent, rgx in patterns:
            if rgx.match(child):
                rows.append({"child": child, "parent": parent, "source": "auto"})
                matched += 1
                break  # first hit wins (conservative)

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates().to_csv(outp, index=False)
    print(f"[auto-taxonomy] wrote {outp} rows={len(rows)} matched={matched}")

if __name__ == "__main__":
    main()
