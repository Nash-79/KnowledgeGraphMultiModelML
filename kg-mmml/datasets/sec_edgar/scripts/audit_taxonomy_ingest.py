# scripts/audit_taxonomy_ingest.py
import json, pandas as pd, pathlib, sys
facts = "data/processed/sec_edgar/facts.jsonl"
tax   = "datasets/sec_edgar/taxonomy/usgaap_combined.csv"

# 1) observed full concepts in facts
full=set(); short=set()
with open(facts,"r",encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        r=json.loads(ln)
        ns=(r.get("ns") or "").strip()
        c =(r.get("concept") or "").strip()
        if not c: continue
        if ns and not c.startswith(ns+":"): full.add(f"{ns}:{c}"); short.add(c)
        elif ":" in c: full.add(c); short.add(c.split(":",1)[1])
        else: full.add(f"us-gaap:{c}"); short.add(c)

df = pd.read_csv(tax)
assert {"child","parent"}.issubset({c.lower() for c in df.columns}), "taxonomy must have child,parent"
df.columns=[c.lower() for c in df.columns]
df['child']=df['child'].str.strip(); df['parent']=df['parent'].str.strip()

# 2) coverage
tot = len(df)
kept = df[df['child'].isin(full)]
dropped = df[~df['child'].isin(full)]
print(f"taxonomy edges: {tot} | kept(children in facts): {len(kept)} | dropped: {len(dropped)}")
print("top parents (kept):")
print(kept['parent'].value_counts().head(10))
print("examples dropped (first 10):")
print(dropped.head(10).to_string(index=False))
