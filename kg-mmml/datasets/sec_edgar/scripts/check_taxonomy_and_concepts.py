# scripts/check_taxonomy_and_concepts.py
import json, pandas as pd, pathlib, sys

facts = "data/processed/sec_edgar/facts.jsonl"
tax   = "datasets/sec_edgar/taxonomy/usgaap_combined.csv"

# load concepts observed in facts
concepts=set()
with open(facts,"r",encoding="utf-8") as f:
    for ln in f:
        if not ln.strip(): continue
        r=json.loads(ln)
        ns=(r.get("ns") or "").strip()
        c =(r.get("concept") or "").strip()
        if not c: continue
        cc = f"{ns}:{c}" if ns and not c.startswith(ns+":") else c
        concepts.add(cc)

df = pd.read_csv(tax)
cols = [c.lower() for c in df.columns]
print("CSV columns:", df.columns.tolist())

def coverage(child_col, parent_col):
    missing_children = df[~df[child_col].isin(concepts)][child_col].nunique()
    used_children    = df[df[child_col].isin(concepts)][child_col].nunique()
    return missing_children, used_children

if set(cols) >= {"child","parent"}:
    mc, uc = coverage("child","parent")
    print(f"[child,parent] children_in_kg={uc} missing_children={mc}, edges={len(df)}")
else:
    print("Warning: CSV does not have columns named exactly 'child' and 'parent'.")

# Heuristic: if swapping improves coverage
if set(cols) >= {"child","parent"}:
    pass
else:
    # try to guess if saved as parent,child
    if "parent" in cols and "child" not in cols and "child" in df.columns[::-1].tolist():
        print("CSV columns look strange; please ensure exact column names 'child','parent'.")
