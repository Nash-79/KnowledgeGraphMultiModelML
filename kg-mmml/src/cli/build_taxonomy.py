"""
Unified taxonomy builder combining manual, pattern, and frequency rules.

Merges parent-child relations from:
1. Manual CSV (seed taxonomy)
2. Regex pattern rules
3. Frequency rules for common families

Optionally materializes transitive closure.
"""
# datasets/sec_edgar/scripts/build_taxonomy.py
import argparse, pathlib, re, yaml, json
import pandas as pd
from collections import defaultdict

def load_concepts_from_facts(facts_path, min_cik_support=1):
    """Extract observed concepts with CIK support counts."""
    short2ciks = defaultdict(set)
    full_set = set()
    with open(facts_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            ns = (r.get("ns") or "").strip()
            c = (r.get("concept") or "").strip()
            cik = str(r.get("cik") or "").strip()
            if not c: continue
            
            full = f"{ns}:{c}" if ns and not c.startswith(ns + ":") else (c if ":" in c else f"us-gaap:{c}")
            short = c if ":" not in c else c.split(":", 1)[1]
            
            full_set.add(full)
            if cik: short2ciks[short].add(cik)
    
    short_supported = {s: len(ciks) for s, ciks in short2ciks.items() if len(ciks) >= min_cik_support}
    return full_set, short_supported

def apply_pattern_rules(concepts_full, concepts_short, rules_yaml):
    """Apply pattern-based taxonomy rules."""
    y = yaml.safe_load(pathlib.Path(rules_yaml).read_text())
    parents_map = y.get("parents", {}) or {}
    
    edges = set()
    for parent, patterns in parents_map.items():
        parent_full = parent if ":" in parent else f"us-gaap:{parent}"
        for pattern_str in (patterns or []):
            rx = re.compile(pattern_str, re.IGNORECASE)
            # Match on short names
            for short in concepts_short.keys():
                if rx.search(short):
                    full = f"us-gaap:{short}"
                    if full in concepts_full:
                        edges.add((full, parent_full))
    return edges

def apply_frequency_rules(short_supported, min_support=3):
    """Frequency-based rules for common concepts."""
    FAMILIES = [
        (re.compile(r"^AccountsReceivable.*", re.I), "us-gaap:CurrentAssets"),
        (re.compile(r"^AccountsPayable.*", re.I), "us-gaap:CurrentLiabilities"),
        (re.compile(r"^Inventory.*", re.I), "us-gaap:CurrentAssets"),
        (re.compile(r"^PropertyPlantAndEquipment.*", re.I), "us-gaap:NoncurrentAssets"),
        (re.compile(r"^Goodwill$|^.*IntangibleAssets.*$", re.I), "us-gaap:NoncurrentAssets"),
        (re.compile(r"^OperatingLease.*Asset.*", re.I), "us-gaap:NoncurrentAssets"),
        (re.compile(r"^OperatingLease.*Liability.*", re.I), "us-gaap:NoncurrentLiabilities"),
        (re.compile(r"^DeferredRevenue.*|^ContractWithCustomerLiability.*", re.I), "us-gaap:CurrentLiabilities"),
        (re.compile(r"^ResearchAndDevelopmentExpense.*", re.I), "us-gaap:OperatingExpenses"),
        (re.compile(r"^SellingGeneralAndAdministrativeExpense.*", re.I), "us-gaap:OperatingExpenses"),
        (re.compile(r"^Revenue.*|^SalesRevenue.*", re.I), "us-gaap:Revenues"),
        (re.compile(r"^CostOfRevenue.*|^CostOfGoodsSold.*", re.I), "us-gaap:CostOfRevenue"),
    ]
    
    edges = set()
    for short, support_count in short_supported.items():
        if support_count < min_support: continue
        for rx, parent in FAMILIES:
            if rx.match(short):
                edges.add((f"us-gaap:{short}", parent))
                break
    return edges

def add_backbone():
    """Core structural relationships."""
    return [
        ("us-gaap:AssetsCurrent", "us-gaap:Assets"),
        ("us-gaap:AssetsNoncurrent", "us-gaap:Assets"),
        ("us-gaap:LiabilitiesCurrent", "us-gaap:Liabilities"),
        ("us-gaap:LiabilitiesNoncurrent", "us-gaap:Liabilities"),
        ("us-gaap:OperatingExpenses", "us-gaap:OperatingIncomeLoss"),
        ("us-gaap:CostOfRevenue", "us-gaap:CostsAndExpenses"),
        ("us-gaap:AdditionalPaidInCapital", "us-gaap:StockholdersEquity"),
        ("us-gaap:RetainedEarningsAccumulatedDeficit", "us-gaap:StockholdersEquity"),
        ("us-gaap:PropertyPlantAndEquipmentNet", "us-gaap:Assets"),
    ]

def transitive_closure(edges_df):
    """Materialize all ancestor paths."""
    parents = defaultdict(set)
    for c, p in edges_df.itertuples(index=False):
        parents[c].add(p)
    
    memo = {}
    def ancestors(c):
        if c in memo: return memo[c]
        A = set()
        for p in parents.get(c, ()):
            A.add(p)
            A |= ancestors(p)
        memo[c] = A
        return A
    
    rows = set()
    for c in set(edges_df["child"]):
        for a in ancestors(c):
            if c != a:
                rows.add((c, a))
    return pd.DataFrame(sorted(rows), columns=["child", "parent"])

def normalize_df(df):
    """Ensure consistent format and deduplication."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    def norm(s):
        s = str(s).strip()
        if not s or s == "nan": return None
        if ":" in s:
            ns, name = s.split(":", 1)
            return f"{ns}:{name}"
        return f"us-gaap:{s}"
    
    df["child"] = df["child"].apply(norm)
    df["parent"] = df["parent"].apply(norm)
    df = df.dropna()
    df = df[df["child"] != df["parent"]].drop_duplicates()
    return df

def main():
    ap = argparse.ArgumentParser(description="Unified taxonomy builder")
    ap.add_argument("--facts", required=True)
    ap.add_argument("--manual", default="datasets/sec_edgar/taxonomy/usgaap_min.csv")
    ap.add_argument("--rules", default="datasets/sec_edgar/taxonomy/pattern_rules.yaml")
    ap.add_argument("--out", default="datasets/sec_edgar/taxonomy/usgaap_combined.csv")
    ap.add_argument("--min_cik_support", type=int, default=3)
    ap.add_argument("--with_closure", action="store_true")
    args = ap.parse_args()
    
    # Load manual base
    manual_df = normalize_df(pd.read_csv(args.manual))
    
    # Extract concepts from facts
    concepts_full, concepts_short = load_concepts_from_facts(args.facts, args.min_cik_support)
    
    # Apply pattern rules
    pattern_edges = apply_pattern_rules(concepts_full, concepts_short, args.rules)
    pattern_df = pd.DataFrame(sorted(pattern_edges), columns=["child", "parent"])
    
    # Apply frequency rules
    freq_edges = apply_frequency_rules(concepts_short, args.min_cik_support)
    freq_df = pd.DataFrame(sorted(freq_edges), columns=["child", "parent"])
    
    # Add backbone
    backbone_df = pd.DataFrame(add_backbone(), columns=["child", "parent"])
    
    # Combine all sources
    combined = pd.concat([manual_df[["child", "parent"]], pattern_df, freq_df, backbone_df], ignore_index=True)
    combined = normalize_df(combined)
    
    # Optional transitive closure
    if args.with_closure:
        closed = transitive_closure(combined)
        combined = pd.concat([combined, closed], ignore_index=True).drop_duplicates()
    
    # Save
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(outp, index=False)
    
    print(f"[taxonomy] manual={len(manual_df)} pattern={len(pattern_df)} freq={len(freq_df)} -> total={len(combined)} -> {outp}")

if __name__ == "__main__":
    main()