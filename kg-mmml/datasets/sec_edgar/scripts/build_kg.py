# datasets/sec_edgar/scripts/build_kg.py
import argparse, json, pathlib, csv
from typing import List, Tuple

def normalise_concept_id(ns: str, concept: str) -> str:
    """
    Ensure concept IDs align with taxonomy (e.g., 'us-gaap:Assets').
    If facts provide ns='us-gaap' and concept='Assets', produce 'us-gaap:Assets'.
    If concept already has a prefix, keep it as-is.
    """
    ns = (ns or "").strip()
    cname = (concept or "").strip()
    if not cname:
        return "UNKNOWN"
    if ":" in cname:
        # Already namespaced; normalise spacing
        prefix, name = cname.split(":", 1)
        return f"{prefix}:{name}"
    if ns:
        return f"{ns}:{cname}"
    # Default to us-gaap if none given (facts usually provide ns)
    return f"us-gaap:{cname}"

def _detect_columns_and_iter(reader: csv.DictReader):
    """
    Detect orientation and yield (child, parent) pairs, normalised.
    Accepts CSVs with header either:
      - child,parent,[...]
      - parent,child,[...]
      - or two unnamed columns in that order.
    """
    # Lowercase mapping for convenience
    cols = {k.lower(): k for k in reader.fieldnames or []}

    def norm_ns(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        if ":" in s:
            pref, name = s.split(":", 1)
            return f"{pref}:{name}"
        return f"us-gaap:{s}"

    # Case 1: has both child & parent headers
    if "child" in cols and "parent" in cols:
        for row in reader:
            child = norm_ns(row.get(cols["child"], ""))
            parent = norm_ns(row.get(cols["parent"], ""))
            if child and parent and child != parent:
                yield (child, parent)
        return

    # Case 2: has 'parent' header but no 'child' header (assume parent,child)
    if "parent" in cols and "child" not in cols:
        # Find a second column name (first two columns)
        fns = reader.fieldnames or []
        if len(fns) >= 2:
            parent_key = cols["parent"]
            # pick the first non-parent as child
            child_key = next((c for c in fns if c != parent_key), fns[0])
            for row in reader:
                parent = norm_ns(row.get(parent_key, ""))
                child = norm_ns(row.get(child_key, ""))
                if child and parent and child != parent:
                    yield (child, parent)
            return

    # Case 3: no headers / unknown layout — treat first two columns as (child,parent) if possible
    # Re-open as plain CSV without DictReader to read raw columns
    raise ValueError("Unable to detect taxonomy columns; ensure CSV has child,parent or parent,child headers.")

def load_taxonomy(csv_path: str) -> List[Tuple[str, str]]:
    """Load taxonomy edges and normalise to (child, parent) tuples with namespaces."""
    p = pathlib.Path(csv_path)
    if not p.exists():
        return []
    pairs = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header; add 'child,parent' or 'parent,child'")
        for child, parent in _detect_columns_and_iter(reader):
            pairs.append((child, parent))
    # de-dup and drop self-loops (already filtered)
    pairs = sorted(set(pairs))
    return pairs

def write_csv(nodes, edges, outdir: pathlib.Path):
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "kg_nodes.csv").open("w", newline="", encoding="utf-8") as nf:
        nw = csv.writer(nf)
        nw.writerow(["node_id", "type", "attrs_json"])
        for n in nodes:
            nw.writerow([n["id"], n["type"], json.dumps(n.get("attrs", {}))])
    with (outdir / "kg_edges.csv").open("w", newline="", encoding="utf-8") as ef:
        ew = csv.writer(ef)
        ew.writerow(["src_id", "edge_type", "dst_id", "attrs_json"])
        for e in edges:
            ew.writerow([e["src"], e["type"], e["dst"], json.dumps(e.get("attrs", {}))])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", default="data/raw/sec_edgar/selected.json",
                    help="JSON produced by select_filings.py (CIK -> {10-K/10-Q:[{accession,doc}]})")
    ap.add_argument("--facts", default="data/processed/sec_edgar/facts.jsonl",
                    help="Normalised facts JSONL (ns, concept, unit, value, period_end, accn)")
    ap.add_argument("--taxonomy", default="datasets/sec_edgar/taxonomy/usgaap_combined.csv",
                    help="CSV of concept hierarchy (child,parent or parent,child supported)")
    ap.add_argument("--snapshot", default="data/kg/sec_edgar_YYYY-MM-DD",
                    help="Output folder for kg_nodes.csv and kg_edges.csv")
    args = ap.parse_args()

    sel_path = pathlib.Path(args.selected)
    facts_path = pathlib.Path(args.facts)
    snap_dir = pathlib.Path(args.snapshot)

    if not sel_path.exists():
        raise FileNotFoundError(f"Missing --selected file: {sel_path}")
    selected = json.loads(sel_path.read_text())

    # ------------------------
    # Collect facts
    # ------------------------
    facts = []
    if facts_path.exists():
        with facts_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        facts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # ------------------------
    # Load taxonomy edges (child,parent), normalised
    # ------------------------
    taxonomy_pairs = load_taxonomy(args.taxonomy)  # list of (child, parent)

    # ------------------------
    # Build graph
    # ------------------------
    nodes, edges = [], []
    seen_nodes = set()            # (id,type)
    seen_edges = set()            # (src,type,dst)

    def add_node(nid: str, ntype: str, attrs=None):
        key = (nid, ntype)
        if key in seen_nodes:
            return
        nodes.append({"id": nid, "type": ntype, "attrs": attrs or {}})
        seen_nodes.add(key)

    def add_edge(src: str, etype: str, dst: str, attrs=None):
        key = (src, etype, dst)
        if key in seen_edges or src == dst:
            return
        edges.append({"src": src, "type": etype, "dst": dst, "attrs": attrs or {}})
        seen_edges.add(key)

    # Companies & filings
    for cik, forms in selected.items():
        cid = f"cik_{cik}"
        add_node(cid, "Company", {"cik": cik})
        for form, items in (forms or {}).items():
            for it in items:
                accn = (it.get("accession") or "").replace("-", "")
                fid = f"filing_{cik}_{accn}" if accn else f"filing_{cik}_UNKNOWN"
                add_node(fid, "Filing", {"form": form, "accession": it.get("accession", "")})
                add_edge(cid, "reports", fid, {})

    # Facts → Concept, Unit, Period
    observed_concepts = set()
    for f in facts:
        ns = (f.get("ns") or "").strip()
        cname = normalise_concept_id(ns, f.get("concept", ""))
        cpt = f"concept_{cname}"
        unit = (f.get("unit") or "").strip()
        period_end = (f.get("period_end") or "").strip()

        unt = f"unit_{unit}" if unit else "unit_UNKNOWN"
        per = f"period_{period_end}" if period_end else "period_UNKNOWN"

        add_node(cpt, "Concept", {"ns": cname.split(":",1)[0] if ":" in cname else ""})
        observed_concepts.add(cname)
        add_node(unt, "Unit", {"symbol": unit})
        add_node(per, "Period", {"end": period_end})

        add_edge(cpt, "measured-in", unt, {})
        add_edge(cpt, "for-period", per, {})

    # Ensure taxonomy parents also become Concept nodes (schema concepts)
    taxonomy_children = {c for (c, p) in taxonomy_pairs}
    taxonomy_parents  = {p for (c, p) in taxonomy_pairs}
    taxonomy_concepts = taxonomy_children | taxonomy_parents

    for cname in taxonomy_concepts:
        add_node(f"concept_{cname}", "Concept", {"ns": cname.split(":",1)[0] if ":" in cname else ""})

    # Taxonomy is-a edges (Concept → Concept): child -> parent
    for child, parent in taxonomy_pairs:
        add_edge(f"concept_{child}", "is-a", f"concept_{parent}", {})

    write_csv(nodes, edges, snap_dir)
    print(f"Snapshot: {snap_dir} | nodes: {len(nodes)} | edges: {len(edges)} | taxonomy_pairs: {len(taxonomy_pairs)}")

if __name__ == "__main__":
    main()
