# src/cli/compute_srs.py
"""
Compute SRS (Semantic Retention Score) for KG quality assessment.

Measures four deterministic metrics:
- HP: Hierarchy Presence (fraction with is-a parent)
- AtP: Attribute Predictability (fraction with unit edges)
- AP: Asymmetry Preservation (fraction directional)
- RTF: Relation Type Fidelity (embedding quality)

SRS = 0.25*HP + 0.20*AtP + 0.20*AP + 0.35*RTF
"""
import argparse, os, csv, json, yaml
from collections import defaultdict

def find_snapshot_folder(cfg_snapshot: str) -> str:
    """Accept either a full path or a snapshot name under data/kg/"""
    if not cfg_snapshot:
        raise ValueError("Config is missing data.kg_snapshot")
    # Try as given
    if os.path.isdir(cfg_snapshot):
        return cfg_snapshot
    # Try relative to cwd
    candidate = os.path.join("data", "kg", cfg_snapshot)
    if os.path.isdir(candidate):
        return candidate
    # Try with kg-mmml prefix (for canonical structure)
    alt_candidate = os.path.join("kg-mmml", "data", "kg", cfg_snapshot)
    if os.path.isdir(alt_candidate):
        return alt_candidate
    raise FileNotFoundError(f"KG snapshot folder not found: {cfg_snapshot} or {candidate} or {alt_candidate}")

def load_nodes_edges(folder: str):
    nodes_path = os.path.join(folder, "kg_nodes.csv")
    edges_path = os.path.join(folder, "kg_edges.csv")
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        raise FileNotFoundError(f"Missing kg files in {folder}")
    # nodes
    concepts, units, periods = set(), set(), set()
    with open(nodes_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = row["type"]
            nid = row["node_id"]
            if t == "Concept": concepts.add(nid)
            elif t == "Unit": units.add(nid)
            elif t == "Period": periods.add(nid)
    # edges
    edges_by_type = defaultdict(list)
    all_edges = []
    with open(edges_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            src, et, dst = row["src_id"], row["edge_type"], row["dst_id"]
            edges_by_type[et].append((src, dst))
            all_edges.append((src, et, dst))
    return concepts, units, periods, edges_by_type, all_edges

def metric_atp(concepts, edges_by_type):
    """Attribute Predictability (structural proxy): share of Concept nodes that have a measured-in Unit edge."""
    mi = edges_by_type.get("measured-in", [])
    with_unit = {src for (src, dst) in mi}
    denom = len(concepts)
    return (len(concepts & with_unit) / denom) if denom else 0.0

def metric_hp_coverage(concepts, edges_by_type):
    """Hierarchy Presence (coverage proxy): share of Concept nodes that have at least one parent via is-a."""
    isa = edges_by_type.get("is-a", [])
    children_with_parent = {src for (src, dst) in isa}
    denom = len(concepts)
    return (len(concepts & children_with_parent) / denom) if denom else 0.0

def metric_ap_directionality(edges_by_type):
    """
    Asymmetry Preservation:
    For directional types, score 1 - (fraction of edges that have an actual reverse edge of the SAME type).
    """
    directional = ["measured-in", "for-period"]
    total = 0
    bad = 0
    for et in directional:
        es = set(edges_by_type.get(et, []))  # set of (src, dst)
        for (s, d) in es:
            total += 1
            if (d, s) in es:   # reverse truly present in data
                bad += 1
    if total == 0:
        return 1.0
    return max(0.0, 1.0 - bad / total)

def weighted_srs(scores: dict, weights_cfg: dict):
    """
    Rescale weights to only the metrics that are not None.
    scores: {"RTF": None/float, "AP": float, "HP": float, "AtP": float}
    weights_cfg: {"RTF": 0.35, "AP": 0.2, "HP": 0.25, "AtP": 0.2}
    """
    # Keep only metrics with values
    avail = {k: v for k, v in scores.items() if v is not None}
    if not avail:
        return 0.0
    # Use cfg weights where present; default to equal if missing
    w = {k: float(weights_cfg.get(k, 1.0)) for k in avail.keys()}
    wsum = sum(w.values())
    if wsum == 0:
        # fallback to equal weights
        equal = 1.0 / len(avail)
        return sum(avail.values()) * equal
    # normalised weighted mean
    return sum(avail[k] * (w[k] / wsum) for k in avail.keys())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (uses data.kg_snapshot)")
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--rtf_score", required=False, help="Path to RTF score JSON file")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    # Try common places for weights
    srs_weights = {}
    try:
        srs_weights = cfg.get("eval", {}).get("srs", {}).get("weights", {})
    except Exception:
        srs_weights = {}
    kg_snapshot = cfg.get("data", {}).get("kg_snapshot", None)
    folder = find_snapshot_folder(kg_snapshot)

    concepts, units, periods, edges_by_type, all_edges = load_nodes_edges(folder)

    # Compute structural proxies now
    atp = metric_atp(concepts, edges_by_type)
    hp = metric_hp_coverage(concepts, edges_by_type)
    apdir = metric_ap_directionality(edges_by_type)
    rtf = None
    # If RTF score file is provided, read it
    if args.rtf_score:
        try:
            with open(args.rtf_score, "r", encoding="utf-8") as f:
                rtf_json = json.load(f)
                # Try common keys for RTF score
                if "rtf_f1_macro" in rtf_json:
                    rtf = float(rtf_json["rtf_f1_macro"])
                elif "rtf_accuracy" in rtf_json:
                    rtf = float(rtf_json["rtf_accuracy"])
                elif "F1 (macro)" in rtf_json:
                    rtf = float(rtf_json["F1 (macro)"])
                elif "RTF" in rtf_json:
                    rtf = float(rtf_json["RTF"])
                elif "scores" in rtf_json and "RTF" in rtf_json["scores"]:
                    rtf = float(rtf_json["scores"]["RTF"])
        except Exception as e:
            print(f"Warning: Could not read RTF score from {args.rtf_score}: {e}")
            rtf = None

    srs = weighted_srs({"RTF": rtf, "AP": apdir, "HP": hp, "AtP": atp}, srs_weights)

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Write single-row CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["RTF", "AP", "HP", "AtP", "SRS"])
        # If rtf is None, print as 'NA'
        rtf_str = "" if rtf is None else f"{rtf:.6f}"
        w.writerow([rtf_str, f"{apdir:.6f}", f"{hp:.6f}", f"{atp:.6f}", f"{srs:.6f}"])

    # Also drop a tiny JSON with counts for debugging (optional)
    debug = {
        "snapshot": folder,
        "counts": {
            "Concept": len(concepts),
            "Unit": len(units),
            "Period": len(periods),
            "edges_by_type": {k: len(v) for k, v in edges_by_type.items()},
        },
        "scores": {"RTF": rtf, "AP": apdir, "HP": hp, "AtP": atp, "SRS": srs},
        "weights_used": srs_weights,
    }
    dbg_path = os.path.splitext(args.out)[0] + "_debug.json"
    with open(dbg_path, "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2)
    print(f"[SRS] Wrote {args.out} and {dbg_path}")

if __name__ == "__main__":
    main()
