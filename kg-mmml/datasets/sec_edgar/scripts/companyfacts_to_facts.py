#!/usr/bin/env python3
"""
companyfacts_to_facts.py

Normalise SEC CompanyFacts JSON -> facts.jsonl

Each line in the output is a single fact:
{
  "cik": "0000320193",
  "entity": "Apple Inc.",
  "ns": "us-gaap",
  "concept": "Assets",
  "unit": "USD",
  "value": 383000000000,
  "period_end": "2024-09-28",
  "accn": "0000320193-24-000010",
  "fy": 2024,
  "fp": "FY",
  "form": "10-K",
  "filed": "2024-11-03",
  "frame": "CY2024"
}

Usage (defaults to us-gaap only, numeric values only):
  python companyfacts_to_facts.py \
    --indir data/processed/sec_edgar/companyfacts \
    --out data/processed/sec_edgar/facts.jsonl \
    --include_ns us-gaap \
    --include_units USD USD/shares shares \
    --include_forms 10-K 10-Q \
    --min_fy 2022 --max_fy 2025 \
    --latest_per_key

All filters are optional; omit to keep everything.
"""

import argparse
import json
import os
import pathlib
from collections import Counter, defaultdict

def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def normalise_cik(cik):
    s = "".join(ch for ch in str(cik) if ch.isdigit())
    return s.zfill(10) if s else ""

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True,
                    help="Folder containing companyfacts_*.json files")
    ap.add_argument("--out", required=True,
                    help="Output JSONL path for normalised facts")
    ap.add_argument("--include_ns", nargs="*", default=["us-gaap"],
                    help="Only include these namespaces (default: us-gaap). Use empty to include all.")
    ap.add_argument("--include_units", nargs="*", default=[],
                    help="Only include these units (e.g. USD USD/shares shares). Empty keeps all.")
    ap.add_argument("--include_forms", nargs="*", default=[],
                    help="Only include facts from these SEC forms (e.g. 10-K 10-Q). Empty keeps all.")
    ap.add_argument("--min_fy", type=int, default=None,
                    help="Minimum fiscal year (inclusive).")
    ap.add_argument("--max_fy", type=int, default=None,
                    help="Maximum fiscal year (inclusive).")
    ap.add_argument("--numeric_only", action="store_true", default=True,
                    help="Keep only numeric values (default True). Use --no-numeric_only to keep all.")
    ap.add_argument("--no-numeric_only", dest="numeric_only", action="store_false")
    ap.add_argument("--latest_per_key", action="store_true",
                    help="Keep only latest filed per (cik,concept,unit,period_end).")
    return ap.parse_args()

def iter_companyfacts(indir):
    p = pathlib.Path(indir)
    # Support both legacy names (companyfacts_*.json) and plain CIK names (*.json).
    files = sorted(set(p.glob("companyfacts_*.json")) | set(p.glob("*.json")))
    for fp in files:
        try:
            doc = json.loads(fp.read_text())
        except Exception:
            continue
        yield fp.name, doc

def fact_records(doc, args):
    """Yield normalised records from a single CompanyFacts JSON doc"""
    facts = doc.get("facts", {}) or {}
    cik = normalise_cik(doc.get("cik") or doc.get("cik_str") or "")
    entity = doc.get("entityName", "").strip()

    for ns, concepts in facts.items():
        if args.include_ns and ns not in args.include_ns:
            continue
        for concept, payload in concepts.items():
            units = (payload.get("units") or {})
            for unit, series in units.items():
                if args.include_units and unit not in args.include_units:
                    continue
                for pt in (series or []):
                    val = pt.get("val", None)
                    if args.numeric_only and not is_number(val):
                        continue
                    fy = pt.get("fy", None)
                    if fy is not None:
                        try:
                            fy = int(fy)
                        except Exception:
                            fy = None
                    if args.min_fy is not None and (fy is None or fy < args.min_fy):
                        continue
                    if args.max_fy is not None and (fy is None or fy > args.max_fy):
                        continue
                    form = (pt.get("form") or "").strip()
                    if args.include_forms and form not in args.include_forms:
                        continue

                    rec = {
                        "cik": cik,
                        "entity": entity,
                        "ns": ns,
                        "concept": concept,               # namespaced later in KG build
                        "unit": unit,
                        "value": float(val) if is_number(val) else val,
                        "period_end": (pt.get("end") or "").strip(),
                        "accn": (pt.get("accn") or "").strip(),
                        "fy": fy,
                        "fp": (pt.get("fp") or "").strip(),
                        "form": form,
                        "filed": (pt.get("filed") or "").strip(),
                        "frame": (pt.get("frame") or "").strip(),
                    }
                    yield rec

def write_jsonl(records, out_path, latest_per_key=False):
    """
    Write records to JSONL with optional 'latest_per_key' reduction.
    Key = (cik, concept, unit, period_end)
    Keeps the most recent 'filed' date.
    """
    outp = pathlib.Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    if latest_per_key:
        buckets = defaultdict(list)
        for r in records:
            key = (r["cik"], r["concept"], r["unit"], r["period_end"])
            buckets[key].append(r)
        # Choose latest filed if available, else first
        chosen = []
        for key, rows in buckets.items():
            rows = sorted(rows, key=lambda x: (x.get("filed") or "", x.get("accn") or ""), reverse=True)
            chosen.append(rows[0])
        with outp.open("w", encoding="utf-8") as f:
            for r in chosen:
                f.write(json.dumps(r) + "\n")
                kept += 1
    else:
        with outp.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
                kept += 1
    return kept

def main():
    args = parse_args()

    ns_counter = Counter()
    unit_counter = Counter()
    concept_counter = Counter()
    files_seen = 0
    raw_count = 0

    # Collect + optionally reduce
    collected = []
    for fname, doc in iter_companyfacts(args.indir):
        files_seen += 1
        for rec in fact_records(doc, args):
            collected.append(rec)
            ns_counter[rec["ns"]] += 1
            unit_counter[rec["unit"]] += 1
            concept_counter[rec["concept"]] += 1
            raw_count += 1

    kept = write_jsonl(collected, args.out, latest_per_key=args.latest_per_key)

    # Write a summary next to the output
    summary = {
        "input_files": files_seen,
        "raw_records": raw_count,
        "kept_records": kept,
        "filters": {
            "include_ns": args.include_ns,
            "include_units": args.include_units,
            "include_forms": args.include_forms,
            "min_fy": args.min_fy,
            "max_fy": args.max_fy,
            "numeric_only": args.numeric_only,
            "latest_per_key": args.latest_per_key,
        },
        "top_ns": ns_counter.most_common(10),
        "top_units": unit_counter.most_common(10),
        "top_concepts": concept_counter.most_common(20),
    }
    summ_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[facts] files={files_seen} raw={raw_count} kept={kept} → {args.out}")
    print(f"[facts] summary → {summ_path}")

if __name__ == "__main__":
    main()
