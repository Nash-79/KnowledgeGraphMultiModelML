#!/usr/bin/env python3
"""
M10 quick smoke test with a single seed.

Runs text-only and text+concept baseline_tfidf experiments for seed=42.
Useful for path/runtime validation before full 5-seed M10 execution.
"""

import json
import pathlib
import subprocess
import sys
from datetime import datetime


FACTS = "data/processed/sec_edgar/facts.jsonl"
TAXONOMY = "datasets/sec_edgar/taxonomy/usgaap_combined.csv"
CONCEPT_NPZ = "data/processed/sec_edgar/features/concept_features_filing.npz"
CONCEPT_INDEX = "data/processed/sec_edgar/features/concept_features_index.csv"


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def main():
    out_dir = pathlib.Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = 42
    start = datetime.now()

    print("=" * 68)
    print("M10 QUICK TEST: seed=42")
    print("=" * 68)

    base_out = out_dir / "m10_test_seed42_baseline.json"
    cmd_base = [
        sys.executable,
        "-m",
        "src.cli.baseline_tfidf",
        "--facts",
        FACTS,
        "--taxonomy",
        TAXONOMY,
        "--out",
        str(base_out),
        "--random_state",
        str(seed),
        "--test_size",
        "0.25",
    ]
    print("\n[1/2] Baseline text-only")
    run(cmd_base)
    base = json.loads(base_out.read_text(encoding="utf-8"))
    print(f"OK baseline: micro_f1={base['micro_f1']:.4f}, macro_f1={base['macro_f1']:.4f}")

    tc_out = out_dir / "m10_test_seed42_text_concept.json"
    cmd_tc = [
        sys.executable,
        "-m",
        "src.cli.baseline_tfidf",
        "--facts",
        FACTS,
        "--taxonomy",
        TAXONOMY,
        "--concept_features_npz",
        CONCEPT_NPZ,
        "--concept_features_index",
        CONCEPT_INDEX,
        "--out",
        str(tc_out),
        "--random_state",
        str(seed),
        "--test_size",
        "0.25",
    ]
    print("\n[2/2] Baseline text+concept")
    run(cmd_tc)
    tc = json.loads(tc_out.read_text(encoding="utf-8"))
    print(f"OK text+concept: micro_f1={tc['micro_f1']:.4f}, macro_f1={tc['macro_f1']:.4f}")

    micro_pp = (tc["micro_f1"] - base["micro_f1"]) * 100.0
    macro_pp = (tc["macro_f1"] - base["macro_f1"]) * 100.0
    secs = (datetime.now() - start).total_seconds()

    print("\n" + "=" * 68)
    print("M10 QUICK TEST: PASSED")
    print("=" * 68)
    print(f"micro-F1 improvement: {micro_pp:+.2f}pp")
    print(f"macro-F1 improvement: {macro_pp:+.2f}pp")
    print(f"runtime: {secs:.1f}s")
    print("=" * 68)


if __name__ == "__main__":
    main()

