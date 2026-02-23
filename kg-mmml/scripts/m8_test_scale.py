#!/usr/bin/env python3
"""
M8 Scalability Test 1: Large-Scale Retrieval

Tests latency performance at N=10,000 documents to validate
system scales beyond baseline experiments (N=3,218).

Benchmarks:
- Exact cosine (sparse TF-IDF)
- Graph-filtered cosine
- Annoy ANN
- FAISS HNSW

Target: All methods maintain <150ms p99 latency at scale.

Usage:
    python scripts/m8_test_scale.py
"""

import json
import subprocess
import sys
from pathlib import Path


def main():
    print("M8 Scalability Test 1: Large-Scale Retrieval\n")

    # Parameters
    facts_path = "data/facts.jsonl"
    out_csv = "reports/tables/m8_latency_scale.csv"
    out_meta = "reports/tables/m8_latency_scale_meta.json"

    # Test at baseline and 10k scale
    sizes = ["3218", "10000"]
    n_queries = 500

    print(f"Testing retrieval latency at N={', '.join(sizes)}")
    print(f"Methods: exact-cosine, filtered-cosine, annoy, faiss-hnsw")
    print(f"Queries: {n_queries} per size\n")

    # Build command
    cmd = [
        sys.executable, "-m", "src.cli.evaluate_latency",
        "--facts", facts_path,
        "--out", out_csv,
        "--meta_out", out_meta,
        "--sizes", *sizes,
        "--queries", str(n_queries),
        "--k", "10",
        "--svd_dim", "256",
        "--filtered",
        "--filter_cap", "1000",
        "--use_annoy",
        "--use_faiss",
        "--seed", "42"
    ]

    print("Running benchmark...")
    result = subprocess.run(cmd, cwd=Path.cwd())

    if result.returncode != 0:
        print(f"\nError: Benchmark failed with code {result.returncode}")
        return 1

    # Read and display results
    import pandas as pd
    df = pd.read_csv(out_csv)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for N in [int(s) for s in sizes]:
        print(f"\nN = {N} documents:")
        subset = df[df['N'] == N]
        for _, row in subset.iterrows():
            method = row['method']
            p99 = row['p99_ms']
            if pd.notna(p99):
                status = "PASS" if p99 < 150.0 else "FAIL"
                print(f"  {method:20s}  p99={p99:8.3f}ms  {status}")
            else:
                print(f"  {method:20s}  Not available")

    print("\n" + "=" * 70)
    print(f"\nResults saved to: {out_csv}")
    print(f"Metadata saved to: {out_meta}\n")

    # Check if all passed
    max_p99 = df['p99_ms'].max()
    if pd.notna(max_p99) and max_p99 < 150.0:
        print("[OK] All methods passed <150ms p99 target")
        return 0
    else:
        print("[WARN] Some methods exceeded 150ms p99 target")
        return 1


if __name__ == "__main__":
    sys.exit(main())
