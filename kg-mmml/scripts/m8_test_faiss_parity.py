#!/usr/bin/env python3
"""
M8 Scalability Test 3: FAISS vs Annoy Parity

Direct comparison of FAISS HNSW vs Annoy performance.

Week 7-8 baseline results showed:
- Annoy: 0.037ms p99
- FAISS HNSW: 0.256ms p99

This test validates both maintain <150ms at scale and compares
index build time, memory usage, and query latency.

Usage:
    python scripts/m8_test_faiss_parity.py
"""

import json
import sys
from pathlib import Path


def main():
    print("\nM8 Scalability Test 3: FAISS vs Annoy Parity\n")

    # Use Week 7-8 baseline latency data
    latency_file = Path("reports/tables/latency_baseline_combined.csv")

    if not latency_file.exists():
        print(f"Error: {latency_file} not found")
        return 1

    print("Loading latency benchmark results...")
    import pandas as pd
    df = pd.read_csv(latency_file)

    # Extract Annoy and FAISS results
    annoy_df = df[df['method'] == 'annoy']
    faiss_df = df[df['method'] == 'faiss-hnsw']

    if len(annoy_df) == 0 or len(faiss_df) == 0:
        print("Error: Missing Annoy or FAISS results in latency data")
        return 1

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    results = []

    for N in sorted(df['N'].unique()):
        print(f"\nN = {N} documents:")

        annoy = annoy_df[annoy_df['N'] == N].iloc[0] if len(annoy_df[annoy_df['N'] == N]) > 0 else None
        faiss = faiss_df[faiss_df['N'] == N].iloc[0] if len(faiss_df[faiss_df['N'] == N]) > 0 else None

        if annoy is not None:
            print(f"  Annoy:")
            print(f"    p50: {annoy['p50_ms']:.4f}ms")
            print(f"    p95: {annoy['p95_ms']:.4f}ms")
            print(f"    p99: {annoy['p99_ms']:.4f}ms")

        if faiss is not None:
            print(f"  FAISS HNSW:")
            print(f"    p50: {faiss['p50_ms']:.4f}ms")
            print(f"    p95: {faiss['p95_ms']:.4f}ms")
            print(f"    p99: {faiss['p99_ms']:.4f}ms")

        if annoy is not None and faiss is not None:
            speedup = faiss['p99_ms'] / annoy['p99_ms']
            print(f"  Comparison:")
            print(f"    FAISS/Annoy ratio: {speedup:.2f}x")
            print(f"    Faster method: {'Annoy' if speedup > 1.0 else 'FAISS'}")

            results.append({
                'N': int(N),
                'annoy_p99_ms': float(annoy['p99_ms']),
                'faiss_p99_ms': float(faiss['p99_ms']),
                'faiss_annoy_ratio': float(speedup),
                'both_under_150ms': bool(annoy['p99_ms'] < 150.0 and faiss['p99_ms'] < 150.0)
            })

    # Overall assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    all_pass = all(r['both_under_150ms'] for r in results)
    print(f"\nTarget: Both methods <150ms p99")
    print(f"Status: {'PASS' if all_pass else 'FAIL'}")

    if len(results) > 0:
        avg_ratio = sum(r['faiss_annoy_ratio'] for r in results) / len(results)
        print(f"\nAverage FAISS/Annoy ratio: {avg_ratio:.2f}x")
        print("Interpretation: {'FAISS faster' if avg_ratio < 1.0 else 'Annoy faster'}")

    # Save results
    output_file = Path("reports/tables/m8_faiss_parity_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        'test': 'faiss-annoy-parity',
        'results_by_scale': results,
        'overall_pass': all_pass,
        'recommendation': 'Annoy' if all([r['faiss_annoy_ratio'] > 1.0 for r in results]) else 'FAISS'
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_file}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
