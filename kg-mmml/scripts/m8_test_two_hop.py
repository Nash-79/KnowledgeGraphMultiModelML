#!/usr/bin/env python3
"""
M8 Scalability Test 2: Two-Hop Graph Queries

Tests latency when expanding queries via graph traversal before vector ranking.

Approach:
1. Start with query concept
2. Traverse to parent concepts (via is-a edges)
3. Traverse to sibling concepts (parent's children)
4. Use expanded concept set for retrieval
5. Measure latency impact

Target: Two-hop expansion adds <50ms overhead vs one-hop

Usage:
    python scripts/m8_test_two_hop.py
"""

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_kg_edges(kg_edges_path):
    """Load KG edges and build adjacency structure."""
    edges = []
    parents = defaultdict(set)  # concept -> parent concepts
    children = defaultdict(set)  # concept -> child concepts

    with open(kg_edges_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src, edge_type, dst = row['src_id'], row['edge_type'], row['dst_id']
            edges.append((src, edge_type, dst))

            if edge_type == 'is-a':
                parents[src].add(dst)
                children[dst].add(src)

    return edges, parents, children


def expand_one_hop(concept, parents):
    """Expand to direct parents."""
    expanded = {concept}
    expanded.update(parents.get(concept, set()))
    return expanded


def expand_two_hop(concept, parents, children):
    """Expand to parents and siblings."""
    expanded = {concept}

    # Add parents
    parent_set = parents.get(concept, set())
    expanded.update(parent_set)

    # Add siblings (parent's other children)
    for parent in parent_set:
        expanded.update(children.get(parent, set()))

    return expanded


def benchmark_expansion(concepts, parents, children, n_queries=100):
    """Benchmark expansion overhead."""
    np.random.seed(42)
    query_concepts = np.random.choice(list(concepts), size=min(n_queries, len(concepts)), replace=False)

    # One-hop timings
    one_hop_times = []
    one_hop_sizes = []
    for concept in query_concepts:
        t0 = time.perf_counter()
        expanded = expand_one_hop(concept, parents)
        one_hop_times.append((time.perf_counter() - t0) * 1000.0)
        one_hop_sizes.append(len(expanded))

    # Two-hop timings
    two_hop_times = []
    two_hop_sizes = []
    for concept in query_concepts:
        t0 = time.perf_counter()
        expanded = expand_two_hop(concept, parents, children)
        two_hop_times.append((time.perf_counter() - t0) * 1000.0)
        two_hop_sizes.append(len(expanded))

    return {
        'one_hop': {
            'mean_ms': float(np.mean(one_hop_times)),
            'p99_ms': float(np.percentile(one_hop_times, 99)),
            'mean_size': float(np.mean(one_hop_sizes))
        },
        'two_hop': {
            'mean_ms': float(np.mean(two_hop_times)),
            'p99_ms': float(np.percentile(two_hop_times, 99)),
            'mean_size': float(np.mean(two_hop_sizes))
        }
    }


def main():
    print("\nM8 Scalability Test 2: Two-Hop Graph Queries\n")

    kg_edges_path = Path("data/kg/sec_edgar_2025-10-12_enhanced/kg_edges.csv")

    if not kg_edges_path.exists():
        print(f"Error: KG edges file not found: {kg_edges_path}")
        print("Run from project root: kg-mmml/")
        return 1

    print("Loading knowledge graph...")
    edges, parents, children = load_kg_edges(kg_edges_path)

    # Get concept nodes
    concepts = set()
    for src, edge_type, dst in edges:
        if edge_type in ['is-a', 'measured-in', 'for-period']:
            concepts.add(src)

    print(f"Loaded {len(edges)} edges")
    print(f"Found {len(concepts)} concept nodes")
    print(f"Hierarchy: {len(parents)} concepts with parents\n")

    print("Benchmarking graph expansion...")
    results = benchmark_expansion(concepts, parents, children, n_queries=500)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nOne-hop expansion (direct parents):")
    print(f"  Mean latency:    {results['one_hop']['mean_ms']:.4f}ms")
    print(f"  p99 latency:     {results['one_hop']['p99_ms']:.4f}ms")
    print(f"  Mean set size:   {results['one_hop']['mean_size']:.1f} concepts")

    print(f"\nTwo-hop expansion (parents + siblings):")
    print(f"  Mean latency:    {results['two_hop']['mean_ms']:.4f}ms")
    print(f"  p99 latency:     {results['two_hop']['p99_ms']:.4f}ms")
    print(f"  Mean set size:   {results['two_hop']['mean_size']:.1f} concepts")

    overhead = results['two_hop']['mean_ms'] - results['one_hop']['mean_ms']
    print(f"\nOverhead:")
    print(f"  Two-hop adds:    {overhead:.4f}ms")
    print(f"  Target:          <50ms")

    status = "PASS" if overhead < 50.0 else "FAIL"
    print(f"  Status:          {status}")

    # Save results
    output_file = Path("reports/tables/m8_two_hop_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_data = {
        'test': 'two-hop-expansion',
        'kg_edges': len(edges),
        'concepts': len(concepts),
        'one_hop': results['one_hop'],
        'two_hop': results['two_hop'],
        'overhead_ms': overhead,
        'target_ms': 50.0,
        'status': status
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n" + "=" * 70)
    print(f"Results saved to: {output_file}\n")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
