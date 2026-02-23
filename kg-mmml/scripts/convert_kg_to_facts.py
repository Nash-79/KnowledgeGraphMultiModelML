#!/usr/bin/env python3
"""
Convert KG edges CSV to facts.jsonl format for KGE training.

Reads kg_edges.csv and converts it to facts.jsonl format with:
{
  "head_id": "...",
  "relation": "...",
  "tail_id": "..."
}
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Convert KG edges to facts.jsonl")
    parser.add_argument("--kg_edges", type=Path, required=True, help="Path to kg_edges.csv")
    parser.add_argument("--outfile", type=Path, required=True, help="Path to output facts.jsonl")
    args = parser.parse_args()

    print(f"Loading KG edges from {args.kg_edges}...")
    df = pd.read_csv(args.kg_edges)

    print(f"Found {len(df)} edges")
    print(f"Edge types: {df['edge_type'].value_counts().to_dict()}")

    facts = []
    for _, row in df.iterrows():
        fact = {
            "head_id": row["src_id"],
            "relation": row["edge_type"],
            "tail_id": row["dst_id"]
        }
        facts.append(fact)

    args.outfile.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(facts)} facts to {args.outfile}...")
    with open(args.outfile, 'w') as f:
        for fact in facts:
            f.write(json.dumps(fact) + '\n')

    print("Conversion complete!")
    print(f"\nSample facts:")
    for i, fact in enumerate(facts[:5]):
        print(f"  {i+1}. ({fact['head_id']}) --[{fact['relation']}]-> ({fact['tail_id']})")


if __name__ == "__main__":
    main()
