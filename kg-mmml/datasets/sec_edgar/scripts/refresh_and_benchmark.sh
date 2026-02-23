#!/usr/bin/env bash
set -euo pipefail

CIX="datasets/sec_edgar/ciks.txt"
FACTS_DIR="data/processed/sec_edgar/companyfacts"
FACTS_OUT="data/processed/sec_edgar/facts.jsonl"
LAT_CSV="reports/tables/latency_baseline.csv"
LAT_META="reports/tables/latency_meta.json"

# 0) Augment CIKs (idempotent)
python -m datasets.sec_edgar.scripts.augment_ciks --in_file "$CIX" --out_file "$CIX"

# 1) Pull/refresh CompanyFacts (requires SEC_USER_AGENT env var)
python -m datasets.sec_edgar.scripts.download_companyfacts \
  --ciks_file "$CIX" \
  --out "$FACTS_DIR"

# 2) Normalise to facts.jsonl
python -m datasets.sec_edgar.scripts.companyfacts_to_facts \
  --in_dir "$FACTS_DIR" \
  --out "$FACTS_OUT"

# 3) Benchmark latency (exact/filtered/ANN/FAISS)
export OMP_NUM_THREADS=1
python -m src.cli.evaluate_latency \
  --facts "$FACTS_OUT" \
  --out "$LAT_CSV" \
  --meta_out "$LAT_META" \
  --sizes 1000 10000 \
  --queries 500 --k 10 --svd_dim 256 \
  --filtered --filter_cap 1200 \
  --use_annoy --use_faiss \
  --threads 1

echo "Done. See $LAT_CSV and $LAT_META"
