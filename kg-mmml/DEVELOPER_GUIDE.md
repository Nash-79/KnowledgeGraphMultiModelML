# Developer Guide

This guide defines the implementation workflow, metric conventions, and documentation rules for
the KG-MMML repository.

## Canonical Status and Scope

- **Final metrics and caveats:** `docs/Architecture.md`
- **Document ownership and status map:** `docs/DOCUMENT_INDEX.md`
- **Terminology and writing standards:** `docs/DOCUMENTATION_STANDARDS.md`

Use those three files before citing any value from milestone reports.

## Final Headline Outcomes

- `SRS (with RTF) = 0.8179` (passes `>= 0.75`)
- `Latency p99 (Annoy) = 0.037 ms` (passes `< 150 ms`)
- `Micro-F1 gain (mean, n=5) = +1.42 pp` (below `+3.0 pp` gate; ceiling effect documented)
- `Macro-F1 gain (mean, n=5) = +3.32 pp` (statistically significant)

## Environment Setup

```bash
make setup
make install-dev
```

## Core Workflows

### Train baseline (text-only)

```bash
python -m src.cli.baseline_tfidf \
  --facts data/processed/sec_edgar/facts.jsonl \
  --taxonomy datasets/sec_edgar/taxonomy/usgaap_combined.csv \
  --out reports/tables/baseline_text_seed42_metrics.json \
  --random_state 42 --test_size 0.25
```

### Train joint (text + concepts)

```bash
python -m src.cli.train_joint \
  --facts data/processed/sec_edgar/facts.jsonl \
  --taxonomy datasets/sec_edgar/taxonomy/usgaap_combined.csv \
  --concept_npz data/processed/sec_edgar/features/concept_features_filing.npz \
  --concept_index data/processed/sec_edgar/features/concept_features_index.csv \
  --consistency_weight 0.0 --epochs 20 --batch 128 --seed 42 \
  --out outputs/joint_with_concepts/metrics.json
```

### Build taxonomy

```bash
python -m src.cli.build_taxonomy \
  --facts data/processed/sec_edgar/facts.jsonl \
  --manual datasets/sec_edgar/taxonomy/usgaap_min.csv \
  --rules datasets/sec_edgar/taxonomy/pattern_rules.yaml \
  --out datasets/sec_edgar/taxonomy/usgaap_combined.csv \
  --min_cik_support 3 --with_closure
```

### Evaluate latency

```bash
python -m src.cli.evaluate_latency \
  --facts data/processed/sec_edgar/facts.jsonl \
  --sizes 1000 3218 --queries 500 --k 10 --svd_dim 256 \
  --filtered --use_annoy --use_faiss \
  --out reports/tables/latency_baseline_combined.csv \
  --meta_out reports/tables/latency_meta_combined.json
```

### Compute SRS

```bash
python -m src.cli.compute_srs \
  --config configs/experiment_joint.yaml \
  --out reports/tables/srs_kge_combined.csv \
  --rtf_score outputs/rtf_results/rtf_score.json
```

## Testing and Quality

```bash
make test
make test-cov
pytest tests/ -v -m unit
pytest tests/ -v -m integration
make lint
make format
```

## Data Pipeline

```text
SEC EDGAR CompanyFacts JSON
  -> normalise -> facts.jsonl
  -> build_taxonomy -> taxonomy CSV
  -> make_concept_features -> sparse concept matrix
  -> train -> model metrics
  -> evaluate_latency -> p50/p95/p99 reports
```

## Metric Definitions

- **HP (Hierarchy Presence):** concept coverage by `is-a` parent links
- **AtP (Attribute Predictability):** concept coverage by attribute/unit links
- **AP (Asymmetry Preservation):** directional integrity of hierarchy edges
- **RTF (Relation Type Fidelity):** embedding separation for relation types
- **SRS:** `0.25*HP + 0.20*AtP + 0.20*AP + 0.35*RTF`

When using `SRS=0.7571`, label it explicitly as **pre-RTF structural SRS**.

## Directory Guide

- `src/cli/` - executable CLI entry points
- `datasets/sec_edgar/` - committed source data resources and taxonomy assets
- `configs/` - pinned experiment configurations
- `reports/` - generated tables and figures
- `archive/` - superseded outputs

## Development Notes

- `src/cli/train.py` is baseline-only orchestration.
- Use `src/cli/train_joint.py` for joint model runs.
- Prefer `Annoy` for validated production-latency paths; use FAISS scripts for comparison studies.
