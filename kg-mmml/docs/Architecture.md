# System Architecture

## Scope

This document describes the implemented KG-MMML architecture and removes ambiguity between
historical milestone snapshots and final thesis claims.

This file is the canonical source for final metrics and status interpretation.

## Architecture Overview

KG-MMML uses a hybrid graph-vector pipeline:

1. **Data layer**: SEC CompanyFacts normalised to `facts.jsonl`.
2. **Knowledge layer**: Taxonomy and graph snapshot with directed `is-a` relationships.
3. **Feature layer**: TF-IDF text features plus sparse binary concept indicators.
4. **Retrieval layer**: ANN index (Annoy) for low-latency ranking.
5. **Evaluation layer**: SRS, classification metrics, latency percentiles, and robustness checks.

## End-to-End Flow

`CompanyFacts JSON -> facts.jsonl -> taxonomy build -> graph snapshot -> feature extraction -> training -> evaluation`

## Production Configuration

- **Classifier**: `sklearn.linear_model.LogisticRegression`
- **Feature set**: TF-IDF + concept indicators
- **Split policy**: Stratified 75/25 train-test
- **Reference CLI**: `src/cli/baseline_tfidf.py`, `src/cli/train_joint.py`, `src/cli/evaluate_latency.py`

## Key Design Decisions

- **Baseline-first evaluation** prevents claiming gains from architecture changes alone.
- **Concept indicators over joint penalties** remain the default because constrained objectives did not
  improve final outcomes.
- **Annoy as default ANN** is retained due to strong p99 latency performance and simple operations.
- **Taxonomy automation** combines manual seed, regex rules, and frequency rules for scalable coverage.

## Status Clarification

- Values such as `99.68% micro-F1` and `+1.36pp` are valid **seed=42 milestone snapshots**.
- Final thesis claims use **multi-seed** values from M10 (for example `+1.42pp micro-F1 mean`).
- `SRS=0.7571` is **pre-RTF structural SRS**; final SRS is `0.8179` with RTF included.
