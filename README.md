# KG-MMML: Hybrid Knowledge Graph + ML

This repository contains the project workspace and thesis artefacts for a hybrid
knowledge-graph plus machine-learning system on SEC EDGAR data.

## Repository Layout

- `kg-mmml/` - Main implementation, experiments, tests, and technical documentation
- `kg-mmml/docs/` - Canonical internal documentation and thesis status references

## Final Status (Canonical)

Use `kg-mmml/docs/Architecture.md` as the authoritative source for final claims.

| Metric | Final Value | Interpretation |
|------|------|------|
| SRS (with RTF) | 0.8179 | Passes semantic preservation threshold (>=0.75) |
| Latency p99 (Annoy) | 0.037 ms | Passes latency threshold (<150 ms) |
| Micro-F1 gain (mean, n=5) | +1.42 pp | Below +3.0 pp gate (documented ceiling effect) |
| Macro-F1 gain (mean, n=5) | +3.32 pp | Statistically significant rare-class improvement |

## Metric Conventions

- Final thesis claims use multi-seed values from M10 (`n=5`) where available.
- Single-seed values (for example `+1.36pp`, `99.68%`) are milestone snapshots only.
- Structural SRS (`0.7571`) must be labelled as pre-RTF when cited.

## Reproducible and Re-runnable Workflow

For setup, deterministic reruns, and full command workflows, use:
- `kg-mmml/DEVELOPER_GUIDE.md`

Expected generated artifacts:

- `kg-mmml/reports/tables/m10_seed42_baseline_text_metrics.json`
- `kg-mmml/reports/tables/m10_seed42_text_concept_metrics.json`
- `kg-mmml/reports/tables/m10_statistical_summary.csv`
- `kg-mmml/reports/tables/m10_statistical_tests.json`

## Documentation Source of Truth

To remove ambiguity across milestone snapshots and final thesis files, use this precedence order:

1. `kg-mmml/docs/Architecture.md` (canonical final metrics and caveats)
2. `kg-mmml/docs/DOCUMENT_INDEX.md` (document ownership and status)
3. `kg-mmml/docs/DOCUMENTATION_STANDARDS.md` (terminology and formatting standards)

Historical progress files remain in the repository for auditability, but they are not final
source-of-truth references unless explicitly marked as current.

