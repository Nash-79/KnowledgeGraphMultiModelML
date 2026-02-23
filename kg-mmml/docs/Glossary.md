# Glossary

Canonical terms and definitions used across KG-MMML documentation.

## Core Terms

- **KG-MMML**: Project shorthand for integrating knowledge graphs with machine learning pipelines.
- **Hybrid architecture**: Graph-based semantic constraints plus vector-based retrieval/ranking.
- **SEC EDGAR CompanyFacts**: Primary public data source for filings and concept facts.
- **Concept**: A financial reporting label (for example `us-gaap:Assets`).
- **Taxonomy**: Parent-child concept hierarchy used for semantic structure.

## Metric Terms

- **SRS (Semantic Retention Score)**: Weighted semantic-fidelity score combining HP, AtP, AP, and RTF.
- **HP (Hierarchy Presence)**: Fraction of concepts with at least one valid parent in taxonomy edges.
- **AtP (Attribute Predictability)**: Fraction of concepts with expected attribute links (for example units).
- **AP (Asymmetry Preservation)**: Degree to which directed hierarchy edges do not appear reversed.
- **RTF (Relation Type Fidelity)**: Probe-based measure of relation-type separability in embeddings.

## Reporting Terms

- **Final metrics**: Canonical metrics defined in `Architecture.md`.
- **Milestone snapshot**: Time-bound metrics in week-specific or milestone-specific documents.
- **Structural SRS**: SRS value calculated before RTF was included (`0.7571`).
- **Final SRS**: SRS value including RTF (`0.8179`).

## Writing Conventions

- Use **`is-a`** in prose.
- Use **`is_a`** only for schema fields, code identifiers, and CSV column names.
- Use percentage points (`pp`) for deltas between percentage metrics.
- Label seed-specific results explicitly as `seed=42 snapshot` when they are not multi-seed means.
