# Documentation Standards

This file defines how KG-MMML documents are written, interpreted, and cited.

## 1) Source-of-Truth Order

When values differ across files, use this precedence:

1. `Architecture.md`
2. `../../README.md`
3. `../DEVELOPER_GUIDE.md`
4. Technical glossary and standards in `docs/`
5. Archived files in `archive/` and historical report snapshots

## 2) Status Labels

Use one of these labels near the top of each operational document:

- `Current` - Intended for active use and current citation
- `Historical snapshot` - Valid only for the date/milestone named in the file
- `Archived` - Retained for traceability only
- `External source extract` - Verbatim source text; not normalised internally

## 3) Metric Conventions

- Use `pp` for percentage-point deltas (for example `+1.42pp`).
- Label `seed=42` values explicitly as single-seed snapshots.
- Label `0.7571` explicitly as pre-RTF structural SRS when used.
- Use `0.8179` as final SRS for final thesis claims.

## 4) Terminology Conventions

- Use `KG-MMML` as the project identifier.
- Use `is-a` in prose and `is_a` in schema/code.
- Use `micro-F1` and `macro-F1` with exact casing.
- Use `SEC EDGAR CompanyFacts` as the data-source name.

## 5) Language and Authorship Conventions

- Use UK English spelling in repository-authored prose (for example `normalise`, `optimise`, `behaviour`).
- Do not alter external titles, quoted text, code identifiers, or script names to force UK spelling.
- Final thesis wording should be human-reviewed and manually edited before submission.
- Where human-only authorship is required by policy, avoid generative drafting for final text blocks.
- If assistance tools are used for drafting or editing, keep provenance notes in compliance artefacts.

## 6) Ambiguity Rules

- Avoid relative timing without dates (`later`, `soon`, `next phase`) unless milestone dates are included.
- Avoid unqualified words such as `improved` or `better`; always include the metric and baseline.
- Avoid mixing final values with milestone values in the same table without explicit labels.
- Mark planned items as planned, and completed items as completed, with dates where possible.
