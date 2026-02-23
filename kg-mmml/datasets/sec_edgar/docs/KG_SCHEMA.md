# Knowledge Graph Schema

# KG Schema — SEC EDGAR + XBRL

## Node types
- **Company**: `company_id` (CIK), `name`, `ticker` (optional)
- **Filing**: `filing_id` (accession), `form_type` (10-K/10-Q), `filed_date`
- **Statement**: `statement_id`, `statement_type` (BalanceSheet, IncomeStatement, CashFlow)
- **Concept**: `concept_id` (e.g., us-gaap:Assets), `label`, `namespace`
- **Unit**: `unit_id` (USD, shares)
- **Period**: `period_id` (YYYY-MM-DD/interval)

## Edge types
- **reports**: Company → Filing
- **includes**: Filing → Statement
- **contains**: Statement → Concept
- **is-a**: Concept → Concept (taxonomy parent/child)
- **measured-in**: Concept → Unit
- **for-period**: Concept → Period

## Minimal CSVs
- `kg_nodes.csv`: `node_id,type,attrs_json`
- `kg_edges.csv`: `src_id,edge_type,dst_id,attrs_json`

