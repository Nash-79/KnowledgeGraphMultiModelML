# DATA_CHOICE — SEC EDGAR + XBRL (Free, Official)

## Rationale
- **Licence/Access:** Public U.S. SEC filings with free access via data.sec.gov endpoints.
- **Modalities:** Text (10-K/10-Q sections: MD&A, Risk Factors) + Structured Tables (XBRL facts).
- **KG-readiness:** Natural entities and relations:
  - Entities: Company (CIK), Filing, Statement, Concept (XBRL), Period, Unit.
  - Relations: company **reports** filing, filing **includes** statement, statement **contains** concept, concept **is-a** parent concept, concept **measured-in** unit, concept **for-period** period.
- **Feasibility:** Small universe (50–100 firms; 2–3 years) trains in hours; easy to compute SRS on concept/statement hierarchies.

## Scope (initial sample)
- Universe: pick ~50 CIKs (or a single sector).
- Years: 2022–2024.
- Filings: 10-K and 10-Q; primary docs + XBRL instance.

## Acceptance (Weeks 3–4)
- Fetch at least one 10-K + 10-Q per firm.
- Extract MD&A/Risk Factors and XBRL facts.
- Build KG CSVs with at least **two relation types** (*is-a* + one of *includes/contains/measured-in*).

