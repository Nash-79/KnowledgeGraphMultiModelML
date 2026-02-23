# SEC Edgar Dataset

This directory contains scripts and documentation for working with SEC Edgar data.# SEC EDGAR + XBRL Starter

## Steps
1) Choose ~50 CIKs (zero-padded). Put them in `ciks.txt` (one per line).
2) Fetch index/filings (stub now):
   ```bash
   python scripts/fetch_filings.py --ciks $(cat ciks.txt) --out downloads
   ```
3) Parse XBRL facts (stub):
   ```bash
   python scripts/parse_xbrl.py --infile downloads/sample_xbrl.xml --outdir parsed
   ```
4) Build KG CSVs:
   ```bash
   python scripts/build_kg.py --index downloads/index.json --outdir data/kg
   ```
5) Train baseline & compute SRS using your main repo CLI.

