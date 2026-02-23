# src/utils/data_utils.py
"""Shared data processing utilities for KG-MMML project."""

import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set

def normalise_concept(ns: Optional[str], concept: Optional[str]) -> Optional[str]:
    """
    Normalise concept IDs to namespace:name format.

    Args:
        ns: Namespace (e.g., 'us-gaap')
        concept: Concept name (e.g., 'Assets')

    Returns:
        Normalised concept ID like 'us-gaap:Assets', or None if invalid
    """
    ns = (ns or "").strip()
    c = (concept or "").strip()
    if not c:
        return None
    if ns and not c.startswith(ns + ":"):
        return f"{ns}:{c}"
    return c if ":" in c else f"us-gaap:{c}"


def doc_id_from_fact(rec: dict) -> Optional[str]:
    """
    Extract standardised document ID from fact record.

    Args:
        rec: Fact record dictionary with 'cik' and 'accn' fields

    Returns:
        Document ID like 'filing_0000320193_000032019324000010', or None
    """
    cik = (rec.get("cik") or "").strip()
    accn = (rec.get("accn") or "").replace("-", "").strip()
    if cik and accn:
        return f"filing_{cik}_{accn}"
    elif cik:
        return f"company_{cik}"
    return None


def build_corpus_from_facts(
    facts_path: str,
    child_to_parents: Optional[Dict[str, Set[str]]] = None
) -> Tuple[List[str], List[str], List[List[str]], List[List[str]]]:
    """
    Build document corpus from facts.jsonl file.
    
    Args:
        facts_path: Path to facts.jsonl file
        child_to_parents: Optional taxonomy mapping for label extraction
    
    Returns:
        Tuple of (doc_ids, texts, labels, concept_lists)
        - doc_ids: List of document IDs
        - texts: List of space-joined concept tokens
        - labels: List of parent label sets (empty if no taxonomy)
        - concept_lists: List of full concept IDs per document
    """
    doc_tokens = defaultdict(list)
    doc_labels = defaultdict(set) if child_to_parents else None
    doc_concepts = defaultdict(list)
    
    with open(facts_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            did = doc_id_from_fact(rec)
            if not did:
                continue
            
            c = normalise_concept(rec.get("ns"), rec.get("concept"))
            if not c:
                continue
            
            # Token for text features (lowercased base name)
            token = c.split(":", 1)[1].lower() if ":" in c else c.lower()
            doc_tokens[did].append(token)
            doc_concepts[did].append(c)
            
            # Extract labels if taxonomy provided
            if child_to_parents:
                for p in child_to_parents.get(c, []):
                    doc_labels[did].add(p)
    
    docs = sorted(doc_tokens.keys())
    texts = [" ".join(doc_tokens[d]) for d in docs]
    labels = [sorted(doc_labels[d]) if doc_labels else [] for d in docs]
    concepts = [doc_concepts[d] for d in docs]
    
    return docs, texts, labels, concepts


def load_taxonomy_parents(tax_path: str) -> Dict[str, Set[str]]:
    """
    Load taxonomy as child -> parents mapping.
    
    Args:
        tax_path: Path to taxonomy CSV with 'child' and 'parent' columns
    
    Returns:
        Dictionary mapping child concept to set of parent concepts
    """
    import pandas as pd
    
    tax = pd.read_csv(tax_path)
    child_to_parents = defaultdict(set)
    
    for _, row in tax.iterrows():
        child = str(row["child"]).strip()
        parent = str(row["parent"]).strip()
        if child and parent:
            child_to_parents[child].add(parent)
    
    return dict(child_to_parents)