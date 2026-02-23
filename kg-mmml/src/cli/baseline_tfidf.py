# src/cli/baseline_tfidf.py
"""
Train and evaluate baseline sklearn LogisticRegression classifier.

Supports two modes:
1. Text-only: TF-IDF features from filing narratives
2. Text+concept: TF-IDF + binary concept indicators
"""
import argparse
import json
import pathlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from scipy import sparse

from ..utils.data_utils import (
    build_corpus_from_facts,
    load_taxonomy_parents
)


def align_and_concat(X_text, tfidf_docs, concept_npz, concept_index_csv, tfidf_docs_list):
    """Align concept-feature rows to TF-IDF rows and hstack."""
    import pandas as pd
    
    Xc = sparse.load_npz(concept_npz)
    idx = pd.read_csv(concept_index_csv)["doc_id"].tolist()
    row_of = {d: i for i, d in enumerate(idx)}
    
    # Build aligned rows for concept features
    sel = []
    for d in tfidf_docs_list:
        i = row_of.get(d, None)
        if i is None:
            sel.append(sparse.csr_matrix((1, Xc.shape[1]), dtype=Xc.dtype))
        else:
            sel.append(Xc[i])
    
    Xc_aligned = sparse.vstack(sel)
    return sparse.hstack([X_text, Xc_aligned], format="csr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="data/processed/sec_edgar/facts.jsonl")
    ap.add_argument("--taxonomy", default="datasets/sec_edgar/taxonomy/usgaap_combined.csv")
    ap.add_argument("--out", default="reports/tables/baseline_text_metrics.json")
    
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--max_features", type=int, default=20000)
    ap.add_argument("--min_df", type=int, default=2)
    
    # Optional KG-as-features concatenation
    ap.add_argument("--concept_features_npz", default="", 
                    help="Path to concept_features_filing.npz")
    ap.add_argument("--concept_features_index", default="", 
                    help="Path to concept_features_index.csv")
    args = ap.parse_args()
    
    # Load taxonomy
    tax_path = pathlib.Path(args.taxonomy)
    if not tax_path.exists():
        tax_path = pathlib.Path("datasets/sec_edgar/taxonomy/usgaap_min.csv")
    
    child_to_parents = load_taxonomy_parents(str(tax_path))
    
    # Build corpus with labels
    docs, texts, labels, _ = build_corpus_from_facts(args.facts, child_to_parents)
    
    # Filter docs with no labels
    keep = [i for i, l in enumerate(labels) if len(l) > 0]
    docs = [docs[i] for i in keep]
    texts = [texts[i] for i in keep]
    labels = [labels[i] for i in keep]
    
    if len(docs) < 20:
        raise RuntimeError(
            "Not enough labelled docs inferred from taxonomy. "
            "Add more taxonomy pairs or facts."
        )
    
    # Multi-label binarization
    mlb = MultiLabelBinarizer(sparse_output=False)
    Y = mlb.fit_transform(labels)
    label_names = list(mlb.classes_)
    
    # TF-IDF text features
    vec = TfidfVectorizer(max_features=args.max_features, min_df=args.min_df)
    X_text = vec.fit_transform(texts)
    
    # Optional: add concept features (KG-as-features)
    if args.concept_features_npz and args.concept_features_index:
        X = align_and_concat(
            X_text, docs, 
            args.concept_features_npz, 
            args.concept_features_index, 
            docs
        )
        mode = "text+concept"
    else:
        X = X_text
        mode = "text"
    
    # Train/test split (using sklearn for stratification - matches train_joint.py)
    train_idx, test_idx = train_test_split(
        np.arange(len(docs)),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=Y.argmax(1)  # Stratify by most-frequent label
    )
    
    Xtr, Xte = X[train_idx], X[test_idx]
    Ytr, Yte = Y[train_idx], Y[test_idx]
    
    # Classifier
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=200, n_jobs=None, solver="liblinear")
    )
    clf.fit(Xtr, Ytr)
    Yhat = clf.predict(Xte)
    
    n_total = len(docs)
    metrics = {
        "mode": mode,
        "n_docs_total": n_total,
        "n_docs_train": int(Xtr.shape[0]),
        "n_docs_test": int(Xte.shape[0]),
        "micro_f1": float(f1_score(Yte, Yhat, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Yte, Yhat, average="macro", zero_division=0)),
        "labels": label_names,
    }
    
    # Per-label report
    report = classification_report(
        Yte, Yhat, target_names=label_names, 
        output_dict=True, zero_division=0
    )
    metrics["per_label"] = report
    
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(metrics, indent=2))
    
    print(f"[baseline] wrote {args.out}")
    print(json.dumps({
        k: metrics[k] for k in ["mode", "micro_f1", "macro_f1"]
    }, indent=2))


if __name__ == "__main__":
    main()