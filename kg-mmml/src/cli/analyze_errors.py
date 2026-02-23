# src/cli/analyze_errors.py
"""
Analyze misclassifications from the sklearn text+concept baseline.

Replicates training pipeline, runs inference on test set, and exports
misclassified examples for manual review.
"""
import argparse
import pathlib
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

from ..utils.data_utils import (
    build_corpus_from_facts,
    load_taxonomy_parents
)

def align_and_concat(X_text, tfidf_docs, concept_npz, concept_index_csv, tfidf_docs_list):
    """Align concept-feature rows to TF-IDF rows and hstack."""
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
    ap.add_argument("--out", default="reports/tables/error_analysis_m9.csv")
    
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--max_features", type=int, default=20000)
    ap.add_argument("--min_df", type=int, default=2)
    
    # KG-as-features arguments (Required for production model)
    ap.add_argument("--concept_features_npz", required=True,
                    help="Path to concept_features_filing.npz")
    ap.add_argument("--concept_features_index", required=True,
                    help="Path to concept_features_index.csv")
    
    args = ap.parse_args()
    
    print("[M9] Loading data and taxonomy...")
    # Load taxonomy
    tax_path = pathlib.Path(args.taxonomy)
    child_to_parents = load_taxonomy_parents(str(tax_path))
    print(f"[DEBUG] Taxonomy size: {len(child_to_parents)}")
    if len(child_to_parents) > 0:
        print(f"[DEBUG] Sample taxonomy keys: {list(child_to_parents.keys())[:5]}")
    
    # Build corpus with labels
    docs, texts, labels, _ = build_corpus_from_facts(args.facts, child_to_parents)
    
    print(f"[DEBUG] Raw docs: {len(docs)}")
    print(f"[DEBUG] Docs with labels: {sum(1 for l in labels if len(l) > 0)}")
    
    # Filter docs with no labels
    keep = [i for i, l in enumerate(labels) if len(l) > 0]
    docs = [docs[i] for i in keep]
    texts = [texts[i] for i in keep]
    labels = [labels[i] for i in keep]
    
    print(f"[M9] Corpus size: {len(docs)} documents")
    
    if len(docs) == 0:
        print("[ERROR] No documents found after filtering! Check taxonomy matching.")
        return

    # Multi-label binarization
    mlb = MultiLabelBinarizer(sparse_output=False)
    Y = mlb.fit_transform(labels)
    label_names = list(mlb.classes_)
    
    # TF-IDF text features
    print("[M9] Vectorizing text...")
    try:
        vec = TfidfVectorizer(max_features=args.max_features, min_df=args.min_df)
        X_text = vec.fit_transform(texts)
    except ValueError as e:
        print(f"[ERROR] TfidfVectorizer failed: {e}")
        print(f"[DEBUG] First 3 texts: {texts[:3]}")
        raise
    
    # Add concept features (KG-as-features)
    print("[M9] Aligning concept features...")
    X = align_and_concat(
        X_text, docs, 
        args.concept_features_npz, 
        args.concept_features_index, 
        docs
    )
    
    # Train/test split (stratified)
    train_idx, test_idx = train_test_split(
        np.arange(len(docs)),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=Y.argmax(1)
    )
    
    Xtr, Xte = X[train_idx], X[test_idx]
    Ytr, Yte = Y[train_idx], Y[test_idx]
    
    # Train Classifier
    print("[M9] Training production model (LogisticRegression)...")
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=200, n_jobs=None, solver="liblinear")
    )
    clf.fit(Xtr, Ytr)
    
    # Predict
    print("[M9] Running inference on test set...")
    Yhat = clf.predict(Xte)
    Yprob = clf.predict_proba(Xte)
    
    # Identify Errors
    errors = []
    
    for i in range(len(test_idx)):
        true_vec = Yte[i]
        pred_vec = Yhat[i]
        
        # Check for mismatch
        if not np.array_equal(true_vec, pred_vec):
            doc_idx = test_idx[i]
            doc_id = docs[doc_idx]
            
            # Get label names
            true_labels = [label_names[j] for j, val in enumerate(true_vec) if val == 1]
            pred_labels = [label_names[j] for j, val in enumerate(pred_vec) if val == 1]
            
            # Jaccard Score
            intersection = len(set(true_labels) & set(pred_labels))
            union = len(set(true_labels) | set(pred_labels))
            jaccard = intersection / union if union > 0 else 0.0
            
            # Top confidence
            top_conf_idx = np.argmax(Yprob[i])
            top_conf_label = label_names[top_conf_idx]
            top_conf_score = Yprob[i][top_conf_idx]
            
            # Feature density (number of non-zero entries in X)
            feature_count = Xte[i].nnz
            
            errors.append({
                "doc_id": doc_id,
                "true_labels": ";".join(true_labels),
                "pred_labels": ";".join(pred_labels),
                "jaccard_score": round(jaccard, 4),
                "feature_count": feature_count,
                "top_conf_label": top_conf_label,
                "top_conf_score": round(top_conf_score, 4)
            })
            
    # Export to CSV
    df_errors = pd.DataFrame(errors)
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df_errors.to_csv(outp, index=False)
    
    print(f"[M9] Analysis complete.")
    print(f"      Total Test Docs: {len(test_idx)}")
    print(f"      Misclassifications: {len(errors)}")
    print(f"      Error Rate: {len(errors)/len(test_idx):.4f}")
    print(f"      Report saved to: {args.out}")

if __name__ == "__main__":
    main()
