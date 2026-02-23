# src/cli/train_joint.py
"""
Train joint text+concept classification with optional consistency penalty.

PyTorch linear model combining TF-IDF text features and binary concept indicators.
Consistency penalty λ regularizes predictions against taxonomy hierarchy.

Ablation study showed λ=0.0 outperforms constrained variants (see Week 7-8 progress).
"""
import argparse
import json
import pathlib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from src.utils.data_utils import (
    build_corpus_from_facts,
    load_taxonomy_parents
)


class LogReg(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
    
    def forward(self, x):
        return self.lin(x)


def make_parent_support(concept_lists, parents_vocab, child_to_parents):
    """
    Build support vectors per doc: count how many observed children 
    map to each parent.
    """
    pv = {p: i for i, p in enumerate(parents_vocab)}
    S = np.zeros((len(concept_lists), len(parents_vocab)), dtype=np.float32)
    
    for i, children in enumerate(concept_lists):
        for ch in children:
            for p in child_to_parents.get(ch, []):
                j = pv.get(p)
                if j is not None:
                    S[i, j] += 1.0
    
    # Normalize rows to sum=1 (if zero, leave zeros)
    row_sums = S.sum(1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return S / row_sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="data/processed/sec_edgar/facts.jsonl")
    ap.add_argument("--taxonomy", default="datasets/sec_edgar/taxonomy/usgaap_combined.csv")
    ap.add_argument("--concept_npz", default="", 
                    help="optional concept features (CSR .npz)")
    ap.add_argument("--concept_index", default="")
    ap.add_argument("--out", default="reports/tables/joint_metrics.json")
    ap.add_argument("--consistency_weight", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load taxonomy and build corpus
    child_to_parents = load_taxonomy_parents(args.taxonomy)
    docs, texts, labels_list, concept_lists = build_corpus_from_facts(
        args.facts, child_to_parents
    )
    
    # Filter docs with labels
    keep = [i for i, l in enumerate(labels_list) if len(l) > 0]
    docs = [docs[i] for i in keep]
    texts = [texts[i] for i in keep]
    labels_list = [labels_list[i] for i in keep]
    concept_lists = [concept_lists[i] for i in keep]
    
    # Multi-label binarization (parents)
    mlb = MultiLabelBinarizer(sparse_output=False)
    Y = mlb.fit_transform(labels_list).astype(np.float32)
    parents_vocab = list(mlb.classes_)
    
    # Text features
    vec = TfidfVectorizer(min_df=2, max_features=50000)
    Xt = vec.fit_transform(texts)
    
    # Optional concept features
    if args.concept_npz and args.concept_index:
        Xc = sparse.load_npz(args.concept_npz)
        idx = pd.read_csv(args.concept_index)["doc_id"].tolist()
        pos = {d: i for i, d in enumerate(idx)}
        sel = [pos.get(d, None) for d in docs]
        sel_rows = [
            Xc[i] if i is not None else sparse.csr_matrix((1, Xc.shape[1])) 
            for i in sel
        ]
        Xc_aligned = sparse.vstack(sel_rows)
        X = sparse.hstack([Xt, Xc_aligned]).tocsr()
    else:
        X = Xt
    
    # Train/test split
    tr, te = train_test_split(
        np.arange(X.shape[0]), 
        test_size=0.25, 
        random_state=args.seed, 
        stratify=Y.argmax(1)
    )
    Xtr, Xte = X[tr], X[te]
    Ytr, Yte = Y[tr], Y[te]
    
    # Normalize to l2 (dense) for torch
    Xtr = normalize(Xtr).astype(np.float32).toarray()
    Xte = normalize(Xte).astype(np.float32).toarray()
    
    S_all = make_parent_support(concept_lists, parents_vocab, child_to_parents)
    Str, _ = S_all[tr], S_all[te]
    
    d_in, d_out = Xtr.shape[1], Y.shape[1]
    model = LogReg(d_in, d_out)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    
    def run_epoch(Xn, Yn, Sn, train=True):
        model.train(train)
        idx = np.arange(Xn.shape[0])
        np.random.shuffle(idx)
        bs = args.batch
        total = 0.0
        for s in range(0, len(idx), bs):
            j = idx[s:s+bs]
            xb = torch.from_numpy(Xn[j])
            yb = torch.from_numpy(Yn[j])
            sb = torch.from_numpy(Sn[j])
            logits = model(xb)
            loss = bce(logits, yb)
            if args.consistency_weight > 0:
                prob = torch.sigmoid(logits)
                loss = loss + args.consistency_weight * mse(prob, sb)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(j)
        return total / len(idx)
    
    def eval_metrics(Xn, Yn):
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xn))
            prob = torch.sigmoid(logits).numpy()
        from sklearn.metrics import f1_score
        Yhat = (prob >= 0.5).astype(np.float32)
        return {
            "micro_f1": float(f1_score(Yn, Yhat, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(Yn, Yhat, average="macro", zero_division=0)),
        }
    
    for ep in range(args.epochs):
        _ = run_epoch(Xtr, Ytr, Str, train=True)
    
    metrics = {
        "epochs": args.epochs,
        "consistency_weight": args.consistency_weight,
        "train": eval_metrics(Xtr, Ytr),
        "test": eval_metrics(Xte, Yte),
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "labels": parents_vocab,
    }
    
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(metrics, indent=2))
    
    print(json.dumps({
        "mode": "joint",
        "micro_f1": metrics["test"]["micro_f1"],
        "macro_f1": metrics["test"]["macro_f1"]
    }, indent=2))


if __name__ == "__main__":
    main()