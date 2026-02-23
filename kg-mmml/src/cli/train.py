# src/cli/train.py
import argparse
import json
import os
import pathlib
import random
import time

import numpy as np
import torch
import yaml

from src.utils.data_utils import build_corpus_from_facts


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def first_present(dct, keys, default=None):
    """Return the first present key from a dict."""
    for k in keys:
        if k in dct and dct[k]:
            return dct[k]
    return default


def load_taxonomy(path):
    import pandas as pd

    df = pd.read_csv(path)
    mp = {}
    for _, r in df.iterrows():
        c, p = str(r["child"]).strip(), str(r["parent"]).strip()
        if c and p:
            mp.setdefault(c, set()).add(p)
    return mp


def run_baseline(cfg, seed):
    """TF-IDF baseline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer
    
    set_seed(seed)
    data_cfg = cfg.get("data", {})
    taxonomy = first_present(
        data_cfg,
        ["taxonomy", "taxonomy_path"],
        default="datasets/sec_edgar/taxonomy/usgaap_combined.csv",
    )
    facts = first_present(data_cfg, ["facts", "facts_path"])
    if not facts:
        raise ValueError("Config missing data path. Set one of: data.facts, data.facts_path")
    # Backward-compatible fallback for older configs.
    if not os.path.exists(facts):
        fallback = "data/processed/sec_edgar/facts.jsonl"
        if os.path.exists(fallback):
            facts = fallback
        else:
            raise FileNotFoundError(f"Facts file not found: {facts}")
    tax = load_taxonomy(taxonomy)
    _, texts, labels, _ = build_corpus_from_facts(facts, tax)
    
    # Filter docs with labels
    keep = [i for i, l in enumerate(labels) if len(l) > 0]
    texts = [texts[i] for i in keep]
    labels = [labels[i] for i in keep]
    if not labels:
        raise RuntimeError("No labelled documents found after taxonomy mapping.")
    
    # Features
    vec = TfidfVectorizer(min_df=2, max_features=20000)
    X = vec.fit_transform(texts)
    
    # Labels
    mlb = MultiLabelBinarizer(sparse_output=False)
    Y = mlb.fit_transform(labels)
    
    # Split
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.25, random_state=seed)
    
    # Train
    clf = OneVsRestClassifier(LogisticRegression(max_iter=200, solver="liblinear"))
    clf.fit(Xtr, Ytr)
    Yhat = clf.predict(Xte)
    
    return {"micro_f1": float(f1_score(Yte, Yhat, average="micro", zero_division=0)), "macro_f1": float(f1_score(Yte, Yhat, average="macro", zero_division=0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_type = cfg.get("model", {}).get("type", "tfidf")
    run_id = f"{cfg['experiment']['name']}_{int(time.time())}"
    os.makedirs("reports/tables", exist_ok=True)
    seeds = cfg.get("experiment", {}).get("seeds")
    if not seeds:
        one_seed = cfg.get("experiment", {}).get("seed", 13)
        seeds = [one_seed]
    
    results = []
    for s in seeds:
        if model_type in {"vl_baseline", "tfidf", "baseline_tfidf"}:
            metrics = run_baseline(cfg, s)
        elif model_type == "joint_model":
            raise NotImplementedError(
                "train.py does not implement joint training. Use src/cli/train_joint.py directly "
                "to avoid ambiguous placeholder metrics."
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        metrics["seed"] = s
        results.append(metrics)
    
    # Aggregate
    agg = {
        "micro_f1_mean": float(np.mean([r["micro_f1"] for r in results])),
        "micro_f1_std": float(np.std([r["micro_f1"] for r in results])),
        "macro_f1_mean": float(np.mean([r["macro_f1"] for r in results])),
        "seeds": seeds,
        "per_seed": results,
    }
    
    out_path = pathlib.Path(f"reports/tables/{run_id}_metrics.json")
    out_path.write_text(json.dumps(agg, indent=2))
    print(f"Saved: {out_path}")
    print(json.dumps({k: v for k, v in agg.items() if k != "per_seed"}, indent=2))

if __name__ == "__main__":
    main()
