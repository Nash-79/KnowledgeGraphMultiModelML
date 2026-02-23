# src/cli/evaluate_latency.py
"""
Benchmark retrieval latency across methods and index sizes.

Measures p50/p95/p99 latency for:
- Exact cosine similarity
- Graph-filtered cosine
- Annoy ANN
- FAISS HNSW
"""
import argparse
import json
import os
import time
import pathlib
import platform
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

from src.utils.data_utils import build_corpus_from_facts


def mem_mb():
    try:
        import psutil
        return int(psutil.Process().memory_info().rss / (1024*1024))
    except Exception:
        return None


def percentiles(ms):
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
    }


def run_exact_cosine(X, q_idx, k, drop_warmup=5):
    Xn = normalize(X, copy=True)
    qn = Xn[q_idx]
    _ = qn[0].dot(Xn.T).toarray().ravel()  # warm
    ms = []
    for i in range(len(q_idx) + drop_warmup):
        v = qn[i % len(q_idx)]
        t0 = time.perf_counter()
        _ = v.dot(Xn.T).toarray().ravel().argpartition(-k)[-k:]
        if i >= drop_warmup:
            ms.append((time.perf_counter() - t0) * 1000.0)
    return ms


def build_filtered_candidates(concept_lists, docs, cap=1000):
    inv = defaultdict(set)
    for i, d in enumerate(docs):
        for t in set(concept_lists[i]):
            inv[t].add(d)
    return inv, cap


def run_filtered_cosine(X, q_idx, k, docs, concept_lists, inv, cap, drop_warmup=5):
    Xn = normalize(X, copy=True)
    ms = []
    sizes = []
    pos = {doc: ix for ix, doc in enumerate(docs)}
    
    for i in range(len(q_idx) + drop_warmup):
        qi = q_idx[i % len(q_idx)]
        cands = set()
        for t in set(concept_lists[qi]):
            cands |= inv.get(t, set())
        
        idx = [pos[c] for c in cands if c in pos]
        if len(idx) > cap:
            idx = idx[:cap]
        sizes.append(len(idx))
        
        if len(idx) == 0:
            continue
        
        t0 = time.perf_counter()
        v = Xn[qi]
        sub = Xn[idx]
        _ = v.dot(sub.T).toarray().ravel().argpartition(-k)[-k:]
        if i >= drop_warmup:
            ms.append((time.perf_counter() - t0) * 1000.0)
    
    avg_size = float(np.mean(sizes[drop_warmup:])) if sizes[drop_warmup:] else 0.0
    return ms, avg_size


def run_annoy(Xd, q_idx, k, trees=20, drop_warmup=5):
    from annoy import AnnoyIndex
    d = Xd.shape[1]
    ann = AnnoyIndex(d, metric='angular')
    for i in range(Xd.shape[0]):
        ann.add_item(i, Xd[i].astype(np.float32).tolist())
    ann.build(trees)
    _ = ann.get_nns_by_vector(Xd[q_idx[0]].astype(np.float32).tolist(), k)
    ms = []
    for i in range(len(q_idx) + drop_warmup):
        v = Xd[q_idx[i % len(q_idx)]].astype(np.float32).tolist()
        t0 = time.perf_counter()
        _ = ann.get_nns_by_vector(v, k)
        if i >= drop_warmup:
            ms.append((time.perf_counter() - t0) * 1000.0)
    return ms


def run_faiss_hnsw(Xd, q_idx, k, M=32, ef=200, drop_warmup=5):
    import faiss
    d = Xd.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efSearch = ef
    index.hnsw.efConstruction = ef
    index.add(Xd.astype("float32"))
    _ = index.search(Xd[q_idx[:1]].astype("float32"), k)
    ms = []
    for i in range(len(q_idx) + drop_warmup):
        t0 = time.perf_counter()
        q = Xd[q_idx[i % len(q_idx)] : q_idx[i % len(q_idx)] + 1].astype("float32")
        _ = index.search(q, k)
        if i >= drop_warmup:
            ms.append((time.perf_counter() - t0) * 1000.0)
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="data/processed/sec_edgar/facts.jsonl")
    ap.add_argument("--out", default="reports/tables/latency_baseline.csv")
    ap.add_argument("--meta_out", default="reports/tables/latency_meta.json")
    ap.add_argument("--sizes", nargs="+", default=["1000", "10000"])
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--svd_dim", type=int, default=256)
    ap.add_argument("--filtered", action="store_true")
    ap.add_argument("--filter_cap", type=int, default=1000)
    ap.add_argument("--use_annoy", action="store_true")
    ap.add_argument("--use_faiss", action="store_true")
    ap.add_argument("--drop_warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=1, help="Set OMP/BLAS threads")
    args = ap.parse_args()
    
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    np.random.seed(args.seed)
    
    # Build corpus (no taxonomy needed for latency)
    docs, texts, _, concept_lists = build_corpus_from_facts(args.facts)
    
    if len(docs) == 0:
        raise SystemExit("No documents built from facts.jsonl — cannot benchmark.")
    
    vec = TfidfVectorizer(min_df=2, max_features=50000)
    X = vec.fit_transform(texts)
    
    sizes = [min(int(s), X.shape[0]) for s in args.sizes]
    rows = []
    
    # Pre-build filtered candidate index if needed
    inv = None
    if args.filtered:
        inv, cap = build_filtered_candidates(concept_lists, docs, cap=args.filter_cap)
    
    # Optional dense projection cache
    Xd_cache = {}
    
    for N in sizes:
        XN = X[:N]
        qn = min(args.queries, N)
        q_idx = np.random.choice(N, size=qn, replace=False)
        
        # Exact cosine
        ms = run_exact_cosine(XN, q_idx, args.k, drop_warmup=args.drop_warmup)
        rows.append({
            "N": N, "method": "exact-cosine", "dim": "tfidf",
            **percentiles(ms), "q": len(ms), "memory_mb": mem_mb(),
            "notes": "sparse dot"
        })
        
        # Filtered cosine
        if args.filtered:
            ms, avg = run_filtered_cosine(
                XN, q_idx, args.k, docs[:N], concept_lists[:N],
                inv, args.filter_cap, drop_warmup=args.drop_warmup
            )
            rows.append({
                "N": N, "method": "filtered-cosine", "dim": "tfidf",
                **percentiles(ms), "q": len(ms), "memory_mb": mem_mb(),
                "notes": f"graph-filter≈{int(round(avg))}"
            })
        
        # Shared SVD projection
        if args.use_annoy or args.use_faiss:
            if N not in Xd_cache:
                svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed)
                Xd_cache[N] = normalize(svd.fit_transform(XN)).astype("float32")
            Xd = Xd_cache[N]
        
        # Annoy
        if args.use_annoy:
            try:
                from annoy import AnnoyIndex  # noqa
                ms = run_annoy(Xd, q_idx, args.k, trees=20, drop_warmup=args.drop_warmup)
                rows.append({
                    "N": N, "method": "annoy", "dim": args.svd_dim,
                    **percentiles(ms), "q": len(ms), "memory_mb": mem_mb(),
                    "notes": "20 trees"
                })
            except Exception as e:
                rows.append({
                    "N": N, "method": "annoy", "dim": "-",
                    "p50_ms": None, "p95_ms": None, "p99_ms": None,
                    "q": 0, "memory_mb": mem_mb(),
                    "notes": f"annoy not available: {e}"
                })
        
        # FAISS HNSW
        if args.use_faiss:
            try:
                import faiss  # noqa
                ms = run_faiss_hnsw(
                    Xd, q_idx, args.k, M=32, ef=200,
                    drop_warmup=args.drop_warmup
                )
                rows.append({
                    "N": N, "method": "faiss-hnsw", "dim": args.svd_dim,
                    **percentiles(ms), "q": len(ms), "memory_mb": mem_mb(),
                    "notes": "M=32, ef=200"
                })
            except Exception as e:
                rows.append({
                    "N": N, "method": "faiss-hnsw", "dim": "-",
                    "p50_ms": None, "p95_ms": None, "p99_ms": None,
                    "q": 0, "memory_mb": mem_mb(),
                    "notes": f"faiss not available: {e}"
                })
    
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outp, index=False)
    
    meta = {
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "threads": args.threads,
        },
        "params": {
            "sizes": sizes, "queries": args.queries, "k": args.k,
            "svd_dim": args.svd_dim, "drop_warmup": args.drop_warmup,
            "filtered": args.filtered, "filter_cap": args.filter_cap,
            "use_annoy": args.use_annoy, "use_faiss": args.use_faiss,
            "seed": args.seed,
        }
    }
    pathlib.Path(args.meta_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.meta_out).write_text(json.dumps(meta, indent=2))
    print(f"[latency] wrote {outp} and {args.meta_out}")

if __name__ == "__main__":
    main()