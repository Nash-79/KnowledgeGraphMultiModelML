# src/cli/compute_rtf.py
"""
Compute Relation Type Fidelity (RTF) using probe classifier.
RTF measures whether embeddings preserve relation type distinctions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

def load_vocab(vocab_path):
    """Loads a vocabulary CSV into a dictionary."""
    df = pd.read_csv(vocab_path, index_col="entity_idx")
    return {row['entity_id']: idx for idx, row in df.iterrows()}

def create_probe_dataset(facts_path, entity_vocab, embeddings):
    """Creates a dataset for the relation prediction probe task."""
    X, y = [], []
    relation_encoder = LabelEncoder()
    
    relations = []
    with open(facts_path, 'r') as f:
        for line in f:
            fact = json.loads(line)
            relations.append(fact['relation'])
    
    # Fit the encoder on all possible relations
    relation_encoder.fit(sorted(list(set(relations))))

    with open(facts_path, 'r') as f:
        f.seek(0) # Reset file pointer
        for line in f:
            fact = json.loads(line)
            try:
                head_idx = entity_vocab[fact['head_id']]
                tail_idx = entity_vocab[fact['tail_id']]
                
                head_emb = embeddings[head_idx]
                tail_emb = embeddings[tail_idx]
                
                # Concatenate embeddings to form the feature vector
                feature_vector = np.concatenate([head_emb, tail_emb])
                X.append(feature_vector)
                
                relation_label = relation_encoder.transform([fact['relation']])[0]
                y.append(relation_label)
                
            except KeyError:
                # Skip facts with entities not in the vocab
                continue
    
    return np.array(X), np.array(y), relation_encoder

def main():
    parser = argparse.ArgumentParser(description="Compute RTF score")
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--embedding_dir", type=Path, required=True)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load embeddings
    try:
        entity_embeddings = np.load(args.embedding_dir / "entity_embeddings.npy")
        entity_vocab = load_vocab(args.embedding_dir / "entity_vocab.csv")
    except FileNotFoundError:
        print(f"Error: Embedding files not found in {args.embedding_dir}")
        print("Run train_kge.py first")
        return

    # Create probe dataset
    X, y, relation_encoder = create_probe_dataset(args.facts, entity_vocab, entity_embeddings)

    if len(X) == 0:
        print("Error: No valid samples created")
        return

    print(f"Dataset: {len(X)} samples, {len(relation_encoder.classes_)} relation types")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Train classifier
    classifier = LogisticRegression(random_state=args.seed, max_iter=1000)
    classifier.fit(X_train, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test)

    rtf_accuracy = accuracy_score(y_test, y_pred)
    rtf_f1_weighted = f1_score(y_test, y_pred, average='weighted')
    rtf_f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f"\nRTF Results:")
    print(f"  Accuracy: {rtf_accuracy:.4f}")
    print(f"  F1 (weighted): {rtf_f1_weighted:.4f}")
    print(f"  F1 (macro): {rtf_f1_macro:.4f}")

    # Save results
    results = {
        "rtf_accuracy": rtf_accuracy,
        "rtf_f1_weighted": rtf_f1_weighted,
        "rtf_f1_macro": rtf_f1_macro,
        "num_samples": len(X),
        "num_classes": len(relation_encoder.classes_),
        "embedding_dim": entity_embeddings.shape[1],
        "model": "LogisticRegression",
    }

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(args.outfile, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {args.outfile}")


if __name__ == "__main__":
    main()
