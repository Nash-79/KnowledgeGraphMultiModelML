# src/cli/train_kge.py
"""
Train TransE model to generate entity embeddings for RTF computation.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class TransE(nn.Module):
    """TransE model: head + relation ≈ tail in embedding space."""
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, positive_triples, negative_triples):
        # Scores for positive triples
        pos_h, pos_r, pos_t = positive_triples.T
        pos_h_emb = self.entity_embeddings(pos_h)
        pos_r_emb = self.relation_embeddings(pos_r)
        pos_t_emb = self.entity_embeddings(pos_t)
        pos_score = torch.norm(pos_h_emb + pos_r_emb - pos_t_emb, p=2, dim=1)

        # Scores for negative triples
        neg_h, neg_r, neg_t = negative_triples.T
        neg_h_emb = self.entity_embeddings(neg_h)
        neg_r_emb = self.relation_embeddings(neg_r)
        neg_t_emb = self.entity_embeddings(neg_t)
        neg_score = torch.norm(neg_h_emb + neg_r_emb - neg_t_emb, p=2, dim=1)

        # Margin-based ranking loss
        loss = torch.relu(self.margin + pos_score - neg_score).mean()
        return loss

def create_triples(facts_path, entity_vocab, relation_vocab):
    triples = []
    with open(facts_path, 'r') as f:
        for line in f:
            fact = json.loads(line)
            try:
                h = entity_vocab[fact['head_id']]
                r = relation_vocab[fact['relation']]
                t = entity_vocab[fact['tail_id']]
                triples.append([h, r, t])
            except KeyError:
                # Skip facts with entities/relations not in the vocab
                continue
    return torch.LongTensor(triples)

def main():
    parser = argparse.ArgumentParser(description="Train TransE KGE model")
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Building vocabularies...")
    entities = set()
    relations = set()
    with open(args.facts, 'r') as f:
        for line in f:
            fact = json.loads(line)
            entities.add(fact['head_id'])
            entities.add(fact['tail_id'])
            relations.add(fact['relation'])

    entity_list = sorted(list(entities))
    relation_list = sorted(list(relations))
    entity_to_idx = {name: i for i, name in enumerate(entity_list)}
    relation_to_idx = {name: i for i, name in enumerate(relation_list)}
    
    num_entities = len(entity_to_idx)
    num_relations = len(relation_to_idx)

    print(f"Found {num_entities} entities and {num_relations} relations")

    print("Creating triples...")
    triples = create_triples(args.facts, entity_to_idx, relation_to_idx)

    model = TransE(num_entities, num_relations, args.embedding_dim, args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # Create a new DataLoader in each epoch to shuffle data
        dataloader = torch.utils.data.DataLoader(triples, batch_size=args.batch_size, shuffle=True)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_triples in pbar:
            optimizer.zero_grad()
            
            # Generate negative samples by corrupting either head or tail
            negative_triples = batch_triples.clone()
            corruption_mask = torch.rand(len(negative_triples)) < 0.5
            
            # Corrupt tails
            num_corrupt_tails = corruption_mask.sum()
            if num_corrupt_tails > 0:
                random_tails = torch.randint(0, num_entities, (num_corrupt_tails,))
                negative_triples[corruption_mask, 2] = random_tails

            # Corrupt heads
            num_corrupt_heads = (~corruption_mask).sum()
            if num_corrupt_heads > 0:
                random_heads = torch.randint(0, num_entities, (num_corrupt_heads,))
                negative_triples[~corruption_mask, 0] = random_heads

            loss = model(batch_triples, negative_triples)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {total_loss / len(dataloader):.4f}")

    print("Training complete")

    print(f"Saving embeddings to {args.outdir}")
    
    # Save entity embeddings
    entity_embeddings = model.entity_embeddings.weight.data.cpu().numpy()
    np.save(args.outdir / "entity_embeddings.npy", entity_embeddings)

    # Save entity vocabulary
    entity_vocab_df = pd.DataFrame(entity_list, columns=["entity_id"])
    entity_vocab_df.to_csv(args.outdir / "entity_vocab.csv", index_label="entity_idx")

    # Save relation embeddings (for potential future use)
    relation_embeddings = model.relation_embeddings.weight.data.cpu().numpy()
    np.save(args.outdir / "relation_embeddings.npy", relation_embeddings)

    # Save relation vocabulary
    relation_vocab_df = pd.DataFrame(relation_list, columns=["relation_id"])
    relation_vocab_df.to_csv(args.outdir / "relation_vocab.csv", index_label="relation_idx")

    print("Done")


if __name__ == "__main__":
    main()
