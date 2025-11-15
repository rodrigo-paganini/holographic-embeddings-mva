#!/usr/bin/env python3
"""
Convert Hetionet dataset into a single .pkl file compatible with skge hole run.
Keys and format matches wn18.bin and obl2021.bin used in run_hole.sh

Hetionet comes in two files:
- nodes.tsv: columns "id name kind"
- edges.sif: columns "source metaedge target"

Usage:
  python convert_hetionet_to_pkl.py --data-dir data/hetionet --out data/hetionet.pkl
"""
import argparse
from pathlib import Path
import pickle
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', '-d', required=True, help='Directory with hetionet files (nodes.tsv, edges.sif)')
    p.add_argument('--out', '-o', default='data/hetionet.pkl', help='Output pickle file')
    p.add_argument('--train-frac', type=float, default=0.8, help='Fraction of edges for training (default 0.8)')
    p.add_argument('--valid-frac', type=float, default=0.1, help='Fraction of edges for validation (default 0.1)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for train/valid/test split')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    nodes_file = data_dir / 'nodes.tsv'
    edges_file = data_dir / 'edges.sif'
    out_file = Path(args.out)

    if not nodes_file.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_file}")
    if not edges_file.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_file}")

    # Read nodes and build entity mapping
    print(f"Reading nodes from {nodes_file}...")
    entity_to_id = {}
    entities = []
    
    with open(nodes_file, 'r') as f:
        header = f.readline().strip()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                node_id = parts[0]
                if node_id not in entity_to_id:
                    entity_to_id[node_id] = len(entities)
                    entities.append(node_id)
    
    print(f"  Found {len(entities)} unique entities")

    # Read edges and build relation mapping
    print(f"Reading edges from {edges_file}...")
    relation_to_id = {}
    relations = []
    triples = []
    
    with open(edges_file, 'r') as f:
        header = f.readline().strip()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                source = parts[0]
                metaedge = parts[1]
                target = parts[2]
                
                # Get or create entity IDs
                if source not in entity_to_id:
                    entity_to_id[source] = len(entities)
                    entities.append(source)
                if target not in entity_to_id:
                    entity_to_id[target] = len(entities)
                    entities.append(target)
                
                # Get or create relation ID
                if metaedge not in relation_to_id:
                    relation_to_id[metaedge] = len(relations)
                    relations.append(metaedge)
                
                s_id = entity_to_id[source]
                o_id = entity_to_id[target]
                p_id = relation_to_id[metaedge]
                
                # Format: (subject, object, predicate) to match OBL format
                triples.append((s_id, o_id, p_id))
    
    print(f"  Found {len(relations)} unique relations")
    print(f"  Found {len(triples)} total triples")

    # Split into train/valid/test
    import numpy as np
    np.random.seed(args.seed)
    
    indices = np.arange(len(triples))
    np.random.shuffle(indices)
    
    n_train = int(len(triples) * args.train_frac)
    n_valid = int(len(triples) * args.valid_frac)
    
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]
    
    train_subs = [triples[i] for i in train_idx]
    valid_subs = [triples[i] for i in valid_idx]
    test_subs = [triples[i] for i in test_idx]
    
    print(f"  Train: {len(train_subs)} triples")
    print(f"  Valid: {len(valid_subs)} triples")
    print(f"  Test: {len(test_subs)} triples")

    # Build output dictionary matching OBL/wn18 format
    out_dict = {
        'entities': entities,
        'relations': relations,
        'train_subs': train_subs,
        'valid_subs': valid_subs,
        'test_subs': test_subs,
    }

    # Save as pickle
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(out_dict, f)
    
    print(f"\nSaved {out_file} with keys: {list(out_dict.keys())}")
    print(f"  Entities: {len(entities)}")
    print(f"  Relations: {len(relations)}")
    print(f"  Total triples: {len(triples)}")


if __name__ == '__main__':
    main()
