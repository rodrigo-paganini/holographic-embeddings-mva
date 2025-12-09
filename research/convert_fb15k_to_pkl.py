#!/usr/bin/env python3
"""
Convert FB15k dataset into a single .pkl file compatible with skge hole run.
Keys and format matches wn18.bin, obl2021.bin, and hetionet.pkl

Downlaod from: https://github.com/TimDettmers/ConvE/blob/master/WN18RR.tar.gz

FB15k comes in three files:
- train.txt: tab-separated triples "subject\trelation\tobject"
- valid.txt: tab-separated triples
- test.txt: tab-separated triples

Usage:
  python convert_fb15k_to_pkl.py --data-dir data/FB15k --out data/fb15k.pkl
"""
import argparse
from pathlib import Path
import pickle
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', '-d', default='data/FB15k', help='Directory with FB15k files (train.txt, valid.txt, test.txt)')
    p.add_argument('--out', '-o', default='data/fb15k.pkl', help='Output pickle file')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    train_file = data_dir / 'freebase_mtr100_mte100-train.txt'
    valid_file = data_dir / 'freebase_mtr100_mte100-valid.txt'
    test_file = data_dir / 'freebase_mtr100_mte100-test.txt'
    out_file = Path(args.out)

    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not valid_file.exists():
        raise FileNotFoundError(f"Valid file not found: {valid_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # Build entity and relation mappings
    entity_to_id = {}
    entities = []
    relation_to_id = {}
    relations = []
    
    def process_file(filepath, split_name):
        """Read a file and return list of triples"""
        triples = []
        print(f"Reading {split_name} from {filepath}...")
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    subject = parts[0]
                    relation = parts[1]
                    obj = parts[2]
                    
                    # Get or create entity IDs
                    if subject not in entity_to_id:
                        entity_to_id[subject] = len(entities)
                        entities.append(subject)
                    if obj not in entity_to_id:
                        entity_to_id[obj] = len(entities)
                        entities.append(obj)
                    
                    # Get or create relation ID
                    if relation not in relation_to_id:
                        relation_to_id[relation] = len(relations)
                        relations.append(relation)
                    
                    s_id = entity_to_id[subject]
                    o_id = entity_to_id[obj]
                    p_id = relation_to_id[relation]
                    
                    # Format: (subject, object, predicate) to match other datasets
                    triples.append((s_id, o_id, p_id))
        
        print(f"  Found {len(triples)} triples")
        return triples
    
    # Process all three splits
    train_subs = process_file(train_file, "train")
    valid_subs = process_file(valid_file, "valid")
    test_subs = process_file(test_file, "test")
    
    print(f"\nDataset summary:")
    print(f"  Entities: {len(entities)}")
    print(f"  Relations: {len(relations)}")
    print(f"  Train: {len(train_subs)} triples")
    print(f"  Valid: {len(valid_subs)} triples")
    print(f"  Test: {len(test_subs)} triples")
    print(f"  Total: {len(train_subs) + len(valid_subs) + len(test_subs)} triples")

    # Build output dictionary matching OBL/wn18/hetionet format
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


if __name__ == '__main__':
    main()
