#!/usr/bin/env python3
"""
Convert OpenBioLink OBL2021 dataset into a single .bin file compatible with skge hole run.
Keys and format matches wn18.bin which is used in the original run_hole.sh

Usage:
  python convert_obl_to_bin.py --data-dir /path/to/obl2021 --out data/obl2021.bin
"""
import argparse
from pathlib import Path
import numpy as np
import pickle

try:
    from openbiolink.obl2021 import OBL2021Dataset
except Exception as e:
    raise RuntimeError("Could not import OBL2021Dataset from openbiolink.obl2021: %s" % e)


def format_tensor(tensor):
    '''
    input: torch.tensor of shape (num_triples, 3) with order (subj, pred, obj)
    output: list of triples as (subj, obj, pred)
    '''
    return [
        tuple(
            int(tensor[i].cpu()[idx])
            for idx in [0,2,1]
        )
        for i in range(tensor.shape[0])
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', '-d', required=True, help='Directory with the OBL2021 files')
    p.add_argument('--out', '-o', default='data/obl2021.bin', help='Output file (will be npz compressed, extension can be .bin)')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_file = Path(args.out)

    ds = OBL2021Dataset(str(data_dir))

    train = format_tensor(ds.training)
    valid = format_tensor(ds.validation)
    test = format_tensor(ds.testing)

    out_dict = {
        'relations': list(ds._relation_label_to_id.keys()),
        'entities': list(ds._entity_label_to_id.keys()),
        'train_subs': train,
        'test_subs': test,
        'valid_subs': valid,
    }


    # Save compressed npz with .bin extension if requested
    with open(out_file, 'wb') as f:
        pickle.dump(out_dict, f)
    print("Saved", out_file, " with keys:", list(out_dict.keys()))


if __name__ == '__main__':
    main()