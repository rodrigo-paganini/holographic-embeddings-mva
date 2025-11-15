#!/bin/sh

python kg/run_hole.py --fin data/hetionet.pkl \
       --test-all 50 --nb 100 --me 500 \
       --margin 0.2 --lr 0.1 --ncomp 150 \
       --fout results/hole_hetionet.pkl \
