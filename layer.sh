#!/bin/bash

echo "Datset: $1"
echo "GPU: $2"

mkdir -p output
mkdir -p output/layer
mkdir -p output/layer/$1

for l in 1 2 4 8 16 32
do
    # MLP
    python train.py --layer $l --data $1 --gpu $2 | tee output/layer/$1/mlp_$l.out

    # GNN (ATARI)
    python train.py --layer $l --data $1 --graph --gpu $2 | tee output/layer/$1/gnn_$l.out

    # HTGNN
    python train.py --layer $l --data $1 --graph --hetero --dynamic --gpu $2 | tee output/layer/$1/htgnn_$l.out
done