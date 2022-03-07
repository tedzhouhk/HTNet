#!/bin/bash

echo "Datset: $1"
echo "GPU: $2"

# gradient boosted tree
python train.py --data $1 --gbrt --output $1_gbrt | tee output/$1_gbrt.out

# MLP
python train.py --data $1 --output $1_mlp --gpu $2 | tee output/$1_mlp.out

# GNN (ATARI)
python train.py --data $1 --output $1_mlp --graph --gpu $2 | tee output/$1_gnn.out

# MLP+LSTM
python train.py --data $1 --output $1_mlp --dynamic --gpu $2 | tee output/$1_mlp+lstm.out

# GNN+LSTM
python train.py --data $1 --output $1_mlp --graph --dynamic --gpu $2 | tee output/$1_gnn+lstm.out

# HTGNN
python train.py --data $1 --output $1_mlp --graph --hetero --dynamic --gpu $2 | tee output/$1_htgnn.out