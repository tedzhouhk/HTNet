#!/bin/bash

echo "Datset: $1"
echo "GPU: $2"

mkdir -p output
mkdir -p output/main

# SINR
python train.py --data $1 --sinr --output main/$1_sinr | tee output/main/$1_sinr.out

# gradient boosted tree
python train.py --data $1 --gbrt --output main/$1_gbrt | tee output/main/$1_gbrt.out

# MLP
python train.py --data $1 --output main/$1_mlp --gpu $2 | tee output/main/$1_mlp.out

# GNN (ATARI)
python train.py --data $1 --output main/$1_gnn --graph --gpu $2 | tee output/main/$1_gnn.out

# MLP+LSTM
python train.py --data $1 --output main/$1_mlp+lstm --dynamic --gpu $2 | tee output/main/$1_mlp+lstm.out

# GNN+LSTM
python train.py --data $1 --output main/$1_gnn+lstm --graph --dynamic --gpu $2 | tee output/main/$1_gnn+lstm.out

# HTGNN
python train.py --data $1 --output main/$1_htgnn --graph --hetero --dynamic --gpu $2 | tee output/main/$1_htgnn.out