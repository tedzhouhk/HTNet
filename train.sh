#!/bin/bash

echo "Datset: $1"
echo "GPU: $2"

mkdir -p output

# SINR
python train.py --data $1 --sinr --output $1_sinr | tee output/$1_sinr.out

# # gradient boosted tree
# python train.py --data $1 --gbrt --output $1_gbrt | tee output/$1_gbrt.out

# # MLP
# python train.py --data $1 --output $1_mlp --gpu $2 | tee output/$1_mlp.out

# # GNN (ATARI)
# python train.py --data $1 --output $1_gnn --graph --gpu $2 | tee output/$1_gnn.out

# # MLP+LSTM
# python train.py --data $1 --output $1_mlp+lstm --dynamic --gpu $2 | tee output/$1_mlp+lstm.out

# # GNN+LSTM
# python train.py --data $1 --output $1_gnn+lstm --graph --dynamic --gpu $2 | tee output/$1_gnn+lstm.out

# # HTGNN
# python train.py --data $1 --output $1_htgnn --graph --hetero --dynamic --gpu $2 | tee output/$1_htgnn.out