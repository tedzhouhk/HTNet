#!/bin/bash

echo "GPU: $1"

for d in setup1 setup2 setup3 setup4 setup5 setup6
do
    # SINR
    echo ${d}_sinr | tee -a output/runtime.out
    python train.py --data $d --sinr | tee -a output/runtime.out

    # gradient boosted tree
    echo ${d}_gbrt | tee -a output/runtime.out
    python train.py --data $d --gbrt | tee -a output/runtime.out

    # MLP
    echo ${d}_mlp | tee -a output/runtime.out
    python train.py --data $d --gpu $1 --epoch 1 | tee -a output/runtime.out

    # GNN (ATARI)
    echo ${d}_gnn | tee -a output/runtime.out
    python train.py --data $d --graph --gpu $1 --epoch 1 | tee -a output/runtime.out

    # MLP+LSTM
    echo ${d}_mlp+lstm | tee -a output/runtime.out
    python train.py --data $d --dynamic --gpu $1 --epoch 1 | tee -a output/runtime.out

    # GNN+LSTM
    echo ${d}_gnn+lstm | tee -a output/runtime.out
    python train.py --data $d --graph --dynamic --gpu $1 --epoch 1 | tee -a output/runtime.out

    # HTGNN
    echo ${d}_htgnn | tee -a output/runtime.out
    python train.py --data $d --graph --hetero --dynamic --gpu $1 --epoch 1 | tee -a output/runtime.out
done