#!/bin/bash

echo "GPU: $1"

mkdir -p output
mkdir -p output/layer

for d in setup1 setup2 setup3 setup4 setup5 setup6
do
    for l in 1 2 3 4 5 6 7 8 9 10 11 12
    do
        # GNN (ATARI)
        echo ${d}_gnn_${l} | tee -a output/layer/runtime.out
        python train.py --epoch 1 --layer $l --data $d --graph --gpu $1 | tee -a output/layer/runtime.out

        # GNN+LSTM
        echo ${d}_gnn+lstm_${l} | tee -a output/layer/runtime.out
        python train.py --epoch 1 --layer $l --data $d --graph --dynamic --gpu $1 | tee -a output/layer/runtime.out

        # HTGNN
        echo ${d}_htgnn_${l} | tee -a output/layer/runtime.out
        python train.py --epoch 1 --layer $l --data $d --graph --hetero --dynamic --gpu $1 | tee -a output/layer/runtime.out
    done
done