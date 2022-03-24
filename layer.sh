#!/bin/bash

echo "Datset: $1"
echo "GPU: $2"

mkdir -p output
mkdir -p output/layer
mkdir -p output/layer/$1

for l in 1 2 3 4 5 6 7 8 9 10 11 12
do
    if [ $(jobs -r | wc -l) -ge 3 ]; 
    then
        wait -n
    fi
    # GNN (ATARI)
    (python train.py --layer $l --data $1 --graph --gpu $2 | /usr/bin/tee output/layer/$1/gnn_$l.out) &


    if [ $(jobs -r | wc -l) -ge 3 ]; 
    then
        wait -n
    fi
    # GNN+LSTM
    (python train.py --layer $l --data $1 --graph --dynamic --gpu $2 | /usr/bin/tee -a output/layer/runtime.out) &

    if [ $(jobs -r | wc -l) -ge 3 ]; 
    then
        wait -n
    fi
    # HTGNN
    (python train.py --layer $l --data $1 --graph --hetero --dynamic --gpu $2 | /usr/bin/tee output/layer/$1/htgnn_$l.out) &
done

wait 