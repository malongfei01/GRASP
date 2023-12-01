#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

hidden_channels=128

batch_size_lst=(5000 10000)
device=1

for batch_size in "${batch_size_lst[@]}"; do
            if [ "$dataset" = "arxiv-year" ] || [ "$dataset" = "genius" ] || [ "$dataset" = "fb100" ]; then
                python train_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                    --method gcn --num_layers 2 --hidden_channels $hidden_channels \
                    --display_step 25 --runs 5 --train_batch graphsaint-rw --batch_size $batch_size \
                    --no_mini_batch_test --saint_num_steps 5 --device $device
            else

                python train_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                    --method gcn --num_layers 2 --hidden_channels $hidden_channels \
                    --display_step 50 --runs 5 --train_batch graphsaint-rw --batch_size $batch_size \
                    --saint_num_steps 5 --test_num_parts 10 --device $device 
            fi
done