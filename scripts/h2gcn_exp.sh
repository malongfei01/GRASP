#!/bin/bash

dataset=$1
sub_dataset=${2:-'None'}

hidden_channels_lst=(8 16 32 64)

num_layers_lst=(1 2)

dropout_lst=(0 .5)

device=4

for hidden_channels in "${hidden_channels_lst[@]}"; do
	for num_layers in "${num_layers_lst[@]}"; do
		for dropout in "${dropout_lst[@]}"; do
			if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
				python train_id.py --dataset $dataset --device $device \
				--method h2gcn --num_layers $num_layers --hidden_channels $hidden_channels \
				--display_step 100 --runs 5 --dropout $dropout  --epochs 500
			else
				python train_id.py --dataset $dataset --device $device \
				--method h2gcn --num_layers $num_layers --hidden_channels $hidden_channels \
				--display_step 100 --runs 5 --dropout $dropout 
			fi
		done
	done            
done