#!/bin/bash

dataset=$1
sub_dataset=""

lr_lst=(0.1 0.01 0.001)

device=0

if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "pokec" ]; then 
	hidden_channels_lst=(4 8 16 32)
else
	hidden_channels_lst=(4 8 16 32 64)
fi 

for hidden_channels in "${hidden_channels_lst[@]}"; do
	for lr in "${lr_lst[@]}"; do
		if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
			python train_id.py --dataset $dataset \
			--method gcn --num_layers 2 --hidden_channels $hidden_channels \
			--lr $lr  --display_step 100 --runs 5  --device $device --epochs 500
		else
			python train_id.py --dataset $dataset \
			--method gcn --num_layers 2 --hidden_channels $hidden_channels \
			--lr $lr  --display_step 100 --runs 5  --device $device 
		fi
	done            
done