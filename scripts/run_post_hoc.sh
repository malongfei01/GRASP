#!/bin/bash

device=0

# cora with different OOD methods
# post-hoc methods
for backbone in "gcn" "h2gcn";do
	python test_ood.py --dataset cora --device $device --method $backbone --ood MSP --runs 5
	python test_ood.py --dataset cora --device $device --method $backbone --ood MSP --runs 5 --prop
	python test_ood.py --dataset cora --device $device --method $backbone --ood MSP --runs 5 --grasp
	python test_ood.py --dataset cora --device $device --method $backbone --ood Energy --runs 5
	python test_ood.py --dataset cora --device $device --method $backbone --ood Energy --runs 5 --prop
	python test_ood.py --dataset cora --device $device --method $backbone --ood Energy --runs 5 --grasp
	python test_ood.py --dataset cora --device $device --method $backbone --ood ODIN --runs 5
	python test_ood.py --dataset cora --device $device --method $backbone --ood ODIN --runs 5 --prop
	python test_ood.py --dataset cora --device $device --method $backbone --ood ODIN --runs 5 --grasp
	python test_ood.py --dataset cora --device $device --method $backbone --ood Mahalanobis --runs 5
	python test_ood.py --dataset cora --device $device --method $backbone --ood Mahalanobis --runs 5 --prop
	python test_ood.py --dataset cora --device $device --method $backbone --ood Mahalanobis --runs 5 --grasp
	python test_ood.py --dataset cora --device $device --method $backbone --ood KNN --runs 5
	python test_ood.py --dataset cora --device $device --method $backbone --ood KNN --runs 5 --prop
	python test_ood.py --dataset cora --device $device --method $backbone --ood KNN --runs 5 --grasp
	python test_ood.py --dataset cora --device $device --method $backbone --ood GNNSafe --runs 5
	python test_ood.py --dataset cora --device $device --method $backbone --ood GRASP --delta 1.02 --runs 5
done

# amazon-photo with different OOD methods
# post-hoc methods
for backbone in "gcn" "h2gcn";do
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood MSP --runs 5
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood MSP --runs 5 --prop
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood MSP --runs 5 --grasp
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood Energy --runs 5
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood Energy --runs 5 --prop
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood Energy --runs 5 --grasp
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood ODIN --runs 5
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood ODIN --runs 5 --prop
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood ODIN --runs 5 --grasp
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood Mahalanobis --runs 5
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood Mahalanobis --runs 5 --prop
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood Mahalanobis --runs 5 --grasp
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood KNN --runs 5
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood KNN --runs 5 --prop
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood KNN --runs 5 --grasp
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood GNNSafe --runs 5
	python test_ood.py --dataset amazon-photo --device $device --method $backbone --ood GRASP --delta 1.02 --runs 5
done


# coauthor-cs with different OOD methods
# post-hoc methods
for backbone in "gcn" "h2gcn";do
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood MSP --runs 5
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood MSP --runs 5 --prop
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood MSP --runs 5 --grasp
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood Energy --runs 5
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood Energy --runs 5 --prop
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood Energy --runs 5 --grasp
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood ODIN --runs 5
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood ODIN --runs 5 --prop
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood ODIN --runs 5 --grasp
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood Mahalanobis --runs 5
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood Mahalanobis --runs 5 --prop
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood Mahalanobis --runs 5 --grasp
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood KNN --runs 5
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood KNN --runs 5 --prop
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood KNN --runs 5 --grasp
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood GNNSafe --runs 5
	python test_ood.py --dataset coauthor-cs --device $device --method $backbone --ood GRASP --delta 1.02 --runs 5
done


# chameleon with different OOD methods
# post-hoc methods
for backbone in "gcn" "h2gcn";do
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood MSP --runs 5
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood MSP --runs 5 --prop
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood MSP --runs 5 --grasp
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood Energy --runs 5
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood Energy --runs 5 --prop
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood Energy --runs 5 --grasp
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood ODIN --runs 5
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood ODIN --runs 5 --prop
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood ODIN --runs 5 --grasp
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood Mahalanobis --runs 5
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood Mahalanobis --runs 5 --prop
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood Mahalanobis --runs 5 --grasp
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood KNN --runs 5
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood KNN --runs 5 --prop
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood KNN --runs 5 --grasp
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood GNNSafe --runs 5
	python test_ood.py --dataset chameleon --device $device --method $backbone --ood GRASP --delta 1.2 --runs 5
done


# squirrel with different OOD methods
# post-hoc methods
for backbone in "gcn" "h2gcn";do
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood MSP --runs 5
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood MSP --runs 5 --prop
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood MSP --runs 5 --grasp
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood Energy --runs 5
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood Energy --runs 5 --prop
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood Energy --runs 5 --grasp
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood ODIN --runs 5
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood ODIN --runs 5 --prop
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood ODIN --runs 5 --grasp
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood Mahalanobis --runs 5
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood Mahalanobis --runs 5 --prop
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood Mahalanobis --runs 5 --grasp
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood KNN --runs 5
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood KNN --runs 5 --prop
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood KNN --runs 5 --grasp
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood GNNSafe --runs 5
	python test_ood.py --dataset squirrel --device $device --method $backbone --ood GRASP --delta 1.2 --runs 5
done


# arxiv-year with different OOD methods
# post-hoc methods
for backbone in "gcn" "h2gcn";do
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood MSP --runs 5
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood MSP --runs 5 --prop
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood MSP --runs 5 --grasp
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood Energy --runs 5
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood Energy --runs 5 --prop
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood Energy --runs 5 --grasp
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood ODIN --runs 5
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood ODIN --runs 5 --prop
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood ODIN --runs 5 --grasp
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood Mahalanobis --runs 5
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood Mahalanobis --runs 5 --prop
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood Mahalanobis --runs 5 --grasp
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood KNN --runs 5
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood KNN --runs 5 --prop
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood KNN --runs 5 --grasp
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood GNNSafe --runs 5
	python test_ood.py --dataset arxiv-year --device $device --method $backbone --ood GRASP --delta 1.2 --runs 5
done

# snap-patents with different OOD methods
# post-hoc methods
for backbone in "gcn" "h2gcn";do
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood MSP --runs 5
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood MSP --runs 5 --prop
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood MSP --runs 5 --grasp
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood Energy --runs 5
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood Energy --runs 5 --prop
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood Energy --runs 5 --grasp
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood ODIN --runs 5
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood ODIN --runs 5 --prop
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood ODIN --runs 5 --grasp
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood Mahalanobis --runs 5
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood Mahalanobis --runs 5 --prop
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood Mahalanobis --runs 5 --grasp
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood KNN --runs 5
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood KNN --runs 5 --prop
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood KNN --runs 5 --grasp
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood GNNSafe --runs 5
	python test_ood.py --dataset snap-patents --device $device --method $backbone --ood GRASP --delta 1.2 --runs 5
done


# wiki with different OOD methods
# post-hoc methods
for backbone in "gcn" "h2gcn";do
	python test_ood.py --dataset wiki --device $device --method $backbone --ood MSP --runs 5
	python test_ood.py --dataset wiki --device $device --method $backbone --ood MSP --runs 5 --prop
	python test_ood.py --dataset wiki --device $device --method $backbone --ood MSP --runs 5 --grasp
	python test_ood.py --dataset wiki --device $device --method $backbone --ood Energy --runs 5
	python test_ood.py --dataset wiki --device $device --method $backbone --ood Energy --runs 5 --prop
	python test_ood.py --dataset wiki --device $device --method $backbone --ood Energy --runs 5 --grasp
	python test_ood.py --dataset wiki --device $device --method $backbone --ood ODIN --runs 5
	python test_ood.py --dataset wiki --device $device --method $backbone --ood ODIN --runs 5 --prop
	python test_ood.py --dataset wiki --device $device --method $backbone --ood ODIN --runs 5 --grasp
	python test_ood.py --dataset wiki --device $device --method $backbone --ood Mahalanobis --runs 5
	python test_ood.py --dataset wiki --device $device --method $backbone --ood Mahalanobis --runs 5 --prop
	python test_ood.py --dataset wiki --device $device --method $backbone --ood Mahalanobis --runs 5 --grasp
	python test_ood.py --dataset wiki --device $device --method $backbone --ood KNN --runs 5
	python test_ood.py --dataset wiki --device $device --method $backbone --ood KNN --runs 5 --prop
	python test_ood.py --dataset wiki --device $device --method $backbone --ood KNN --runs 5 --grasp
	python test_ood.py --dataset wiki --device $device --method $backbone --ood GNNSafe --runs 5
	python test_ood.py --dataset wiki --device $device --method $backbone --ood GRASP --delta 1.2 --runs 5
done