#!/bin/bash

device=0

# cora with different OOD methods
# GKDE
python train_ood.py --dataset cora --method gcn --ood SGCN  --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset cora --method gcn --ood SGCN  --GPN_detect_type Epist --device $device --runs 5
# GPN
python train_ood.py --dataset cora --ood GPN --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset cora --ood GPN --GPN_detect_type Epist --device $device --runs 5
python train_ood.py --dataset cora --ood GPN --GPN_detect_type Epist_wo_Net --device $device --runs 5
# OODGAT
python train_oodgat.py --dataset cora --ood OODGAT  --OODGAT_detect_type ATT --device $device --runs 5
python train_oodgat.py --dataset cora --ood OODGAT  --OODGAT_detect_type ENT --device $device --runs 5

# amazon-photo with different OOD methods
# GKDE
python train_ood.py --dataset amazon-photo --method gcn --ood SGCN  --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset amazon-photo --method gcn --ood SGCN  --GPN_detect_type Epist --device $device --runs 5
# GPN
python train_ood.py --dataset amazon-photo --ood GPN --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset amazon-photo --ood GPN --GPN_detect_type Epist --device $device --runs 5
python train_ood.py --dataset amazon-photo --ood GPN --GPN_detect_type Epist_wo_Net --device $device --runs 5
# OODGAT
python train_oodgat.py --dataset amazon-photo --ood OODGAT  --OODGAT_detect_type ATT --device $device --runs 5
python train_oodgat.py --dataset amazon-photo --ood OODGAT  --OODGAT_detect_type ENT --device $device --runs 5

# coauthor-cs with different OOD methods
# GKDE
python train_ood.py --dataset coauthor-cs --method gcn --ood SGCN  --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset coauthor-cs --method gcn --ood SGCN  --GPN_detect_type Epist --device $device --runs 5
# GPN
python train_ood.py --dataset coauthor-cs --ood GPN --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset coauthor-cs --ood GPN --GPN_detect_type Epist --device $device --runs 5
python train_ood.py --dataset coauthor-cs --ood GPN --GPN_detect_type Epist_wo_Net --device $device --runs 5
# OODGAT
python train_oodgat.py --dataset coauthor-cs --ood OODGAT  --OODGAT_detect_type ATT --device $device --runs 5
python train_oodgat.py --dataset coauthor-cs --ood OODGAT  --OODGAT_detect_type ENT --device $device --runs 5

# chameleon with different OOD methods
# GKDE
python train_ood.py --dataset chameleon --method gcn --ood SGCN  --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset chameleon --method gcn --ood SGCN  --GPN_detect_type Epist --device $device --runs 5
# GPN
python train_ood.py --dataset chameleon --ood GPN --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset chameleon --ood GPN --GPN_detect_type Epist --device $device --runs 5
python train_ood.py --dataset chameleon --ood GPN --GPN_detect_type Epist_wo_Net --device $device --runs 5
# OODGAT
python train_oodgat.py --dataset chameleon --ood OODGAT  --OODGAT_detect_type ATT --device $device --runs 5
python train_oodgat.py --dataset chameleon --ood OODGAT  --OODGAT_detect_type ENT --device $device --runs 5

# squirrel with different OOD methods
# GKDE
python train_ood.py --dataset squirrel --method gcn --ood SGCN  --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset squirrel --method gcn --ood SGCN  --GPN_detect_type Epist --device $device --runs 5
# GPN
python train_ood.py --dataset squirrel --ood GPN --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset squirrel --ood GPN --GPN_detect_type Epist --device $device --runs 5
python train_ood.py --dataset squirrel --ood GPN --GPN_detect_type Epist_wo_Net --device $device --runs 5
# OODGAT
python train_oodgat.py --dataset squirrel --ood OODGAT  --OODGAT_detect_type ATT --device $device --runs 5
python train_oodgat.py --dataset squirrel --ood OODGAT  --OODGAT_detect_type ENT --device $device --runs 5

# arxiv-year with different OOD methods
# GKDE
python train_ood.py --dataset arxiv-year --method gcn --ood SGCN  --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset arxiv-year --method gcn --ood SGCN  --GPN_detect_type Epist --device $device --runs 5
# GPN
python train_ood.py --dataset arxiv-year --ood GPN --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset arxiv-year --ood GPN --GPN_detect_type Epist --device $device --runs 5
python train_ood.py --dataset arxiv-year --ood GPN --GPN_detect_type Epist_wo_Net --device $device --runs 5
# OODGAT
python train_oodgat.py --dataset arxiv-year --ood OODGAT  --OODGAT_detect_type ATT --device $device --runs 5
python train_oodgat.py --dataset arxiv-year --ood OODGAT  --OODGAT_detect_type ENT --device $device --runs 5

# snap-patents with different OOD methods
# GKDE
python train_ood.py --dataset snap-patents --method gcn --ood SGCN  --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset snap-patents --method gcn --ood SGCN  --GPN_detect_type Epist --device $device --runs 5
# GPN
python train_ood.py --dataset snap-patents --ood GPN --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset snap-patents --ood GPN --GPN_detect_type Epist --device $device --runs 5
python train_ood.py --dataset snap-patents --ood GPN --GPN_detect_type Epist_wo_Net --device $device --runs 5
# OODGAT
python train_oodgat.py --dataset snap-patents --ood OODGAT  --OODGAT_detect_type ATT --device $device --runs 5
python train_oodgat.py --dataset snap-patents --ood OODGAT  --OODGAT_detect_type ENT --device $device --runs 5

# wiki with different OOD methods
# GKDE
python train_ood.py --dataset wiki --method gcn --ood SGCN  --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset wiki --method gcn --ood SGCN  --GPN_detect_type Epist --device $device --runs 5
# GPN
python train_ood.py --dataset wiki --ood GPN --GPN_detect_type Alea --device $device --runs 5
python train_ood.py --dataset wiki --ood GPN --GPN_detect_type Epist --device $device --runs 5
python train_ood.py --dataset wiki --ood GPN --GPN_detect_type Epist_wo_Net --device $device --runs 5
# OODGAT
python train_oodgat.py --dataset wiki --ood OODGAT  --OODGAT_detect_type ATT --device $device --runs 5
python train_oodgat.py --dataset wiki --ood OODGAT  --OODGAT_detect_type ENT --device $device --runs 5
