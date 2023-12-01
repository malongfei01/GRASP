import argparse
import sys
import os
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, sort_edge_index
#from torch_geometric.data import NeighborSampler, ClusterData, ClusterLoader, Data, GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler, RandomNodeSampler
from torch_geometric.loader import NeighborSampler, ClusterData, ClusterLoader, GraphSAINTNodeSampler,GraphSAINTEdgeSampler,GraphSAINTRandomWalkSampler,RandomNodeSampler
from torch_scatter import scatter

from logger import Logger, SimpleLogger
from dataset import load_dataset, NCDataset
from data_utils import normalize, gen_normalized_adjs, evaluate, eval_acc, eval_rocauc, to_sparse_tensor
from parse import parse_method, parser_add_main_args
from batch_utils import nc_dataset_to_torch_geo, torch_geo_to_nc_dataset, AdjRowLoader, make_loader
from hyparams import hparams

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
np.random.seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
parser.add_argument('--train_batch', type=str, default='cluster', help='type of mini batch loading scheme for training GNN')
parser.add_argument('--no_mini_batch_test', action='store_true', help='whether to test on mini batches as well')
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--num_parts', type=int, default=100, help='number of partitions for partition batching')
parser.add_argument('--cluster_batch_size', type=int, default=1, help='number of clusters to use per cluster-gcn step')
parser.add_argument('--saint_num_steps', type=int, default=5, help='number of steps for graphsaint')
parser.add_argument('--test_num_parts', type=int, default=10, help='number of partitions for testing')
args = parser.parse_args()
print(args)
args_dict = vars(args)

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

### Load and preprocess data ###
dataset_ind, dataset_ood_te = load_dataset(args.dataset, args.sub_dataset)

if len(dataset_ind.y.shape) == 1:
    dataset_ind.y = dataset_ind.y.unsqueeze(1)

n = dataset_ind.num_nodes
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
d = dataset_ind.num_node_features

print(f"ind dataset {args.dataset}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.size(1)} | "
      + f"classes {c} | feats {d}")
print(f"ood te dataset {args.dataset}: all nodes {dataset_ood_te.num_nodes} | centered nodes {dataset_ood_te.node_idx.shape[0]} | edges {dataset_ood_te.edge_index.size(1)}")

# whether or not to symmetrize matters a lot!! pay attention to this
# e.g. directed edges are temporally useful in arxiv-year,
# so we usually do not symmetrize, but for label prop symmetrizing helps
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset_ind.edge_index = to_undirected(dataset_ind.edge_index)

train_loader, subgraph_loader = None, None

print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###
model = parse_method(args, dataset_ind, n, c, d, device)
model = model.to(device)


# using rocauc as the eval function
if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc
else:
    criterion = nn.NLLLoss()
    eval_func = eval_acc

logger = Logger(args.runs, args)
model_path = f'{args.dataset}-{args.sub_dataset}' if args.sub_dataset else f'{args.dataset}'
model_dir = f'models/{model_path}/{args.method}'
print(model_dir)
if not os.path.exists(model_dir) and args.method != 'lp':
    os.makedirs(model_dir)

model.train()
print('MODEL:', model)
saved_indicator = '_'.join([f'{hname}-{getattr(args, hname)}' for hname in hparams[args.dataset][args.method].keys()])


def train():
    model.train()

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        batch_train_idx = batch.mask.to(torch.bool)
        optimizer.zero_grad()
        out = model(batch)
        if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins'):
            if dataset_ind.y.shape[1] == 1:
                # change -1 instances to 0 for one-hot transform
                # dataset.label[dataset.label==-1] = 0
                true_label = F.one_hot(batch.y, batch.y.max() + 1).squeeze(1)
            else:
                true_label = batch.y

            loss = criterion(out[batch_train_idx], true_label[batch_train_idx].to(out.dtype))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[batch_train_idx], batch.y.squeeze(1)[batch_train_idx])
        total_loss += loss
        loss.backward()
        optimizer.step()
 
    return total_loss

def test():
    # needs a loader that includes every node in the graph
    model.eval()
    
    full_out = torch.zeros(n, c, device=device)
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            node_ids = batch.node_ids
            out = model(batch)
            full_out[node_ids] = out
    result = evaluate(model, dataset_ind, split_idx, eval_func, result=full_out, sampling=args.sampling, subgraph_loader=subgraph_loader)
    logger.add_result(run, result[:-1])
    return result


split_idx_lst = torch.load(f'./data/splits/{args.dataset}-splits.pt')
### Training loop ###
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train']

    print('making train loader')
    """ save_dir = f'data/minibatch/{args.dataset}/{run}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) """
    train_loader = make_loader(args, dataset_ind, train_idx, device=device)
    if not args.no_mini_batch_test:
        test_loader = make_loader(args, dataset_ind, dataset_ind.node_idx, device=device, test=True)
    else:
        test_loader = make_loader(args, dataset_ind, split_idx['test'], mini_batch = False, device=device)

    model.reset_parameters()
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float('-inf')
    for epoch in range(args.epochs):
        total_loss = train()
        result = test()

        if result[1] > best_val:
            best_logit = result[-1]
            best_out = F.log_softmax(result[-1], dim=1)
            best_val = result[1]
            best_state = copy.deepcopy(model.state_dict())

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {total_loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
            if args.print_prop:
                pred = out.argmax(dim=-1, keepdim=True)
                print("Predicted proportions:", pred.unique(return_counts=True)[1].float()/pred.shape[0])
    logger.print_statistics(run)
    torch.save(best_state, f'{model_dir}/model{run}-{saved_indicator}.pt')
    torch.save(best_logit, f'{model_dir}/logit{run}-{saved_indicator}.pt')
    torch.save(best_out, f'{model_dir}/out{run}-{saved_indicator}.pt')

### Save results ###
best_val, best_test = logger.print_statistics()
filename = f'results/{args.dataset}-{args.method}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{args.method}," + f"{saved_indicator}," +
                    f"{sub_dataset}" +
                    f"{best_val.mean():.3f} ± {best_val.std():.3f}," +
                    f"{best_test.mean():.3f} ± {best_test.std():.3f}\n")
