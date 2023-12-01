from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
import pickle
import pandas as pd
from sklearn.preprocessing import label_binarize
import gdown
from os import path
import os
import torch_geometric.transforms as T

from load_data import load_twitch, load_fb100, load_twitch_gamer, DATAPATH
from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url

from homophily import our_measure, edge_homophily_edge_idx

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikipediaNetwork, Actor, WebKB
from torch_geometric.transforms import NormalizeFeatures
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from ogb.nodeproppred import NodePropPredDataset


class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None
        self.node_idx = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(dataname, sub_dataname=''):
    """ Loader for NCDataset, returns NCDataset. """
    if dataname == 'twitch-e':
        # twitch-explicit graph
        if sub_dataname not in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'):
            print('Invalid sub_dataname, deferring to DE graph')
            sub_dataname = 'DE'
        dataset = load_twitch_dataset(sub_dataname)
    elif dataname == 'fb100':
        if sub_dataname not in ('Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Reed98'):
            print('Invalid sub_dataname, deferring to Penn94 graph')
            sub_dataname = 'Penn94'
        dataset = load_fb100_dataset(sub_dataname)
    elif dataname == 'ogbn-proteins':
        dataset = load_proteins_dataset()
    elif dataname == 'deezer-europe':
        dataset = load_deezer_dataset()
    elif dataname == 'arxiv-year':
        dataset_ind, dataset_ood_te = load_arxiv_year_dataset()
    elif dataname == 'pokec':
        dataset = load_pokec_mat()
    elif dataname == 'snap-patents':
        dataset_ind, dataset_ood_te = load_snap_patents_mat()
    elif dataname == 'yelp-chi':
        dataset = load_yelpchi_dataset()
    elif dataname in ('ogbn-arxiv', 'ogbn-products'):
        dataset = load_ogb_dataset(dataname)
    elif dataname in ('Cora', 'CiteSeer', 'PubMed'):
        dataset = load_planetoid_dataset(dataname)
    elif dataname in ('chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin'):
        dataset_ind, dataset_ood_te = load_geom_gcn_dataset(dataname)
    elif dataname == "genius":
        dataset = load_genius()
    elif dataname == "twitch-gamer":
        dataset = load_twitch_gamer_dataset() 
    elif dataname == "wiki":
        dataset_ind, dataset_ood_te = load_wiki()
    elif dataname in ('cora', 'amazon-photo', 'coauthor-cs'):
        dataset_ind, dataset_ood_te = load_graph_dataset(dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset_ind, dataset_ood_te

def load_graph_dataset(dataname):
    transform = T.NormalizeFeatures()
    if dataname in ('cora', 'citeseer', 'pubmed'):
        torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', split='public',
                              name=dataname, transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'amazon-photo':
        torch_dataset = Amazon(root=f'{DATAPATH}Amazon',
                               name='Photo', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'amazon-computer':
        torch_dataset = Amazon(root=f'{DATAPATH}Amazon',
                               name='Computers', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{DATAPATH}Coauthor',
                                 name='CS', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{DATAPATH}Coauthor',
                                 name='Physics', transform=transform)
        dataset = torch_dataset[0]
    else:
        raise NotImplementedError

    
    dataset.node_idx = torch.arange(dataset.num_nodes)
    label = dataset.y

    if dataname == 'cora':
        class_t = 4
    elif dataname == 'amazon-photo':
        class_t = 5
    elif dataname == 'coauthor-cs':
        class_t = 5
    y = label - class_t
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=dataset.x, edge_index=dataset.edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[y>=0]
    dataset_ood_te = Data(x=dataset.x, edge_index=dataset.edge_index, y=y)
    dataset_ood_te.node_idx = idx[y<0]
    return dataset_ind, dataset_ood_te


def load_twitch_dataset(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    A, label, features = load_twitch(lang)
    dataset = NCDataset(lang)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_fb100_dataset(filename):
    A, metadata = load_fb100(filename)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_deezer_dataset():
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{DATAPATH}deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_arxiv_year_dataset(nclass=5):
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root="data")
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = even_quantile_labels(
        ogb_dataset.graph['node_year'].flatten(), nclass, verbose=False)
    label = torch.as_tensor(label)
    y = label - 2
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[y>=0]
    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[y<0]
    return dataset_ind, dataset_ood_te

    """ dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[label>=2]
    
    dataset_ood_tr = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_tr.node_idx = idx[label==1]

    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[label==0]

    return dataset_ind, dataset_ood_tr, dataset_ood_te """


def load_proteins_dataset():
    ogb_dataset = NodePropPredDataset(name='ogbn-proteins')
    dataset = NCDataset('ogbn-proteins')

    def protein_orig_split(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        return {'train': torch.as_tensor(split_idx['train']),
                'valid': torch.as_tensor(split_idx['valid']),
                'test': torch.as_tensor(split_idx['test'])}

    dataset.get_idx_split = protein_orig_split
    dataset.graph, dataset.label = ogb_dataset.graph, ogb_dataset.labels

    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['edge_feat'] = torch.as_tensor(dataset.graph['edge_feat'])
    dataset.label = torch.as_tensor(dataset.label)
    return dataset


def load_ogb_dataset(name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name)
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx

    dataset.get_idx_split = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset


def load_pokec_mat():
    """ requires pokec.mat
    """
    if not path.exists(f'{DATAPATH}pokec.mat'):
        gdown.download(id=dataset_drive_url['pokec'], \
            output=f'{DATAPATH}pokec.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def load_snap_patents_mat(nclass=5):
    if not path.exists(f'{DATAPATH}snap_patents.mat'):
        p = dataset_drive_url['snap-patents']
        print(f"Snap patents url: {p}")
        gdown.download(id=dataset_drive_url['snap-patents'], \
            output=f'{DATAPATH}snap_patents.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}snap_patents.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(
        fulldata['node_feat'].todense(), dtype=torch.float)
    
    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    label = torch.tensor(label, dtype=torch.long)
    y = label - 2
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[y>=0]
    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[y<0]
    return dataset_ind, dataset_ood_te
    """ dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[label>=2]
    
    dataset_ood_tr = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_tr.node_idx = idx[label==1]

    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[label==0]

    return dataset_ind, dataset_ood_tr, dataset_ood_te """


def load_yelpchi_dataset():
    if not path.exists(f'{DATAPATH}YelpChi.mat'):
        gdown.download(id=dataset_drive_url['yelp-chi'], \
            output=f'{DATAPATH}YelpChi.mat', quiet=False)
    fulldata = scipy.io.loadmat(f'{DATAPATH}YelpChi.mat')
    A = fulldata['homo']
    edge_index = np.array(A.nonzero())
    node_feat = fulldata['features']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset('YelpChi')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset


def load_planetoid_dataset(name):
    torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid',
                              name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    def planetoid_orig_split(**kwargs):
        return {'train': torch.as_tensor(dataset.train_idx),
                'valid': torch.as_tensor(dataset.valid_idx),
                'test': torch.as_tensor(dataset.test_idx)}

    dataset.get_idx_split = planetoid_orig_split
    dataset.label = label

    return dataset


def load_geom_gcn_dataset(name):
    fulldata = scipy.io.loadmat(f'{DATAPATH}/{name}.mat')
    edge_index = torch.as_tensor(fulldata['edge_index'], dtype=torch.long)
    edge_index = remove_self_loops(edge_index)[0]
    node_feat = torch.as_tensor(fulldata['node_feat'])
    label = np.array(fulldata['label'], dtype=int).flatten()
    label = torch.as_tensor(label)
    y = label - 2
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[y>=0]
    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[y<0]
    return dataset_ind, dataset_ood_te

    """ torch_dataset = torch.load(f'{DATAPATH}/{name}.pt')
    dataset = torch_dataset[0]
    dataset.node_idx = torch.arange(dataset.num_nodes)
    label = dataset.y
    y = label - 2
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=dataset.x, edge_index=dataset.edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[y>=0]
    dataset_ood_te = Data(x=dataset.x, edge_index=dataset.edge_index, y=y)
    dataset_ood_te.node_idx = idx[y<0]
    return dataset_ind, dataset_ood_te """

    """ y = label - 2
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[label>=2]
    
    dataset_ood_tr = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_tr.node_idx = idx[label==1]

    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[label==0]

    return dataset_ind, dataset_ood_tr, dataset_ood_te """


def load_genius():
    filename = 'genius'
    dataset = NCDataset(filename)
    fulldata = scipy.io.loadmat(f'data/genius.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_twitch_gamer_dataset(task="mature", normalize=True):
    if not path.exists(f'{DATAPATH}twitch-gamer_feat.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_feat'],
            output=f'{DATAPATH}twitch-gamer_feat.csv', quiet=False)
    if not path.exists(f'{DATAPATH}twitch-gamer_edges.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_edges'],
            output=f'{DATAPATH}twitch-gamer_edges.csv', quiet=False)
    
    edges = pd.read_csv(f'{DATAPATH}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{DATAPATH}twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    dataset = NCDataset("twitch-gamer")
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_wiki():

    if not path.exists(f'{DATAPATH}wiki_features2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_features'], \
            output=f'{DATAPATH}wiki_features2M.pt', quiet=False)
    
    if not path.exists(f'{DATAPATH}wiki_edges2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_edges'], \
            output=f'{DATAPATH}wiki_edges2M.pt', quiet=False)

    if not path.exists(f'{DATAPATH}wiki_views2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_views'], \
            output=f'{DATAPATH}wiki_views2M.pt', quiet=False)


    node_feat = torch.load(f'{DATAPATH}wiki_features2M.pt')
    edge_index = torch.load(f'{DATAPATH}wiki_edges2M.pt').T
    #print(f"edges shape: {edge_index.shape}")
    label = torch.load(f'{DATAPATH}wiki_views2M.pt') 
    #print(f"features shape: {node_feat.shape[1]}")
    #print(f"Label shape: {label.shape[0]}")

    y = label - 2
    y = torch.where(y<0, -1, y)
    dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[y>=0]
    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[y<0]
    return dataset_ind, dataset_ood_te


    """ dataset_ind = Data(x=node_feat, edge_index=edge_index, y=y)
    idx = torch.arange(label.size(0))
    dataset_ind.node_idx = idx[label>=2]
    
    dataset_ood_tr = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_tr.node_idx = idx[label==1]

    dataset_ood_te = Data(x=node_feat, edge_index=edge_index, y=y)
    dataset_ood_te.node_idx = idx[label==0]

    return dataset_ind, dataset_ood_tr, dataset_ood_te """

