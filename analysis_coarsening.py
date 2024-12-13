import torch
import numpy as np

from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data.pyg_dataset import Dpr2Pyg
from torch_geometric.utils import to_undirected
from torch_geometric.utils.convert import from_scipy_sparse_matrix

import argparse
from scipy.sparse import csr_matrix
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm

from service.coarsen import coarsen_poison_mapping
from service.eval import calc_acc
from service.poison import poison

from deeprobust.graph.defense import GCN


import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cs', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self',
        choices=['Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train'], help='Mettack model variant')

parser.add_argument('--attack', type=str, default='mettack', choices=['dice', 'mettack', 'prbcd', 'nea', 'pgd'], help='attack')
parser.add_argument('--reduction', type=str, default='coarsening', 
                    choices=['coarsening', 'sparsification'], help='reduction')
parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods',
                    choices=['variation_neighborhoods_degree', 'variation_neighborhoods','variation_edges_degree','variation_edges', 'variation_cliques_degree', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'kron'],
                    help="Method of coarsening")
parser.add_argument('--sparsification_method', type=str, default='random_node_edge',
                    choices=['random_node_edge', 'random_edge', 'local_degree', 'forest_fire', 'local_similarity', 'scan', 'simmelian'],
                    help="Method of sparsification")
parser.add_argument('--device_id', type=int, default=0, help='Device ID')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

if args.dataset == 'cs':
    with open('./data/cs.pkl', 'rb') as f:
        data = pickle.load(f)
else:
    data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
if args.dataset == 'pubmed':
    adj = adj.todense()
    # convert adj and features from scipy.sparse._arrays.csr_array to csr_matrix
    for i in range(adj.shape[0]):
        if adj[i,i] != 0:
            adj[i,i] = 0
    adj = csr_matrix(adj)
    features = csr_matrix(features)

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

data = Dpr2Pyg(data).data
data = data.to(device)
data.edge_index = to_undirected(data.edge_index, data.num_nodes)

print(args)
print('adj: ', adj.shape, type(adj))
print('features: ', features.shape, type(features))
print('labels: ', labels.shape, type(labels))
print('idx_train: ', len(idx_train), type(idx_train))
print('idx_val: ', len(idx_val), type(idx_val))
print('idx_test: ', len(idx_test), type(idx_test))

perturbations = int(args.ptb_rate * (adj.sum()//2))

if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5


def main(results):

    """
    Test Clean Accuracy
    """
    perturbations = int(args.ptb_rate * (adj.sum()//2))

    # Setup Surrogate Model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
            dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val)

    modified_adj = poison(args.attack, adj, features, labels, surrogate, data, 
                          idx_train, idx_test, idx_unlabeled, 
                          perturbations, ptb_rate=args.ptb_rate, lambda_=lambda_, device=device)

    coarsening_ratio = 0.3
    coarsen_adj, coarsen_feature, coarsen_labels, coarsen_idx_train, coarsen_idx_val, node_ratio, edge_ratio, mapping = coarsen_poison_mapping(
        num_classes=(labels.max() + 1).item(),
        num_nodes=features.shape[0],
        poisoned_adj=adj,
        features=features, 
        labels=labels, 
        idx_train=idx_train, 
        idx_val=idx_val, 
        coarsening_rate=coarsening_ratio,
        method=args.coarsening_method,
    )
    poison_coarsen_adj, poison_coarsen_feature, poison_coarsen_labels, poison_coarsen_idx_train, poison_coarsen_idx_val, node_ratio, edge_ratio, poison_mapping = coarsen_poison_mapping(
        num_classes=(labels.max() + 1).item(),
        num_nodes=features.shape[0],
        poisoned_adj=modified_adj,
        features=features, 
        labels=labels, 
        idx_train=idx_train, 
        idx_val=idx_val, 
        coarsening_rate=coarsening_ratio,
        method=args.coarsening_method,
    )
    print('mapping:', len(mapping))
    print('poison_mapping:', len(poison_mapping))
    
    distance_li = []
    poison_distance_li = []
    label_diff_li = []
    poison_label_diff_li = []
    node_li = []
    for node in range(adj.shape[0]):
        if node not in poison_mapping or node not in mapping:
            distance_li.append(0)
            poison_distance_li.append(0)
            label_diff_li.append(0)
            poison_label_diff_li.append(0)
            node_li.append(node)
            continue
        node_li.append(node)
        super_node = mapping[node]
        node_feature = features[node]
        super_node_feature = coarsen_feature[super_node]
        feature_distance = np.mean(np.abs(node_feature - super_node_feature))
        distance_li.append(feature_distance)

        poison_super_node = poison_mapping[node]
        poison_super_node_feature = poison_coarsen_feature[poison_super_node]
        poison_feature_distance = np.mean(np.abs(node_feature - poison_super_node_feature))
        poison_distance_li.append(poison_feature_distance)

        if labels[node] != coarsen_labels[super_node]:
            label_diff_li.append(1)
        else:
            label_diff_li.append(0)

        if labels[node] != poison_coarsen_labels[poison_super_node]:
            poison_label_diff_li.append(1)
        else:
            poison_label_diff_li.append(0)
    print('distance:', len(distance_li))
    print('poison distance:', len(poison_distance_li))

    results['seed'] += [args.seed] * adj.shape[0]
    results['node'] += node_li
    results['clean_feature_distance'] += distance_li
    results['poison_feature_distance'] += poison_distance_li
    results['clean_label_diff'] += label_diff_li
    results['poison_label_diff'] += poison_label_diff_li


if __name__ == '__main__':
    results = {
        'seed': [],
        'node': [],
        'clean_feature_distance': [],
        'poison_feature_distance': [],
        'clean_label_diff': [],
        'poison_label_diff': [],
    }
    for seed in range(1, 6):
        print(f"=====seed: {seed}=====")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)
        args.seed = seed

        results = main(results)
    
    result_path = f'./results/{datetime.now().strftime("%b_%d")}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_name = f'{result_path}/CoarsenDistance_{args.dataset}_{args.attack}_{args.ptb_rate}_{args.coarsening_method}.csv'

    df = pd.DataFrame(results)
    df.to_csv(file_name, index=False)