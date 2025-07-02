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

from service.coarsen import coarsen_poison
from service.sparsify import sparsify
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

parser.add_argument('--attack', type=str, default='mettack', choices=['dice', 'mettack', 'prbcd', 'nea', 'pgd', 'strg', 'grad'], help='attack')
parser.add_argument('--reduction', type=str, default='coarsening', 
                    choices=['coarsening', 'sparsification'], help='reduction')
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
# adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

# Setup Attack Model
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

    surrogate.eval()
    output = surrogate.predict(features, adj)
    clean_acc = accuracy(output[idx_test], labels[idx_test]).item()
    print('clean acc:', clean_acc)
    results['clean_acc'].append(clean_acc)

    modified_adj = poison(args, args.attack, adj, features, labels, surrogate, data, 
                          idx_train, idx_test, idx_unlabeled, 
                          perturbations, ptb_rate=args.ptb_rate, lambda_=lambda_, device=device)
    poison_acc = calc_acc(args, modified_adj, features, labels,
                          idx_train, idx_val, 
                          modified_adj, features, labels, idx_test, 
                          test_gnn='gcn', device=device)
    print('poison acc:', poison_acc)
    results['poison_acc'].append(poison_acc)
    evasion_acc = calc_acc(args, adj, features, labels,
                          idx_train, idx_val, 
                          modified_adj, features, labels, idx_test, 
                          test_gnn='gcn', device=device)
    print('evasion acc:', evasion_acc)
    results['evasion_acc'].append(evasion_acc)
    
    print('-'*10)
    original_edge_index = from_scipy_sparse_matrix(adj)[0]
    original_edge_index = original_edge_index.cpu().detach().numpy()
    original_edge_index = original_edge_index.T
    print('original edge index:', original_edge_index.shape)
    # convert to set
    original_edge_index = [tuple(edge) for edge in original_edge_index]
    original_edge_index = set(original_edge_index)

    modified_edge_index = from_scipy_sparse_matrix(modified_adj)[0]
    modified_edge_index = modified_edge_index.cpu().detach().numpy()
    modified_edge_index = modified_edge_index.T
    print('modified edge index:', modified_edge_index.shape)
    modified_edge_index = [tuple(edge) for edge in modified_edge_index]
    modified_edge_index = set(modified_edge_index)

    intersect = modified_edge_index.intersection(original_edge_index)
    add_edges = modified_edge_index - original_edge_index
    remove_edges = original_edge_index - modified_edge_index
    print('add edges:', len(add_edges))
    print('remove edges:', len(remove_edges))
    results['add_edges'].append(len(add_edges))
    results['remove_edges'].append(len(remove_edges))
    
    if len(add_edges) != 0:
        add_edges = list(add_edges)
        add_edges = np.array(add_edges).T
        add_edges = torch.from_numpy(add_edges)

        adj_add = torch.zeros((data.num_nodes, data.num_nodes))
        adj_add[add_edges[0], add_edges[1]] = 1
        adj_add[add_edges[1], add_edges[0]] = 1

        adj_add = csr_matrix(adj_add.cpu().detach().numpy())

        reduced_adj = modified_adj - adj_add


        reduced_acc = calc_acc(args, 
                               reduced_adj, features, labels, 
                               idx_train, idx_val, 
                               modified_adj, features, labels, idx_test, 
                               test_gnn='gcn', device=device)
        print('remove only acc:', reduced_acc)
        results['remove_only_acc'].append(reduced_acc)
    else:
        print('remove only acc:', -1)
        results['remove_only_acc'].append(-1)
    if len(remove_edges) != 0:
        remove_edges = list(remove_edges)
        remove_edges = np.array(remove_edges).T
        remove_edges = torch.from_numpy(remove_edges)

        adj_remove = torch.zeros((data.num_nodes, data.num_nodes))
        adj_remove[remove_edges[0], remove_edges[1]] = 1
        adj_remove[remove_edges[1], remove_edges[0]] = 1

        adj_remove = csr_matrix(adj_remove.cpu().detach().numpy())

        reduced_adj = modified_adj + adj_remove
        reduced_acc = calc_acc(args, 
                               reduced_adj, features, labels, 
                               idx_train, idx_val, 
                               modified_adj, features, labels, idx_test, 
                               test_gnn='gcn', device=device)
        print('add only acc:', reduced_acc)
        results['add_only_acc'].append(reduced_acc)
    else:
        print('add only acc:', -1)
        results['add_only_acc'].append(-1)

    return results


if __name__ == '__main__':
    results = {
        'clean_acc': [],
        'poison_acc': [],
        'remove_only_acc': [],
        'add_only_acc': [],
        'evasion_acc': [],
        'add_edges': [],
        'remove_edges': [],
    }

    for seed in range(1, 6):
        print(f"=====seed: {seed}=====")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)

        results = main(results)
    
    result_path = f'./results/{datetime.now().strftime("%b_%d")}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_name = f'{result_path}/AddRemoveEvasion_{args.dataset}_{args.attack}_{args.ptb_rate}.csv'

    df = pd.DataFrame(results)
    df.to_csv(file_name, index=False)