import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from injection.AGIA import AGIA
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, Dpr2Pyg
import argparse
from scipy.sparse import csr_matrix
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm

from torch_geometric.utils.convert import from_scipy_sparse_matrix
from service.coarsen import coarsen_poison
from service.sparsify import sparsify
from service.train import train_gnn
from service.eval import eval_gnn

from deeprobust.graph.defense import RGCN
from model.gnn_guard import GNNGuard
from model.median_gcn import MedianGCN
from model.sage import GraphSage
from model.gat import GAT
from model.strg import STRG

import torch_sparse

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
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cs', 'cora', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
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



# adj_tensor = torch.Tensor(adj.todense()).to(device)
# features_tensor = torch.FloatTensor(features.todense()).to(device)
# labels_tensor = torch.LongTensor(labels).to(device)

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

print(args)
print('adj: ', adj.shape, type(adj))
print('features: ', features.shape, type(features))
print('labels: ', labels.shape, type(labels))
print('idx_train: ', len(idx_train), type(idx_train))
print('idx_val: ', len(idx_val), type(idx_val))
print('idx_test: ', len(idx_test), type(idx_test))

perturbations = int(args.ptb_rate * (adj.sum()//2))
# Setup Surrogate Model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val)

def main(result):
    attacker = AGIA(epsilon=0.01,
                 n_epoch=500,
                 a_epoch=300,
                 n_inject_max= 60,
                 n_edge_max= 20,
                 feat_lim_min=-1,
                 feat_lim_max=1,
                 device=device,
    )
    # model = model.to(device)
    modified_adj, modified_features = attacker.attack(
        model=surrogate,
        adj=torch.Tensor(adj.todense()).to(device),
        features=torch.FloatTensor(features.todense()).to(device),
        target_idx=torch.LongTensor(idx_test).to(device),
        labels=None
    )
    modified_adj = modified_adj.to_dense().detach().cpu().numpy()
    modified_adj = csr_matrix(modified_adj)

    modified_features = torch.cat([torch.FloatTensor(features.todense()).to(device), modified_features], dim=0)

    modified_features = modified_features.detach().cpu().numpy()
    modified_features = csr_matrix(modified_features)

    print('modified_adj: ', modified_adj.shape, type(modified_adj))
    print('modified_features: ', modified_features.shape, type(modified_features))

    reduce_ratio_li = np.linspace(0.1, 1.0, 5)
    for test_gnn in ['gcn', 'gat', 'sage', 'gnnguard', 'median', 'strg']:
        clean_gnn = train_gnn(adj, features, labels, idx_train, idx_val, args, test_gnn=test_gnn, device=device)
        clean_acc = eval_gnn(clean_gnn, adj, features, labels, idx_test, args, test_gnn=test_gnn, device=device)
        print(f'clean_gnn: {test_gnn}, clean_acc: {clean_acc}')
        
        poison_acc = eval_gnn(clean_gnn, modified_adj, modified_features, labels, idx_test, args, test_gnn=test_gnn, device=device)
        # check = eval_gnn(surrogate, modified_adj, modified_features, labels, idx_test, args, test_gnn=test_gnn, device=device)
        print(f'poison_acc: {poison_acc}')
        # print(f'check: {check}')
        for reduce_ratio in reduce_ratio_li:
            if reduce_ratio is None or reduce_ratio >= 1.0:
                continue
            else:
                if args.reduction == 'coarsening':
                    # for coarsening_method in ['variation_cliques']:
                    for coarsening_method in ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'kron']:
                        if test_gnn == 'rgcn':
                            reduce_acc = -1
                        coarsen_adj, coarsen_feature, coarsen_labels, coarsen_idx_train, coarsen_idx_val, node_ratio, edge_ratio = coarsen_poison(
                            num_classes=(labels.max() + 1).item(),
                            num_nodes=features.shape[0],
                            poisoned_adj=adj,
                            features=features, 
                            labels=labels, 
                            idx_train=idx_train, 
                            idx_val=idx_val, 
                            coarsening_rate=reduce_ratio,
                            method=coarsening_method
                        )
                        reduce_gnn = train_gnn(coarsen_adj, coarsen_feature, coarsen_labels, coarsen_idx_train, coarsen_idx_val, args, test_gnn=test_gnn, device=device)
                        reduce_acc = eval_gnn(reduce_gnn, modified_adj, modified_features, labels, idx_test, args, test_gnn=test_gnn, device=device)
                        result['seed'].append(seed)
                        result['dataset'].append(args.dataset)
                        result['method'].append(coarsening_method)
                        result['reduce_ratio'].append(reduce_ratio)
                        result['clean_acc'].append(clean_acc)
                        result['poison_acc'].append(poison_acc)
                        result['reduce_acc'].append(reduce_acc)
                        result['model'].append(test_gnn)

                        print(f'\tcoarsening_method: {coarsening_method} reduce_ratio: {reduce_ratio}, test_gnn: {test_gnn}, reduce_acc: {reduce_acc}')
                elif args.reduction == 'sparsification':
                    # for sparsification_method in ['local_degree']:
                    for sparsification_method in ['random_node_edge', 'local_degree', 'local_similarity', 'scan']:
                        reduced_adj, edge_ratio = sparsify(
                            poisoned_adj=adj,
                            target_ratio=reduce_ratio,
                            method=sparsification_method,
                        )
                        reduce_gnn = train_gnn(reduced_adj, features, labels, idx_train, idx_val, args, test_gnn=test_gnn, device=device)
                        reduce_acc = eval_gnn(reduce_gnn, modified_adj, modified_features, labels, idx_test, args, test_gnn=test_gnn, device=device)

                        result['seed'].append(seed)
                        result['dataset'].append(args.dataset)
                        result['method'].append(sparsification_method)
                        result['reduce_ratio'].append(reduce_ratio)
                        result['clean_acc'].append(clean_acc)
                        result['poison_acc'].append(poison_acc)
                        result['reduce_acc'].append(reduce_acc)
                        result['model'].append(test_gnn)
                
                        print(f'\tsparsification_method: {sparsification_method} reduce_ratio: {reduce_ratio}, test_gnn: {test_gnn}, reduce_acc: {reduce_acc}')
    return result

if __name__ == '__main__':
    result = {
        'seed': [],
        'dataset': [],
        'method': [],
        'model': [],
        'reduce_ratio': [],
        'clean_acc': [],
        'poison_acc': [],
        'reduce_acc': []
    }
    
    for seed in tqdm(range(1, 6)):
        args.seed = seed
        print(f"=====seed: {args.seed}=====")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)

        result = main(result)

            

    result_path = f'./results/{datetime.now().strftime("%b_%d")}'
    attack = 'agia'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file_name = f'{result_path}/PoisonACC_{args.dataset}_{attack}_{args.ptb_rate}_{args.reduction}.csv'
    
    df = pd.DataFrame(result)
    df.to_csv(file_name, index=False)
