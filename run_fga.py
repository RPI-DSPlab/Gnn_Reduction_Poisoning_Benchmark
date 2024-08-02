import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import FGA
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from tqdm import tqdm
import argparse
from scipy.sparse import csr_matrix
import pandas as pd
from datetime import datetime
import os

from service.coarsen import coarsen_poison
from service.sparsify import sparsify

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')

parser.add_argument('--defense', type=str, default='coarsening', 
                    choices=['coarsening', 'sparsification', 'sparsification_svd', 'jaccard', 'svd', 'rgcn', 'median', 'airgnn'])
parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods',
                    choices=['variation_neighborhoods_degree', 'variation_neighborhoods','variation_edges_degree','variation_edges', 'variation_cliques_degree', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'kron'],
                    help="Method of coarsening")
parser.add_argument('--sparsification_method', type=str, default='random_node_edge',
                    choices=['random_node_edge', 'random_edge', 'local_degree', 'forest_fire', 'local_similarity', 'scan', 'simmelian'],
                    help="Method of sparsification")
parser.add_argument('--device_id', type=int, default=0, help='Device ID')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
if args.dataset == 'pubmed':
    # convert adj and features from scipy.sparse._arrays.csr_array to csr_matrix
    adj = csr_matrix(adj)
    features = csr_matrix(features)

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

print(args)
print('adj: ', adj.shape, type(adj))
print('features: ', features.shape, type(features))
print('labels: ', labels.shape, type(labels))
print('idx_train: ', len(idx_train), type(idx_train))
print('idx_val: ', len(idx_val), type(idx_val))
print('idx_test: ', len(idx_test), type(idx_test))

idx_unlabeled = np.union1d(idx_val, idx_test)

# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val)

surrogate.eval()
output = surrogate.predict(features, adj)
clean_acc = accuracy(output[idx_test], labels[idx_test])

print('clean accuracy', clean_acc.item())

# Setup Attack Model
target_node = 0


def select_nodes(target_gcn=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    if target_gcn is None:
        target_gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)
        target_gcn = target_gcn.to(device)
        target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    target_gcn.eval()
    output = target_gcn.predict()

    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high, low, other


def single_test(adj, features, target_node, reduce_ratio=None):
    gcn = GCN(nfeat=features.shape[1],
                nhid=16,
                nclass=labels.max().item() + 1,
                dropout=0.5, device=device)
    gcn = gcn.to(device)
    if reduce_ratio is None or reduce_ratio >= 1.0:
        # print("No defense applied")
        gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    else:
        # print("Defense applied")
        if args.defense == 'coarsening':
            coarsen_adj, coarsen_feature, coarsen_labels, coarsen_idx_train, coarsen_idx_val, node_ratio, edge_ratio = coarsen_poison(
                num_classes=(labels.max() + 1).item(),
                num_nodes=features.shape[0],
                poisoned_adj=adj,
                features=features, 
                labels=labels, 
                idx_train=idx_train, 
                idx_val=idx_val, 
                coarsening_rate=reduce_ratio,
                method=args.coarsening_method
            )
            gcn.fit(coarsen_feature, coarsen_adj, coarsen_labels, coarsen_idx_train, coarsen_idx_val, patience=30)
        elif args.defense == 'sparsification':
            reduced_adj, edge_ratio = sparsify(
                poisoned_adj=adj,
                target_ratio=reduce_ratio,
                method=args.sparsification_method,
            )
            gcn.fit(features, reduced_adj, labels, idx_train, idx_val, patience=30)
    gcn.eval()
    output = gcn.predict(features, adj)

    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])

    idx_test_used = np.setdiff1d(idx_test, target_node)
    clean_acc = accuracy(output[idx_test_used], labels[idx_test_used])

    return acc_test.item(), clean_acc.item()


def multi_test_poison(reduce_ratio):
    def test_poison(node_list):
        temp_cnt = 0
        num = len(node_list)
        clean_acc_li = []
        for target_node in tqdm(node_list):
            n_perturbations = int(degrees[target_node])
            model = FGA(surrogate, nnodes=adj.shape[0], device=device)
            model = model.to(device)
            model.attack(features, adj, labels, idx_train, target_node, n_perturbations)
            modified_adj = model.modified_adj
            modified_features = features
            acc, clean_acc = single_test(modified_adj, modified_features, target_node, reduce_ratio)
            if acc == 0:
                temp_cnt += 1
            clean_acc_li.append(clean_acc)
        
        return temp_cnt, np.mean(np.array(clean_acc_li))
    # test on 40 nodes on poisoining attack
    cnt = 0
    degrees = adj.sum(0).A1
    high, low, middle = select_nodes()
    clean_acc_li = []

    high_cnt, clean_acc = test_poison(high)
    high_asr = high_cnt / 10
    clean_acc_li.append(clean_acc)
    print('High misclassification rate : %s' % (high_asr))
    low_cnt, clean_acc = test_poison(low)
    low_asr = low_cnt / 10
    clean_acc_li.append(clean_acc)
    print('Low misclassification rate : %s' % (low_asr))
    middle_cnt, clean_acc = test_poison(middle)
    middle_asr = middle_cnt / 20
    clean_acc_li.append(clean_acc)
    print('Middle misclassification rate : %s' % (middle_asr))

    cnt = high_cnt + low_cnt + middle_cnt
    overall_asr = cnt / 40
    overall_acc = np.mean(np.array(clean_acc_li))
    print('Overall misclassification rate : %s' % (overall_asr))

    return high_asr, low_asr, middle_asr, overall_asr, overall_acc


if __name__ == '__main__':
    result = {
        'seed': [],
        'dataset': [],
        'method': [],
        'reduce_ratio': [],
        'high_asr': [],
        'low_asr': [],
        'middle_asr': [],
        'overall_asr': [],
        'overall_acc': []
    }
    if args.defense == 'coarsening':
        method = args.coarsening_method
    elif args.defense == 'sparsification':
        method = args.sparsification_method
    else:
        raise ValueError('Invalid defense method')
    for seed in range(1, 6):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)
        
        reduce_ratio_li = np.linspace(1.0, 0.1, 5)
        for reduce_ratio in reduce_ratio_li:
            print('=== [Seed %s] Reduce ratio %s ===' % (seed, reduce_ratio))
            high_asr, low_asr, middle_asr, overall_asr, overall_acc = multi_test_poison(reduce_ratio)

            result['seed'].append(seed)
            result['dataset'].append(args.dataset)
            result['method'].append(method)
            result['reduce_ratio'].append(reduce_ratio)
            result['high_asr'].append(high_asr)
            result['low_asr'].append(low_asr)
            result['middle_asr'].append(middle_asr)
            result['overall_asr'].append(overall_asr)
            result['overall_acc'].append(overall_acc)
    
    result_path = f'./results/{datetime.now().strftime("%b_%d")}'
    gnn = 'gcn'
    attack = 'fga'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if args.defense == 'coarsening':
        file_name = f'{result_path}/{args.dataset}_{attack}_{args.defense}_{args.coarsening_method}_{gnn}.csv'
    elif args.defense == 'sparsification':
        file_name = f'{result_path}/{args.dataset}_{attack}_{args.defense}_{args.sparsification_method}_{gnn}.csv'
    
    df = pd.DataFrame(result)
    df.to_csv(file_name, index=False)