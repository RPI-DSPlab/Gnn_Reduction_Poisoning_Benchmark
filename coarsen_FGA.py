from deeprobust.graph.targeted_attack import FGA
from deeprobust.graph.defense import GCN, GAT
from deeprobust.graph.utils import accuracy, classification_margin
from deeprobust.graph.data import Dataset

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix

import argparse
from tqdm import tqdm
from datetime import datetime
from utils import coarsening, load_data, get_logger



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora', 
                    choices=['Cora', 'Pubmed', 'Flickr', 'Polblogs'])
parser.add_argument('--gnn', type=str, default='gcn',
                    choices=['gcn', 'gat'])
parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods',
                    choices=['variation_neighborhoods_degree', 'variation_neighborhoods','variation_edges_degree','variation_edges', 'variation_cliques_degree', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'kron'],
                    help="Method of coarsening")
parser.add_argument('--no_cuda', type=bool, default=False)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')

args = parser.parse_args()
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# set up logger
log_path = f'./log/{datetime.now().strftime("%b_%d")}'
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_path += f'/coarsen_FGA_{args.seed}_{args.dataset}_{args.coarsening_method}.log'
logger = get_logger(
    logpath=log_path,
    filepath=os.path.abspath(__file__)
)

# Setup results
path = f'./results/'
if not os.path.exists(path):
    os.makedirs(path)
if os.path.exists(f'{path}/{args.gnn}.csv'):
    df = pd.read_csv(f'{path}/{args.gnn}.csv')
else:
    df = {
        'seed': [],
        'dataset': [],
        'coarsening_method': [],
        'coarsening_rate': [],
        'misclassification_rate': [],
        'node_ratio_avg': [],
        'edge_ratio_avg': [],
        'node_ratio_std': [],
        'edge_ratio_std': []
    }
    df = pd.DataFrame(df)
df['seed'] = df['seed'].astype(int)

def export(results):
    result_df = pd.concat([df, pd.DataFrame(results)])
    result_df.to_csv(f'{path}/{args.gnn}.csv', index=False)

# Load dataset
data = Dataset(root='./data', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
if args.dataset == 'Pubmed':
    # convert adj and features from scipy.sparse._arrays.csr_array to csr_matrix
    adj = csr_matrix(adj)
    features = csr_matrix(features)
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

num_classes = (labels.max() + 1).item()
idx_unlabeled = np.union1d(idx_val, idx_test)
logger.info(f'adj: {adj.shape} {type(adj)}')
logger.info(f'features: {features.shape} {type(features)}')
logger.info(f'labels: {labels.shape} {type(labels)}')
logger.info(f'training nodes: {idx_train.shape} {type(idx_train)}')
logger.info(f'validation nodes: {idx_val.shape} {type(idx_val)}')
logger.info(f'test nodes: {idx_test.shape} {type(idx_test)}')
logger.info(f'unlabeled nodes: {idx_unlabeled.shape}')
logger.info(f'num_classes: {num_classes}')

# Setup Surrogate model
if args.gnn == 'gcn':
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, device=device)
elif args.gnn == 'gat':
    surrogate = GAT(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=8, device=device)
else:
    logger.info("Model not implemented, use gcn as default")
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, device=device)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val)

# Setup Attack Model
target_node = 0
model = FGA(surrogate, nnodes=adj.shape[0], device=device)
model = model.to(device)


def coarsen(poisoned_adj, features, labels, idx_train, idx_val, coarsening_rate, method=None):
    # convert adj sparse matrix to edge_index
    poison_edge_index = from_scipy_sparse_matrix(poisoned_adj)[0]
    # convert features from csr_matrix to tensor
    poisoned_x = torch.tensor(features.todense(), dtype=torch.float)
    poison_labels = torch.tensor(labels, dtype=torch.long)
    if method is not None and coarsening_rate < 1.0:
        candidate, C_list, Gc_list = coarsening(poison_edge_index, 1-coarsening_rate, method)
        coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, coarsen_labels = load_data(num_classes, candidate, C_list, Gc_list, poisoned_x, poison_labels, idx_train, idx_val, verbose=False)
        coarsen_adj = torch.zeros((coarsen_features.shape[0], coarsen_features.shape[0]))
        for i in range(coarsen_edge.shape[1]):
            coarsen_adj[coarsen_edge[0][i]][coarsen_edge[1][i]] = 1
            coarsen_adj[coarsen_edge[1][i]][coarsen_edge[0][i]] = 1
        coarsen_adj = csr_matrix(coarsen_adj)

        coarsen_poison_idx_train = torch.nonzero(coarsen_train_mask).flatten()
        coarsen_poison_idx_train = coarsen_poison_idx_train.cpu().numpy()
        coarsen_poison_idx_val = torch.nonzero(coarsen_val_mask).flatten()
        coarsen_poison_idx_val = coarsen_poison_idx_val.cpu().numpy()
        edge_ratio = coarsen_edge.shape[1] / poison_edge_index.shape[1]
        node_ratio = coarsen_adj.shape[0] / adj.shape[0]

        coarsen_labels = coarsen_labels.numpy()
        coarsen_features = csr_matrix(coarsen_features.numpy())
    else:
        coarsen_features = features
        coarsen_labels = labels
        coarsen_adj = poisoned_adj
        coarsen_poison_idx_train = idx_train
        coarsen_poison_idx_val = idx_val
        node_ratio = 1.0
        edge_ratio = 1.0
    
    return coarsen_adj, coarsen_features, coarsen_labels, coarsen_poison_idx_train, coarsen_poison_idx_val, node_ratio, edge_ratio


def single_test(adj, features, target_node, coarsening_rate=1.0, method=None):
    coarsen_adj, coarsen_feature, coarsen_labels, coarsen_idx_train, coarsen_idx_val, node_ratio, edge_ratio = coarsen(
        poisoned_adj=adj, 
        features=features, 
        labels=labels, 
        idx_train=idx_train, 
        idx_val=idx_val, 
        coarsening_rate=coarsening_rate,
        method=method
    )
    # test on GCN (poisoning attack)
    if args.gnn == 'gcn':
        gnn = GCN(nfeat=coarsen_feature.shape[1],
                nhid=16,
                nclass=coarsen_labels.max().item() + 1,
                dropout=0.5, device=device)
    elif args.gnn == 'gat':
        gnn = GAT(nfeat=coarsen_feature.shape[1],
                nhid=16,
                nclass=coarsen_labels.max().item() + 1,
                dropout=0.5, device=device)
    else:
        logger.info("Model not implemented, use gcn as default")
        gnn = GCN(nfeat=coarsen_feature.shape[1],
                nhid=16,
                nclass=coarsen_labels.max().item() + 1,
                dropout=0.5, device=device)
    gnn = gnn.to(device)
    gnn.fit(coarsen_feature, coarsen_adj, coarsen_labels, coarsen_idx_train, coarsen_idx_val, patience=30)
    gnn.eval()
    output = gnn.predict(features=features, adj=adj)
    probs = torch.exp(output[[target_node]])

    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    
    # exclude the target node from the test set
    idx_test_used = np.setdiff1d(idx_test, target_node)
    clean_acc = accuracy(output[idx_test_used], labels[idx_test_used])
    return acc_test.item(), node_ratio, edge_ratio, clean_acc.item()


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

    return high + low + other


def coarsen_poison():
    # test on 40 nodes on poisoining attack
    cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes()
    num = len(node_list)
    
    coarsening_rates = np.linspace(0.1, 1.0, 9)
    results = {
        'seed': [],
        'dataset': [],
        'coarsening_method': [],
        'coarsening_rate': [],
        'misclassification_rate': [],
        'clean_acc_avg': [],
        'clean_acc_std': [],
        'node_ratio_avg': [],
        'edge_ratio_avg': [],
        'node_ratio_std': [],
        'edge_ratio_std': []
    }
    logger.info("=== Coarsen Poison ===")
    count = df[(df['seed'] == args.seed) & (df['dataset'] == args.dataset) & (df['coarsening_method'] == args.coarsening_method)].shape[0]
    try:
        for coarsening_rate in coarsening_rates:
            # logger.info(df[(df['seed'] == args.seed) & (df['dataset'] == args.dataset) & (df['coarsening_method'] == args.coarsening_method) & (df['coarsening_rate'] == coarsening_rate)])
            if count > 0:
                count -= 1
                logger.info(f"Seed {args.seed} {args.dataset} {args.coarsening_method} {coarsening_rate} already exists in the results")
                continue
            avg_node_ratio_li = []
            avg_edge_ratio_li = []
            avg_clean_acc_li = []

            logger.info(f'-----{args.coarsening_method} method with {coarsening_rate} coarsening rate-----')
            for target_node in tqdm(node_list):
                n_perturbations = int(degrees[target_node])
                model = FGA(surrogate, nnodes=adj.shape[0], device=device)
                model = model.to(device)
                model.attack(features, adj, labels, idx_train, target_node, n_perturbations)
                modified_adj = model.modified_adj
                acc, node_ratio, edge_ratio, clean_acc = single_test(modified_adj, features, target_node, coarsening_rate=coarsening_rate, method=args.coarsening_method)
                if acc == 0:
                    cnt += 1
                avg_node_ratio_li.append(node_ratio)
                avg_edge_ratio_li.append(edge_ratio)
                avg_clean_acc_li.append(clean_acc)
            
            misclassification_rate = cnt/num
            logger.info(f'misclassification rate: {misclassification_rate}; avg clean acc: {np.mean(avg_clean_acc_li)}')
            results['seed'].append(args.seed)
            results['dataset'].append(args.dataset)
            results['coarsening_method'].append(args.coarsening_method)
            results['coarsening_rate'].append(coarsening_rate)
            results['misclassification_rate'].append(misclassification_rate)
            results['clean_acc_avg'].append(np.mean(avg_clean_acc_li))
            results['clean_acc_std'].append(np.std(avg_clean_acc_li))
            results['node_ratio_avg'].append(np.mean(avg_node_ratio_li))
            results['edge_ratio_avg'].append(np.mean(avg_edge_ratio_li))
            results['node_ratio_std'].append(np.std(avg_node_ratio_li))
            results['edge_ratio_std'].append(np.std(avg_edge_ratio_li))
            cnt = 0
        export(results)
    except Exception as e:
        logger.info(f"Error occured: {args.seed} {args.dataset} {args.coarsening_method}")
        logger.info(e)
        logger.info("Exporting results")
        export(results)


if __name__ == "__main__":
    coarsen_poison()

