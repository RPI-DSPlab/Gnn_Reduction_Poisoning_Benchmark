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
from service.sparsify import sparsify

from tqdm import tqdm
from datetime import datetime
from utils import setup_logger, parse_args





args = parse_args()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# set up logger
logger = setup_logger(args)

# Setup results
path = f'./results/{datetime.now().strftime("%b_%d")}'
file_name = f'{path}/FGA_{args.technique}_{args.gnn}.csv'
if not os.path.exists(path):
    os.makedirs(path)
if os.path.exists(file_name):
    df = pd.read_csv(f'{path}/{args.technique}_{args.gnn}.csv')
else:
    df = {
        'seed': [],
        'dataset': [],
        'method': [],
        'rate': [],
        'misclassification_rate': [],
        'clean_acc_avg': [],
        'clean_acc_std': [],
        'node_ratio_avg': [],
        'edge_ratio_avg': [],
        'node_ratio_std': [],
        'edge_ratio_std': []
    }
    df = pd.DataFrame(df)
df['seed'] = df['seed'].astype(int)

def export(results):
    result_df = pd.concat([df, pd.DataFrame(results)])
    result_df.to_csv(f'{path}/{args.technique}_{args.gnn}.csv', index=False)

# Load dataset
data = Dataset(root='./data', name=args.dataset)
clean_adj, clean_features, clean_labels = data.adj, data.features, data.labels
# if args.dataset == 'Pubmed':
#     # convert adj and features from scipy.sparse._arrays.csr_array to csr_matrix
#     clean_adj = csr_matrix(clean_adj)
#     clean_features = csr_matrix(clean_features)
clean_idx_train, clean_idx_val, clean_idx_test = data.idx_train, data.idx_val, data.idx_test

num_classes = (clean_labels.max() + 1).item()
idx_unlabeled = np.union1d(clean_idx_val, clean_idx_test)
logger.info(f'clean_adj: {clean_adj.shape} {type(clean_adj)}')
logger.info(f'clean_features: {clean_features.shape} {type(clean_features)}')
logger.info(f'clean_labels: {clean_labels.shape} {type(clean_labels)}')
logger.info(f'training nodes: {clean_idx_train.shape} {type(clean_idx_train)}')
logger.info(f'validation nodes: {clean_idx_val.shape} {type(clean_idx_val)}')
logger.info(f'test nodes: {clean_idx_test.shape} {type(clean_idx_test)}')
logger.info(f'unlabeled nodes: {idx_unlabeled.shape}')
logger.info(f'num_classes: {num_classes}')

# Setup Surrogate model
if args.gnn == 'gcn':
    surrogate = GCN(nfeat=clean_features.shape[1], nclass=clean_labels.max().item()+1,
                    nhid=16, device=device)
elif args.gnn == 'gat':
    surrogate = GAT(nfeat=clean_features.shape[1], nclass=clean_labels.max().item()+1,
                    nhid=8, device=device)
else:
    logger.info("Model not implemented, use gcn as default")
    surrogate = GCN(nfeat=clean_features.shape[1], nclass=clean_labels.max().item()+1,
                    nhid=16, device=device)
surrogate = surrogate.to(device)
surrogate.fit(clean_features, clean_adj, clean_labels, clean_idx_train, clean_idx_val)

# Setup Attack Model
target_node = 0
model = FGA(surrogate, nnodes=clean_adj.shape[0], device=device)
model = model.to(device)
    

def single_test(poisoned_adj, reduced_adj, features, labels, idx_train, idx_val, target_node):
    # test on GCN (poisoning attack)
    if args.gnn == 'gcn':
        gnn = GCN(nfeat=features.shape[1],
                nhid=16,
                nclass=labels.max().item() + 1,
                dropout=0.5, device=device)
    elif args.gnn == 'gat':
        gnn = GAT(nfeat=features.shape[1],
                nhid=16,
                nclass=labels.max().item() + 1,
                dropout=0.5, device=device)
    else:
        logger.info("Model not implemented, use gcn as default")
        gnn = GCN(nfeat=features.shape[1],
                nhid=16,
                nclass=labels.max().item() + 1,
                dropout=0.5, device=device)
    gnn = gnn.to(device)
    gnn.fit(features, reduced_adj, labels, idx_train, idx_val, patience=30)
    gnn.eval()
    output = gnn.predict(features=features, adj=poisoned_adj)

    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    # exclude the target node from the test set
    idx_test_used = np.setdiff1d(clean_idx_test, target_node)
    clean_acc = accuracy(output[idx_test_used], labels[idx_test_used])
    return acc_test.item(), clean_acc.item()


def select_nodes(target_gcn=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    if target_gcn is None:
        target_gcn = GCN(nfeat=clean_features.shape[1],
                  nhid=16,
                  nclass=clean_labels.max().item() + 1,
                  dropout=0.5, device=device)
        target_gcn = target_gcn.to(device)
        target_gcn.fit(clean_features, clean_adj, clean_labels, clean_idx_train, clean_idx_val, patience=30)
    target_gcn.eval()
    output = target_gcn.predict()

    margin_dict = {}
    for idx in clean_idx_test:
        margin = classification_margin(output[idx], clean_labels[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other


def sparsify_poison_test():
    # test on 40 nodes on poisoining attack
    cnt = 0
    degrees = clean_adj.sum(0).A1
    node_list = select_nodes()
    num = len(node_list)
    
    reduce_ratio_li = np.linspace(0.1, 1.0, 9)
    results = {
        'seed': [],
        'dataset': [],
        'method': [],
        'rate': [],
        'misclassification_rate': [],
        'clean_acc_avg': [],
        'clean_acc_std': [],
        'node_ratio_avg': [],
        'edge_ratio_avg': [],
        'node_ratio_std': [],
        'edge_ratio_std': []
    }
    logger.info("=== Coarsen Poison ===")
    try:
        for reduce_rate in reduce_ratio_li:
            avg_node_ratio_li = []
            avg_edge_ratio_li = []
            avg_clean_acc_li = []

            logger.info(f'-----{args.sparsification_method} method with {reduce_rate} coarsening rate-----')
            for target_node in tqdm(node_list):
                n_perturbations = int(degrees[target_node])
                model = FGA(surrogate, nnodes=clean_adj.shape[0], device=device)
                model = model.to(device)
                model.attack(clean_features, clean_adj, clean_labels, clean_idx_train, target_node, n_perturbations)
                modified_adj = model.modified_adj

                reduced_adj, edge_ratio = sparsify(
                    poisoned_adj=modified_adj, 
                    target_ratio=reduce_rate,
                    method=args.sparsification_method
                )
                acc, clean_acc = single_test(
                    poisoned_adj=modified_adj,
                    reduced_adj=reduced_adj,
                    features=clean_features,
                    labels=clean_labels,
                    idx_train=clean_idx_train,
                    idx_val=clean_idx_val,
                    target_node=target_node
                )
                if acc == 0:
                    cnt += 1
                avg_node_ratio_li.append(1)
                avg_edge_ratio_li.append(edge_ratio)
                avg_clean_acc_li.append(clean_acc)
            
            misclassification_rate = cnt/num
            logger.info(f'misclassification rate: {misclassification_rate}; avg clean acc: {np.mean(avg_clean_acc_li)}')
            
            # store results
            results['seed'].append(args.seed)
            results['dataset'].append(args.dataset)
            results['method'].append(args.sparsification_method)
            results['rate'].append(reduce_rate)
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
        logger.info(f"Error occured: {args.seed} {args.dataset} {args.sparsification_method}")
        logger.info(e)
        logger.info("Exporting results")
        export(results)
    

if __name__ == "__main__":
    sparsify_poison_test()