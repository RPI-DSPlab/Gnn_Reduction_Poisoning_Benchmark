from deeprobust.graph.targeted_attack import FGA
from deeprobust.graph.defense import GCN, GAT, GCNJaccard, GCNSVD, RGCN, MedianGCN
from deeprobust.graph.defense_pyg import AirGNN
from deeprobust.graph.utils import accuracy
from deeprobust.graph.data import Dataset, Dpr2Pyg

import os
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from tqdm import tqdm
from datetime import datetime
from utils import setup_logger, parse_args, select_nodes
from service.coarsen import coarsen_poison
from service.sparsify import sparsify
from dao.targetted_result import TargettedResultDAO


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
    df = pd.read_csv(file_name)
    previous_result = TargettedResultDAO(df)
else:
    previous_result = TargettedResultDAO()

# Load dataset
data = Dataset(root='./data', name=args.dataset)
clean_adj, clean_features, clean_labels = data.adj, data.features, data.labels
if args.dataset == 'Pubmed':
    # convert adj and features from scipy.sparse._arrays.csr_array to csr_matrix
    clean_adj = csr_matrix(clean_adj)
    clean_features = csr_matrix(clean_features)
clean_idx_train, clean_idx_val, clean_idx_test = data.idx_train, data.idx_val, data.idx_test

num_classes = (clean_labels.max() + 1).item()
idx_unlabeled = np.union1d(clean_idx_val, clean_idx_test)
logger.info(f'adj: {clean_adj.shape} {type(clean_adj)}')
logger.info(f'features: {clean_features.shape} {type(clean_features)}')
logger.info(f'clean_labels: {clean_labels.shape} {type(clean_labels)}')
logger.info(f'training nodes: {clean_idx_train.shape} {type(clean_idx_train)}')
logger.info(f'validation nodes: {clean_idx_val.shape} {type(clean_idx_val)}')
logger.info(f'test nodes: {clean_idx_test.shape} {type(clean_idx_test)}')
logger.info(f'unlabeled nodes: {idx_unlabeled.shape}')
logger.info(f'num_classes: {num_classes}')

# Setup Surrogate model
if args.gnn == 'gcn':
    surrogate = GCN(nfeat=clean_features.shape[1], nclass=clean_labels.max().item()+1,
                    nhid=args.nhid, device=device)
elif args.gnn == 'gat':
    surrogate = GAT(nfeat=clean_features.shape[1], nclass=clean_labels.max().item()+1,
                    nhid=args.nhid, device=device)
else:
    logger.info("Model not implemented, use gcn as default")
    surrogate = GCN(nfeat=clean_features.shape[1], nclass=clean_labels.max().item()+1,
                    nhid=args.nhid, device=device)
surrogate = surrogate.to(device)
surrogate.fit(clean_features, clean_adj, clean_labels, clean_idx_train, clean_idx_val)


def single_test(poisoned_adj, reduced_adj, features, labels, idx_train, idx_val, target_node):
    # test on GCN (poisoning attack)
    if args.gnn == 'gcn':
        gnn = GCN(nfeat=features.shape[1],
                nhid=args.nhid,
                nclass=labels.max().item() + 1,
                dropout=0.5, device=device)
    elif args.gnn == 'gat':
        gnn = GAT(nfeat=features.shape[1],
                nhid=args.nhid,
                nclass=labels.max().item() + 1,
                dropout=0.5, device=device)
    else:
        logger.info("Model not implemented, use gcn as default")
        gnn = GCN(nfeat=features.shape[1],
                nhid=args.nhid,
                nclass=labels.max().item() + 1,
                dropout=0.5, device=device)
    gnn = gnn.to(device)
    gnn.fit(features, reduced_adj, labels, idx_train, idx_val, patience=30)
    gnn.eval()
    output = gnn.predict(features=clean_features, adj=poisoned_adj)

    acc_test = (output.argmax(1)[target_node] == clean_labels[target_node])
    
    # exclude the target node from the test sets
    idx_test_used = np.setdiff1d(clean_idx_test, target_node)
    clean_acc = accuracy(output[idx_test_used], clean_labels[idx_test_used])
    return acc_test.item(), clean_acc.item()


def poison(n_perturbations, target_node):
    model = FGA(surrogate, nnodes=clean_adj.shape[0], device=device)
    model = model.to(device)
    model.attack(clean_features, clean_adj, clean_labels, clean_idx_train, target_node, n_perturbations)
    modified_adj = model.modified_adj
    return modified_adj


def coarsen_poison_test(degrees, reduce_rate, target_node):
    success = 0
    n_perturbations = int(degrees[target_node])
    modified_adj = poison(n_perturbations, target_node)
    coarsen_adj, coarsen_feature, coarsen_labels, coarsen_idx_train, coarsen_idx_val, node_ratio, edge_ratio = coarsen_poison(
        num_classes=num_classes,
        num_nodes=modified_adj.shape[0],
        poisoned_adj=modified_adj, 
        features=clean_features, 
        labels=clean_labels, 
        idx_train=clean_idx_train, 
        idx_val=clean_idx_val, 
        coarsening_rate=reduce_rate,
        method=args.coarsening_method
    )
    acc, clean_acc = single_test(
        poisoned_adj=modified_adj,
        reduced_adj=coarsen_adj,
        features=coarsen_feature,
        labels=coarsen_labels,
        idx_train=coarsen_idx_train,
        idx_val=coarsen_idx_val,
        target_node=target_node
    )
    if acc == 0:
        success = 1

    return success, node_ratio, edge_ratio, clean_acc


def sparsify_poison_test(degrees, reduce_rate, target_node):
    success = 0
    n_perturbations = int(degrees[target_node])
    modified_adj = poison(n_perturbations, target_node)
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
        success += 1
    
    return success, 1, edge_ratio, clean_acc


def jaccard_poison_test(degrees, target_node):
    success = 0
    n_perturbations = int(degrees[target_node])
    modified_adj = poison(n_perturbations, target_node)

    model = GCNJaccard(nfeat=clean_features.shape[1],
        nhid=args.nhid,
        nclass=clean_labels.max().item() + 1,
        device=device
    )
    model = model.to(device)
    model.fit(clean_features, modified_adj, clean_labels, clean_idx_train, clean_idx_val, threshold=0.01, verbose=False)
    output = model.predict(clean_features, modified_adj)
    acc = (output.argmax(1)[target_node] == clean_labels[target_node])

    if acc == 0:
        success = 1
    # exclude the target node from the test sets
    idx_test_used = np.setdiff1d(clean_idx_test, target_node)
    clean_acc = accuracy(output[idx_test_used], clean_labels[idx_test_used])

    return success, -1, -1, clean_acc


def svd_poison_test(degrees, target_node):
    success = 0
    n_perturbations = int(degrees[target_node])
    modified_adj = poison(n_perturbations, target_node)
    
    model = GCNSVD(nfeat=clean_features.shape[1],
        nhid=args.nhid,
        nclass=clean_labels.max().item() + 1,
        device=device
    )
    model = model.to(device)
    model.fit(clean_features, modified_adj, clean_labels, clean_idx_train, clean_idx_val, k=15, verbose=False)
    output = model.predict(clean_features, modified_adj)
    acc = (output.argmax(1)[target_node] == clean_labels[target_node])

    if acc == 0:
        success = 1
    # exclude the target node from the test sets
    idx_test_used = np.setdiff1d(clean_idx_test, target_node)
    clean_acc = accuracy(output[idx_test_used], clean_labels[idx_test_used])

    return success, -1, -1, clean_acc


def rgcn_poison_test(degrees, target_node):
    success = 0
    n_perturbations = int(degrees[target_node])
    modified_adj = poison(n_perturbations, target_node)
    
    model = RGCN(
        nnodes=modified_adj.shape[0],
        nfeat=clean_features.shape[1],
        nhid=32,
        nclass=clean_labels.max().item() + 1,
        device=device
    )
    model = model.to(device)
    model.fit(clean_features, modified_adj, clean_labels, clean_idx_train, clean_idx_val, train_iters=200, verbose=False)
    output = model.predict()
    acc = (output.argmax(1)[target_node] == clean_labels[target_node])

    if acc == 0:
        success = 1
    # exclude the target node from the test sets
    idx_test_used = np.setdiff1d(clean_idx_test, target_node)
    clean_acc = accuracy(output[idx_test_used], clean_labels[idx_test_used])

    return success, -1, -1, clean_acc


def median_gcn_poison_test(degrees, target_node):
    success = 0
    n_perturbations = int(degrees[target_node])
    modified_adj = poison(n_perturbations, target_node)

    model = MedianGCN(nfeat=clean_features.shape[1],
        nhid=args.nhid,
        nclass=clean_labels.max().item() + 1,
        dropout=0.5, device=device)
    model = model.to(device)

    pyg_data = Dpr2Pyg(data)
    pyg_data.update_edge_index(modified_adj)

    model.fit(pyg_data=pyg_data)
    output = model.predict()
    acc = (output.argmax(1)[target_node] == clean_labels[target_node])

    if acc == 0:
        success = 1
    # exclude the target node from the test sets
    idx_test_used = np.setdiff1d(clean_idx_test, target_node)
    clean_acc = accuracy(output[idx_test_used], clean_labels[idx_test_used])

    return success, -1, -1, clean_acc


def arignn_poison_test(degrees, target_node):
    success = 0
    n_perturbations = int(degrees[target_node])
    modified_adj = poison(n_perturbations, target_node)

    pyg_data = Dpr2Pyg(data)
    pyg_data.update_edge_index(modified_adj)

    model = AirGNN(
        nfeat=clean_features.shape[1], 
        nhid=64, 
        dropout=0.5, 
        with_bn=0,
        K=10, 
        weight_decay=5e-4, 
        nlayers=2,
        nclass=max(clean_labels).item()+1, 
        device=device
    ).to(device)
    model.fit(pyg_data, train_iters=1000, patience=1000, verbose=True)
    output = model.predict()
    acc = (output.argmax(1)[target_node] == clean_labels[target_node])

    if acc == 0:
        success = 1
    # exclude the target node from the test sets
    idx_test_used = np.setdiff1d(clean_idx_test, target_node)
    clean_acc = accuracy(output[idx_test_used], clean_labels[idx_test_used])

    return success, -1, -1, clean_acc



def exp(degrees, reduce_rate, node_li, avg_node_ratio_li, avg_edge_ratio_li, avg_clean_acc_li):
    cnt = 0
    temp_clean_acc = []
    for target_node in tqdm(node_li):
        if args.technique == 'coarsen':
            temp_count, node_ratio, edge_ratio, clean_acc = coarsen_poison_test(
                degrees=degrees,
                reduce_rate=reduce_rate,
                target_node=target_node
            )
        elif args.technique == 'sparsification':
            temp_count, node_ratio, edge_ratio, clean_acc = sparsify_poison_test(
                degrees=degrees,
                reduce_rate=reduce_rate,
                target_node=target_node
            )
        elif args.technique == 'jaccard':
            temp_count, node_ratio, edge_ratio, clean_acc = jaccard_poison_test(
                degrees=degrees,
                target_node=target_node
            )
        elif args.technique == 'svd':
            temp_count, node_ratio, edge_ratio, clean_acc = svd_poison_test(
                degrees=degrees,
                target_node=target_node
            )
        elif args.technique == 'rgcn':
            temp_count, node_ratio, edge_ratio, clean_acc = rgcn_poison_test(
                degrees=degrees,
                target_node=target_node
            )
        elif args.technique == 'median':
            temp_count, node_ratio, edge_ratio, clean_acc = median_gcn_poison_test(
                degrees=degrees,
                target_node=target_node
            )
        elif args.technique == 'airgnn':
            temp_count, node_ratio, edge_ratio, clean_acc = arignn_poison_test(
                degrees=degrees,
                target_node=target_node
            )
        cnt += temp_count
        avg_node_ratio_li.append(node_ratio)
        avg_edge_ratio_li.append(edge_ratio)
        avg_clean_acc_li.append(clean_acc)
        temp_clean_acc.append(clean_acc)

    local_asr = cnt/len(node_li)
    local_acc = np.mean(temp_clean_acc)
    return cnt, local_asr, local_acc


def main():
    # test on 40 nodes on poisoining attack
    degrees = clean_adj.sum(0).A1
    hard, middle, easy = select_nodes(
        clean_adj=clean_adj,
        clean_features=clean_features,
        clean_labels=clean_labels,
        clean_idx_train=clean_idx_train,
        clean_idx_val=clean_idx_val,
        clean_idx_test=clean_idx_test,
        target_gcn=surrogate,
        device=device
    )
    print("hard:", len(hard))
    print("middle:", len(middle))
    print("easy:", len(easy))
    node_list = hard + middle + easy
    num = len(node_list)
    
    reduce_ratio_li = np.linspace(0.1, 1.0, args.ratio_number)
    results = TargettedResultDAO()
    logger.info("=== Coarsen Poison ===")
    # try:
    for reduce_rate in reduce_ratio_li:
        cnt = 0
        avg_node_ratio_li = []
        avg_edge_ratio_li = []
        avg_clean_acc_li = []

        if args.technique == 'coarsen':
            logger.info(f'-----{args.coarsening_method} method with {reduce_rate} coarsening rate-----')
        elif args.technique == 'sparsification':
            logger.info(f'-----{args.sparsification_method} method with {reduce_rate} sparsification rate-----')
        # Robust node
        temp_cnt, hard_asr, temp_acc = exp(
            degrees=degrees,
            reduce_rate=reduce_rate,
            node_li=hard,
            avg_node_ratio_li=avg_node_ratio_li,
            avg_edge_ratio_li=avg_edge_ratio_li,
            avg_clean_acc_li=avg_clean_acc_li,
        )
        print('hard_asr:', hard_asr, 'hard_clean_acc:', temp_acc)
        cnt += temp_cnt
        # Random node
        temp_cnt, middle_asr, temp_acc = exp(
            degrees=degrees,
            reduce_rate=reduce_rate,
            node_li=middle,
            avg_node_ratio_li=avg_node_ratio_li,
            avg_edge_ratio_li=avg_edge_ratio_li,
            avg_clean_acc_li=avg_clean_acc_li,
        )
        print('middle_asr:', middle_asr, 'middle_clean_acc:', temp_acc)
        cnt += temp_cnt
        
        # Vulnerable node
        temp_cnt, easy_asr, temp_acc = exp(
            degrees=degrees,
            reduce_rate=reduce_rate,
            node_li=easy,
            avg_node_ratio_li=avg_node_ratio_li,
            avg_edge_ratio_li=avg_edge_ratio_li,
            avg_clean_acc_li=avg_clean_acc_li,
        )
        print('easy_asr:', easy_asr, 'easy_clean_acc:', temp_acc)
        cnt += temp_cnt
      
        misclassification_rate = cnt/num
        print()
        print(f'misclassification rate: {misclassification_rate}; avg clean acc: {np.mean(avg_clean_acc_li)}')

        # store results
        results.append(
            seed=args.seed,
            dataset=args.dataset,
            method=args.coarsening_method,
            rate=reduce_rate,
            misclassification_rate=misclassification_rate,
            hard_asr=hard_asr,
            middle_asr=middle_asr,
            easy_asr=easy_asr,
            clean_acc_avg=np.mean(avg_clean_acc_li),
            node_ratio_avg=np.mean(avg_node_ratio_li),
            edge_ratio_avg=np.mean(avg_edge_ratio_li),
            node_ratio_std=np.std(avg_node_ratio_li),
            edge_ratio_std=np.std(avg_edge_ratio_li),
            clean_acc_std=np.std(avg_clean_acc_li)
        )
    results.concat(previous_result)
    results.export(file_name)
    # except Exception as e:
    #     logger.info(f"Error occured: {args.seed} {args.dataset} {args.coarsening_method}")
    #     logger.info(e)
    #     logger.info("Exporting results")
    #     export(results)


if __name__ == "__main__":
    main()

