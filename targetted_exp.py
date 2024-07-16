from model.gcn import GCN
from model.gat import GAT
from model.gin import GIN
from deeprobust.graph.targeted_attack import Nettack, FGA
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import pickle5 as pickle
import os
import numpy as np
from scipy.sparse import csr_matrix
from deeprobust.graph.data import Dataset
import argparse
from utils import setup_logger, select_nodes, parse_args
from deeprobust.graph.utils import accuracy
from service.coarsen import coarsen_poison
from service.sparsify import sparsify
import torch
import pandas as pd
from tqdm import tqdm
from dao.targetted_result import TargettedResultDAO
from datetime import datetime


def store_surrogate(args):
    path = './data/surrogate'
    file_path = os.path.join(path, f'{args.dataset}_{args.gnn}_nhid{args.nhid}_{args.seed}.pkl')
    print(file_path)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            surrogate = pickle.load(f)
        print("Surrogate model loaded")
        return surrogate
    else:
        # Setup Surrogate model
        if args.gnn == 'gcn':
            surrogate = GCN(nfeat=clean_features.shape[1],
                            nhid=args.nhid,
                            nclass=num_classes,
                            device=device)
        elif args.gnn == 'gat':
            surrogate = GAT(nfeat=clean_features.shape[1],
                    nhid=args.nhid,
                    nclass=num_classes,
                    device=device)
        elif args.gnn == 'gin':
            surrogate = GIN(nfeat=clean_features.shape[1],
                    nhid=args.nhid,
                    nclass=num_classes,
                    device=device)
        else:
            logger.info("Model not implemented, use gcn as default")
            surrogate = GCN(nfeat=clean_features.shape[1],
                    nhid=args.nhid,
                    nclass=num_classes,
                    device=device)
        surrogate = surrogate.to(device)
        surrogate.fit(
            features=clean_features,
            adj=clean_adj,
            labels=clean_labels,
            idx_train=clean_idx_train,
            idx_val=clean_idx_val,
            train_iters=200,
            verbose=False
        )

        path = './data/surrogate'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(file_path, 'wb') as f:
            pickle.dump(surrogate, f)
        
        logger.info("Surrogate model stored")
        return surrogate

def store_nodes(clean_labels, clean_idx_test, surrogate, device):
    path = f'./data/target'
    file_path = os.path.join(path, f'{args.dataset}_{args.gnn}_nhid{args.nhid}_{args.seed}.csv')
    if os.path.exists(file_path):
        df_nodes = pd.read_csv(file_path)
        print("Nodes loaded")
        return df_nodes
    else: 
        hard, middle, easy = select_nodes(
            clean_labels=clean_labels,
            clean_idx_test=clean_idx_test,
            target_gcn=surrogate,
            device=device
        )
        results = {
            'node': [],
            'label': []
        }
        for node in hard:
            results['node'].append(node)
            results['label'].append('hard')
        for node in middle:
            results['node'].append(node)
            results['label'].append('middle')
        for node in easy:
            results['node'].append(node)
            results['label'].append('easy')

        results = pd.DataFrame(results)
        if not os.path.exists(path):
            os.makedirs(path)
        results.to_csv(file_path, index=False)

        logger.info("Nodes stored")
        return results


def Nettack_poison(n_perturbations, target_node):
    path = './data/poison/Nettack'
    file_path = os.path.join(path, f'{args.dataset}_{args.gnn}_nhid{args.nhid}_seed{args.seed}_node{target_node}.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print("Nettack poison loaded")
        return model.modified_adj, model.modified_features
    else:
        model = Nettack(
            model=surrogate,
            nnodes=clean_adj.shape[0],
            attack_structure=True,
            attack_features=True,
            device=device
        )
        model = model.to(device)
        model.attack(
            features=clean_features,
            adj=clean_adj,
            labels=clean_labels,
            target_node=target_node,
            n_perturbations=n_perturbations,
            verbose=False
        )

        if not os.path.exists(path):
            os.makedirs(path)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info("Nettack poison stored")
        return model.modified_adj, model.modified_features


def FGA_poison(n_perturbations, target_node):
    path = './data/poison/FGA'
    file_path = os.path.join(path, f'{args.dataset}_{args.gnn}_nhid{args.nhid}_seed{args.seed}_node{target_node}.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print("FGA poison loaded")
        return model.modified_adj
    else:
        model = FGA(surrogate, nnodes=clean_adj.shape[0], device=device)
        model = model.to(device)
        model.attack(
            ori_features=clean_features,
            ori_adj=clean_adj,
            labels=clean_labels,
            idx_train=clean_idx_train,
            target_node=target_node,
            n_perturbations=n_perturbations,
            verbose=False
        )
        
        if not os.path.exists(path):
            os.makedirs(path)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info("FGA poison stored")
        return model.modified_adj


def single_test(poisoned_adj, reduced_adj, features, labels, idx_train, idx_val, target_node):
    # test on GCN (poisoning attack)
    if args.gnn == 'gcn':
        gnn = GCN(nfeat=features.shape[1],
                nhid=args.nhid,
                nclass=labels.max().item() + 1,
                device=device)
    elif args.gnn == 'gat':
        gnn = GAT(nfeat=features.shape[1],
                nhid=args.nhid,
                nclass=labels.max().item() + 1,
                device=device)
    elif args.gnn == 'gin':
        gnn = GIN(nfeat=features.shape[1],
                nhid=args.nhid,
                nclass=labels.max().item() + 1,
                device=device)
    else:
        logger.info("Model not implemented, use gcn as default")
        gnn = GCN(nfeat=features.shape[1],
                nhid=args.nhid,
                nclass=labels.max().item() + 1,
                device=device)
    gnn = gnn.to(device)
    gnn.fit(features, reduced_adj, labels, idx_train, idx_val)
    gnn.eval()
    output = gnn.predict(features=clean_features, adj=poisoned_adj)

    acc_test = (output.argmax(1)[target_node] == clean_labels[target_node])
    
    # exclude the target node from the test sets
    idx_test_used = np.setdiff1d(clean_idx_test, target_node)
    clean_acc = accuracy(output[idx_test_used], clean_labels[idx_test_used])
    return acc_test.item(), clean_acc.item()


def coarsen_poison_test(modified_adj, modified_features, clean_labels, clean_idx_train, clean_idx_val, reduce_rate, target_node):
    success = 0
    path = './data/poison/coarsen'
    file_name = f'{path}/{args.attack}_{args.dataset}_{args.gnn}_nhid{args.nhid}_method{args.coarsening_method}_seed{args.seed}_node{target_node}_rate{reduce_rate}.pkl'
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            coarsen_adj, coarsen_feature, coarsen_labels, coarsen_idx_train, coarsen_idx_val, node_ratio, edge_ratio = pickle.load(f)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        coarsen_adj, coarsen_feature, coarsen_labels, coarsen_idx_train, coarsen_idx_val, node_ratio, edge_ratio = coarsen_poison(
            num_classes=(clean_labels.max() + 1).item(),
            num_nodes=modified_features.shape[0],
            poisoned_adj=modified_adj,
            features=modified_features, 
            labels=clean_labels, 
            idx_train=clean_idx_train, 
            idx_val=clean_idx_val, 
            coarsening_rate=reduce_rate,
            method=args.coarsening_method
        )
        with open(file_name, 'wb') as f:
            pickle.dump((coarsen_adj, coarsen_feature, coarsen_labels, coarsen_idx_train, coarsen_idx_val, node_ratio, edge_ratio), f)


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


def sparsify_poison_test(modified_adj, modified_features, clean_labels, clean_idx_train, clean_idx_val, reduce_rate, target_node):
    success = 0
    path = './data/poison/sparsify'
    file_name = f'{path}/{args.attack}_{args.dataset}_{args.gnn}_nhid{args.nhid}_method{args.sparsification_method}_seed{args.seed}_node{target_node}_rate{reduce_rate}.pkl'
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            reduced_adj, edge_ratio = pickle.load(f)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        reduced_adj, edge_ratio = sparsify(
            poisoned_adj=modified_adj,
            target_ratio=reduce_rate,
            method=args.sparsification_method,
        )
        with open(file_name, 'wb') as f:
            pickle.dump((reduced_adj, edge_ratio), f)

    acc, clean_acc = single_test(
        poisoned_adj=modified_adj,
        reduced_adj=reduced_adj,
        features=modified_features,
        labels=clean_labels,
        idx_train=clean_idx_train,
        idx_val=clean_idx_val,
        target_node=target_node
    )
    if acc == 0:
        success = 1
    
    return success, 1, edge_ratio, clean_acc


def exp(difficulty):
    temp_count = 0
    temp_clean_acc = []
    node_li = df_nodes[df_nodes['label'] == difficulty]['node']
    for target_node in tqdm(df_nodes[df_nodes['label'] == difficulty]['node']):
        # poison the graph
        n_perturbations = int(degrees[target_node])
        if args.attack == 'nettack':
            modified_adj, modified_features = Nettack_poison(n_perturbations, target_node)
        elif args.attack == 'fga':
            modified_adj = FGA_poison(n_perturbations, target_node)
            modified_features = clean_features
        else:
            logger.info("Attack not implemented, use nettack as default")
            modified_adj, modified_features = Nettack_poison(n_perturbations, target_node)
        
        # defense
        if args.defense == 'coarsening':
            success, node_ratio, edge_ratio, clean_acc = coarsen_poison_test(
                modified_adj=modified_adj,
                modified_features=modified_features,
                clean_labels=clean_labels,
                clean_idx_train=clean_idx_train,
                clean_idx_val=clean_idx_val,
                reduce_rate=reduct_rate,
                target_node=target_node
            )
        elif args.defense == 'sparsification':
            success, node_ratio, edge_ratio, clean_acc = sparsify_poison_test(
                modified_adj=modified_adj,
                modified_features=modified_features,
                clean_labels=clean_labels,
                clean_idx_train=clean_idx_train,
                clean_idx_val=clean_idx_val,
                reduce_rate=reduct_rate,
                target_node=target_node
            )
        temp_count += success
        temp_clean_acc.append(clean_acc)
    local_asr = temp_count/node_li.shape[0]
    local_acc = np.mean(temp_clean_acc)
    logger.info(f"Difficulty: {difficulty}, ASR: {local_asr}, Clean Acc: {local_acc}")

    return local_asr, temp_clean_acc, temp_count, node_ratio, edge_ratio


if __name__ == "__main__":
    args = parse_args()
    if args.defense == 'coarsening':
        log_path = f'{args.attack}_{args.dataset}_{args.defense}_{args.coarsening_method}_{args.gnn}.log'
    elif args.defense == 'sparsification':
        log_path = f'{args.attack}_{args.dataset}_{args.defense}_{args.sparsification_method}_{args.gnn}.log'
    else:
        raise ValueError("Defense method not implemented")
    logger = setup_logger(args, path=log_path)
    device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

    # Load dataset
    data = Dataset(root='./data', name=args.dataset)
    clean_adj, clean_features, clean_labels = data.adj, data.features, data.labels
    num_classes = (clean_labels.max() + 1).item()
    if args.dataset == 'Pubmed':
        # convert adj and features from scipy.sparse._arrays.csr_array to csr_matrix
        clean_adj = csr_matrix(clean_adj)
        clean_features = csr_matrix(clean_features)
    clean_idx_train, clean_idx_val, clean_idx_test = data.idx_train, data.idx_val, data.idx_test

    logger.info(f'features: {clean_features.shape} {type(clean_features)}')
    logger.info(f'adj: {clean_adj.shape} {type(clean_adj)}')
    logger.info(f'clean_labels: {clean_labels.shape} {type(clean_labels)}')
    logger.info(f'training nodes: {clean_idx_train.shape} {type(clean_idx_train)}')
    logger.info(f'validation nodes: {clean_idx_val.shape} {type(clean_idx_val)}')
    logger.info(f'test nodes: {clean_idx_test.shape} {type(clean_idx_test)}')
    logger.info(f'num_classes: {num_classes}')
    degrees = clean_adj.sum(0).A1

    results = TargettedResultDAO()
    for seed in range(1, 6):
        logger.info("=" * 10 + f" Seed {seed} " + "=" * 10)
        args.seed = seed
        logger.info(args)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # store surrogate model
        surrogate = None

        # store nodes
        df_nodes = store_nodes(clean_labels, clean_idx_test, surrogate, device)


        reduce_ratio_li = np.linspace(0.1, 1.0, args.ratio_number)
        for reduct_rate in reduce_ratio_li:
            cnt = 0
            clean_acc = []
            node_ratio_li = []
            edge_ratio_li = []
            temp_asr_li = []
            for difficulty in ['hard', 'middle', 'easy']:
                temp_asr, temp_acc_li, temp_count, node_ratio, edge_ratio = exp(difficulty)
                cnt += temp_count
                clean_acc += temp_acc_li
                node_ratio_li.append(node_ratio)
                edge_ratio_li.append(edge_ratio)
                temp_asr_li.append(temp_asr)

            overall_asr = cnt/df_nodes.shape[0]
            overall_acc = np.mean(clean_acc)

            results.append(
                seed=args.seed,
                dataset=args.dataset,
                method=args.coarsening_method,
                rate=reduct_rate,
                misclassification_rate=overall_asr,
                hard_asr=temp_asr_li[0],
                middle_asr=temp_asr_li[1],
                easy_asr=temp_asr_li[2],
                clean_acc_avg=overall_acc,
                node_ratio_avg=np.mean(node_ratio_li),
                edge_ratio_avg=np.mean(edge_ratio_li),
                node_ratio_std=np.std(node_ratio_li),
                edge_ratio_std=np.std(edge_ratio_li),
                clean_acc_std=np.std(clean_acc)
            )
    path = f'./results/{datetime.now().strftime("%b_%d")}'
    if not os.path.exists(path):
        os.makedirs(path)
    if args.defense == 'coarsening':
        file_name = f'{path}/{args.dataset}_{args.attack}_{args.defense}_{args.coarsening_method}_{args.gnn}.csv'
    elif args.defense == 'sparsification':
        file_name = f'{path}/{args.dataset}_{args.attack}_{args.defense}_{args.sparsification_method}_{args.gnn}.csv'
    results.export(file_name)