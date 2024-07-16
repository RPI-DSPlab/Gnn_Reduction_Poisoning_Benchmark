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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', 
                        choices=['Cora', 'Pubmed', 'Polblogs'])
    parser.add_argument('--gnn', type=str, default='gcn',
                        choices=['gcn', 'gin', 'sage'])
    parser.add_argument('--nhid', type=int, default=16, help='Hidden layer size.')
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--device_id', type=int, default=0)

    args = parser.parse_args()

    return args


def test(adj, features, surrogate):

    surrogate.fit(features, adj, clean_labels, clean_idx_train)

    surrogate.eval()
    output = surrogate.predict()
    acc_test = accuracy(output[clean_idx_test], clean_labels[clean_idx_test])


    return acc_test.item()


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


def single_test(adj, features, target_node):
    if args.gnn == 'gcn':
        poisoned_model = GCN(nfeat=features.shape[1],
                        nhid=args.nhid,
                        nclass=num_classes,
                        device=device)
    elif args.gnn == 'gat':
        poisoned_model = GAT(nfeat=features.shape[1],
                nhid=args.nhid,
                nclass=num_classes,
                device=device)
    elif args.gnn == 'gin':
        poisoned_model = GIN(nfeat=features.shape[1],
                nhid=args.nhid,
                nclass=num_classes,
                device=device)
    else:
        logger.info("Model not implemented, use gcn as default")
        poisoned_model = GCN(nfeat=features.shape[1],
                nhid=args.nhid,
                nclass=num_classes,
                device=device)
    poisoned_model = poisoned_model.to(device)
    poisoned_model.fit(
        features=features,
        adj=adj,
        labels=clean_labels,
        idx_train=clean_idx_train,
        idx_val=clean_idx_val,
        train_iters=200,
        verbose=False
    )
    output = poisoned_model.predict(features, adj)

    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == clean_labels[target_node])
    # exclude the target node from the test sets
    idx_test_used = np.setdiff1d(clean_idx_test, target_node)
    clean_acc = accuracy(output[idx_test_used], clean_labels[idx_test_used])
    return acc_test.item(), clean_acc.item()


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger(args, path=f'{args.dataset}_{args.gnn}_{args.nhid}.log')
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

    logger.info(f'adj: {clean_adj.shape} {type(clean_adj)}')
    logger.info(f'features: {clean_features.shape} {type(clean_features)}')
    logger.info(f'clean_labels: {clean_labels.shape} {type(clean_labels)}')
    logger.info(f'training nodes: {clean_idx_train.shape} {type(clean_idx_train)}')
    logger.info(f'validation nodes: {clean_idx_val.shape} {type(clean_idx_val)}')
    logger.info(f'test nodes: {clean_idx_test.shape} {type(clean_idx_test)}')
    logger.info(f'num_classes: {num_classes}')
    degrees = clean_adj.sum(0).A1

    results = {
        'seed': [],
        'fga_hard_ASR': [],
        'fga_middle_ASR': [],
        'fga_easy_ASR': [],
        'fga_overall_ASR': [],
        'fga_overall_ACC': [],
        'nettack_hard_ASR': [],
        'nettack_middle_ASR': [],
        'nettack_easy_ASR': [],
        'nettack_overall_ASR': [],
        'nettack_overall_ACC': []
    }
    for seed in range(1, 6):
        args.seed = seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print()
        print("=" * 10 + f" Seed {args.seed} " + "=" * 10)
        print()

        # store surrogate model
        surrogate = store_surrogate(args)
        print("Surrogate model stored")
        acc_test = test(clean_adj, clean_features, surrogate)
        logger.info(f"Test accuracy: {acc_test}")

        # store nodes
        df_nodes = store_nodes(clean_labels, clean_idx_test, surrogate, device)
        print("Nodes stored")

        difficulty_li = ['hard', 'middle', 'easy']

        nettack_cnt = 0
        fga_cnt = 0

        results['seed'].append(seed)
        nettack_clean_acc_li = []
        fga_clean_acc_li = []
        for difficulty in difficulty_li:
            node_li = df_nodes[df_nodes['label'] == difficulty]['node']

            temp_cnt = 0
            if args.gnn == 'gat' or args.gnn == 'gin':
                fga_asr = -1
            else:
                for target_node in tqdm(node_li):
                    n_perturbations = int(degrees[target_node])
                    modified_adj = FGA_poison(n_perturbations, target_node)

                    acc, clean_acc = single_test(modified_adj, clean_features, target_node)
                    if acc == 0:
                        fga_cnt += 1
                        temp_cnt += 1
                    fga_clean_acc_li.append(clean_acc)
                fga_asr = temp_cnt / node_li.shape[0]
                logger.info(f"FGA poison stored in {difficulty}, ASR: {fga_asr}")

            temp_cnt = 0
            for target_node in tqdm(node_li):
                n_perturbations = int(degrees[target_node])
                modified_adj, modified_features = Nettack_poison(n_perturbations, target_node)

                acc, clean_acc = single_test(modified_adj, modified_features, target_node)
                if acc == 0:
                    nettack_cnt += 1
                    temp_cnt += 1
                nettack_clean_acc_li.append(clean_acc)
            nettack_asr = temp_cnt / node_li.shape[0]
            logger.info(f"Nettack poison stored in {difficulty}, ASR: {nettack_asr}")

            results[f'fga_{difficulty}_ASR'].append(fga_asr)
            results[f'nettack_{difficulty}_ASR'].append(nettack_asr)
        if args.gnn == 'gat':
            results['fga_overall_ASR'].append(-1)
            results['fga_overall_ACC'].append(-1)
        else:
            results['fga_overall_ASR'].append(fga_cnt / df_nodes.shape[0])
            results['fga_overall_ACC'].append(np.mean(np.array(fga_clean_acc_li)))
        
        results['nettack_overall_ASR'].append(nettack_cnt / df_nodes.shape[0])
        results['nettack_overall_ACC'].append(np.mean(np.array(nettack_clean_acc_li)))
        logger.info(f"FGA overall ASR: {fga_cnt / df_nodes.shape[0]}")
        logger.info(f"Nettack overall ASR: {nettack_cnt / df_nodes.shape[0]}")
        
    
    results = pd.DataFrame(results)
    results.to_csv(f'./results/{args.dataset}_{args.gnn}_{args.nhid}.csv', index=False)



