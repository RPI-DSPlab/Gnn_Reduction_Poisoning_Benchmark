from deeprobust.graph.global_attack import PRBCD
from deeprobust.graph.global_attack import Metattack
from deeprobust.graph.global_attack import DICE
from deeprobust.graph.global_attack import NodeEmbeddingAttack
from deeprobust.graph.global_attack import PGDAttack
from attack.STRG import compute_lambda, heuristic_attack
from attack.GraD import GraD

from scipy.sparse import csr_matrix
from deeprobust.graph.utils import *

import torch
import numpy as np


def poison(args, attack_name, adj, features, labels, surrogate, data, 
           idx_train, idx_test, idx_unlabeled, 
           perturbations, ptb_rate=0.05, lambda_=0, device=None):
    """
    Test Poisoned Accuracy
    """
    if attack_name == 'prbcd':
        # PRBCD attack
        agent = PRBCD(data, device=device)
        edge_index, edge_weight = agent.attack(ptb_rate=ptb_rate)
        edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)
        modified_adj = torch.zeros((data.num_nodes, data.num_nodes)).to(device)
        modified_adj[edge_index[0], edge_index[1]] = edge_weight
        modified_adj[edge_index[1], edge_index[0]] = edge_weight

        modified_adj = csr_matrix(modified_adj.cpu().detach().numpy())
    elif attack_name == 'mettack':
        # Mettack attack
        model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=False, device=device, lambda_=lambda_)
        model = model.to(device)
        model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
        modified_adj = model.modified_adj
        modified_adj = modified_adj.detach().cpu().numpy()
        modified_adj = csr_matrix(modified_adj)
    elif attack_name == 'dice':
        # DICE
        model = DICE()
        n_perturbations = int(ptb_rate * (adj.sum()//2))
        model = model.to(device)
        model.attack(adj, labels, n_perturbations)
        modified_adj = model.modified_adj
    elif attack_name == 'nea':
        model = NodeEmbeddingAttack()
        model.attack(adj,
                    n_perturbations=perturbations,
                    attack_type='add',
                    n_candidates=perturbations             
        )
        modified_adj = model.modified_adj
    elif attack_name == 'pgd':
        # Here for the labels we need to replace it with predicted ones
        fake_labels = surrogate.predict(features, adj)
        fake_labels = torch.argmax(fake_labels, 1).cpu()
        # Besides, we need to add the idx into the whole process
        idx_fake = np.concatenate([idx_train,idx_test])
        idx_others = list(set(np.arange(len(labels))) - set(idx_train))
        fake_labels = torch.cat([torch.LongTensor(labels), fake_labels[idx_others]])
        model = PGDAttack(model=surrogate, nnodes=adj.shape[0], loss_type='CE', device=device)
        model = model.to(device)
        model.attack(torch.FloatTensor(features.todense()), torch.FloatTensor(adj.todense()), fake_labels, idx_fake, perturbations)
        modified_adj = model.modified_adj
        modified_adj = modified_adj.detach().cpu().numpy()
        modified_adj = csr_matrix(modified_adj)
    elif attack_name == 'strg':
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj = adj.to_dense()
        adj = adj.to(device)
        lambda_1, lambda_2, lambda_3 = compute_lambda(adj, idx_train, idx_unlabeled)
        modified_adj = heuristic_attack(adj, labels, perturbations, idx_train, idx_unlabeled, lambda_1, lambda_2)

        modified_adj = csr_matrix(modified_adj.detach().cpu().numpy())
    elif attack_name == 'grad':
        attack_model = GraD(nfeat=features.shape[1], hidden_sizes=[args.hidden],
                        nnodes=adj.shape[0], nclass=labels.max().item()+1, dropout=0.5,
                        train_iters=100, attack_features=False, lambda_=0, device=device, args=args)
        attack_model = attack_model.to(device)
        
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj = adj.to(device)
        features = torch.FloatTensor(features.todense()).to(device)
        labels = torch.LongTensor(labels).to(device)
        
        modified_adj = attack_model(features, adj, labels, idx_train,
                            idx_unlabeled, perturbations)
        modified_adj = csr_matrix(modified_adj.detach().cpu().numpy())
    
    return modified_adj