import torch
import numpy as np


def compute_lambda(adj, idx_train, idx_test):
    num_all = adj.sum().item() / 2
    train_train = adj[idx_train][:, idx_train].sum().item() / 2
    test_test = adj[idx_test][:, idx_test].sum().item() / 2
    train_test = num_all - train_train - test_test
    return train_train / num_all, train_test / num_all, test_test / num_all


def heuristic_attack(adj, labels, n_perturbations, idx_train, idx_unlabeled, lambda_1, lambda_2):
    """
    labels: torch.LongTensor
    """
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_unlabeled)
    degree = adj.sum(dim=1).to('cpu')
    canditate_train_idx = idx_train[degree[idx_train] < (int(degree.mean()) + 1)]
    candidate_test_idx = idx_test[degree[idx_test] < (int(degree.mean()) + 1)]
    #     candidate_test_idx = idx_test
    perturbed_adj = adj.clone()
    cnt = 0
    train_ratio = lambda_1 / (lambda_1 + lambda_2)
    n_train = int(n_perturbations * train_ratio)
    n_test = n_perturbations - n_train
    while cnt < n_train:
        node_1 = np.random.choice(canditate_train_idx, 1)
        node_2 = np.random.choice(canditate_train_idx, 1)
        if labels[node_1] != labels[node_2] and adj[node_1, node_2] == 0:
            perturbed_adj[node_1, node_2] = 1
            perturbed_adj[node_2, node_1] = 1
            cnt += 1

    cnt = 0
    while cnt < n_test:
        node_1 = np.random.choice(canditate_train_idx, 1)
        node_2 = np.random.choice(candidate_test_idx, 1)
        if labels[node_1] != labels[node_2] and perturbed_adj[node_1, node_2] == 0:
            perturbed_adj[node_1, node_2] = 1
            perturbed_adj[node_2, node_1] = 1
            cnt += 1
    return perturbed_adj