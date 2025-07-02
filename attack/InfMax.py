import argparse
import random
# from torch.autograd.gradcheck import zero_gradients
import torch as th
import torch.nn.functional as F
from torch.nn.functional import normalize
import collections

import numpy as np
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.algorithms.centrality import betweenness_centrality as betweenness
from copy import deepcopy 
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from deeprobust.graph.utils import *

import json


def zero_gradients(x):
    if isinstance(x, th.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def pick_feature(grad, k, features):
    score = grad.sum(dim=0)
    _, indexs = th.topk(score.abs(), k)
    signs = th.zeros(features.shape[1])
    for i in indexs:
        signs[i] = score[i].sign()
    return signs, indexs


def getM(K, Prob):
    '''
    Nearly the same as function getScore. Return the random walk matrix directly rather than calculate the col sum.
    '''
    Random = Prob
    for i in range(K - 1):
        Random = th.sparse.mm(Random, Prob)
    return Random


def New_sort(alpha, M, limit, bar, g):
    '''
    New sort method
    :param alpha: an int as the threshold in cutting too large element
    :param M: M is typically the original random walk M
    :param limit: limit is typically the args.num_node
    :param bar: an int used to set the threshold of degree that can be chosen to attack
    :param g: the graph, used to calculate the out_degree of an node
    :return: a list contains the indexs of nodes that needed to be attacked.
    '''
    s = np.zeros((M.shape[0],1)) # zero vector
    res = [] # res vector

    # make those i has larger degree to -inf
    for i in range(M.shape[0]): 
        if g.degree(i) > bar:
            M[:,i] = -float("inf")
    
    # debug
    # print("New_sort(debug): alpha = ", alpha)

    # Greedyly choose the point
    for _ in range(limit):
        L = np.minimum(s+M, alpha)
        L = L.sum(axis=0)
        i = np.argmax(L)
        res.append(i)
        s = s + M[:,i].reshape(M.shape[0],1)
        M[:,i] = -float("inf")
        # delete neighbour
        # print(g.edges(i)[1])
        for neighbor in g.neighbors(i):
            M[:,neighbor] = -float("inf")
    return res


def getThrehold(g, size, threshold):
    degree = g.degree(range(size))
    Cand_degree = sorted([(degree[i], i) for i in range(size)], reverse=True)
    threshold = int(size * threshold)
    bar, _ = Cand_degree[threshold]
    return bar


def infmax_attack(model, adj, features, labels, idx_train, n_perturbation, norm_length=10):
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
    adj_tensor = adj_tensor.to_dense()
    adj_tensor = adj_tensor.to(model.device)

    features_tensor = th.FloatTensor(features.todense()).to(model.device)
    labels_tensor = th.LongTensor(labels).to(model.device)

    Prob = normalize(adj_tensor, p=1, dim=1)
    Important_matrix = getM(4, Prob)

    num_node = n_perturbation
    g = nx.from_scipy_sparse_array(adj)
    size = labels.shape[0]
    bar = getThrehold(g, size, 0.1)
    
    RWCS_NEW = New_sort(0.01, Important_matrix.detach().cpu().numpy(), num_node, bar, g)

    features_tensor.requires_grad_(True)
    model.eval()
    logits = model(features_tensor, adj_tensor)
    loss = F.nll_loss(logits[idx_train], labels_tensor[idx_train])
    zero_gradients(features_tensor)
    loss.backward(retain_graph=True)
    grad = features_tensor.grad.detach().clone()
    signs, indexs = pick_feature(grad, 74, features_tensor)
    features_tensor.requires_grad_(False)

    targets = RWCS_NEW
    for target in targets:
        for index in indexs:
            features_tensor[target][index] += norm_length * signs[index]
    # for target in targets:
    #     for index in indexs:
    #         features_tensor[target][index] -= norm_length * signs[index]
    
    result = features_tensor.cpu().detach().numpy()
    result = sp.csr_matrix(result)
    return result