import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix
from graph_coarsening.coarsening_utils import *
import numpy as np


def extract_components(H):
        if H.A.shape[0] != H.A.shape[1]:
            H.logger.error('Inconsistent shape to extract components. '
                           'Square matrix required.')
            return None

        if H.is_directed():
            raise NotImplementedError('Directed graphs not supported yet.')

        graphs = []

        visited = np.zeros(H.A.shape[0], dtype=bool)

        while not visited.all():
            stack = set([np.nonzero(~visited)[0][0]])
            comp = []

            while len(stack):
                v = stack.pop()
                if not visited[v]:
                    comp.append(v)
                    visited[v] = True

                    stack.update(set([idx for idx in H.A[v, :].nonzero()[1]
                                      if not visited[idx]]))

            comp = sorted(comp)
            G = H.subgraph(comp)
            G.info = {'orig_idx': comp}
            graphs.append(G)

        return graphs


def coarsening(poison_edge_index, coarsening_ratio, coarsening_method):
    G = gsp.graphs.Graph(to_dense_adj(poison_edge_index.cpu())[0])
    components = extract_components(G)
    # print('the number of subgraphs is', len(components))
    candidate = sorted(components, key=lambda x: len(x.info['orig_idx']), reverse=True)
    number = 0
    C_list=[]  # coarsening matrix
    Gc_list=[]  # coarsened graph
    while number < len(candidate):
        H = candidate[number]
        if len(H.info['orig_idx']) > 10:
            C, Gc, Call, Gall = coarsen(H, r=coarsening_ratio, method=coarsening_method)
            C_list.append(C)
            Gc_list.append(Gc)
        number += 1
    return candidate, C_list, Gc_list


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]



def load_data_mapping(n_classes, candidate, C_list, Gc_list, poison_x, poison_labels, bkd_tn_nodes, idx_val):
    
    N = poison_x.shape[0]
    train_mask = torch.zeros(N, dtype=torch.bool).cpu()
    val_mask = torch.zeros(N, dtype=torch.bool).cpu()
    train_mask[bkd_tn_nodes] = True
    val_mask[idx_val] = True
    mask = train_mask + val_mask

    poison_labels = poison_labels.cpu()
    labels = torch.zeros(N, dtype=poison_labels.dtype)
    labels[:len(poison_labels)] = poison_labels
    features = poison_x.cpu()

    coarsen_node = 0
    number = 0
    coarsen_row = None
    coarsen_col = None
    coarsen_features = torch.Tensor([])
    coarsen_train_labels = torch.Tensor([])
    coarsen_train_mask = torch.Tensor([]).bool()
    coarsen_val_labels = torch.Tensor([])
    coarsen_val_mask = torch.Tensor([]).bool()
    coarsen_labels = torch.Tensor([])
    mapping = {}

    while number < len(candidate):
        H = candidate[number]
        keep = H.info['orig_idx']
        # keep = [num for num in keep if num <= n_nodes-1]
        H_features = features[keep]
        H_labels = labels[keep]
        H_train_mask = train_mask[keep]
        H_val_mask = val_mask[keep]
        H_mask = mask[keep]
        

        if len(H.info['orig_idx']) > 10 and torch.sum(H_train_mask)+torch.sum(H_val_mask) > 0:
            train_labels = one_hot(H_labels, n_classes)
            train_labels[~H_train_mask] = torch.Tensor([0 for _ in range(n_classes)])
            val_labels = one_hot(H_labels, n_classes)
            val_labels[~H_val_mask] = torch.Tensor([0 for _ in range(n_classes)])
            whole_labels = one_hot(H_labels, n_classes)
            whole_labels[~H_mask] = torch.Tensor([0 for _ in range(n_classes)])

            C = C_list[number]
            Gc = Gc_list[number]

            new_train_mask = torch.BoolTensor(np.sum(C.dot(train_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(train_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_train_mask[mix_mask > 1] = False

            new_val_mask = torch.BoolTensor(np.sum(C.dot(val_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(val_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_val_mask[mix_mask > 1] = False

            coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, torch.argmax(torch.FloatTensor(C.dot(val_labels)), dim=1).float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, new_val_mask], dim=0)
            coarsen_labels = torch.cat([coarsen_labels, torch.argmax(torch.FloatTensor(C.dot(whole_labels)), dim=1).float()], dim=0)


            if coarsen_row is None:
                coarsen_row = Gc.W.tocoo().row
                coarsen_col = Gc.W.tocoo().col
            else:
                current_row = Gc.W.tocoo().row + coarsen_node
                current_col = Gc.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            # get the coarsened node index
            coarsened_idx = C.argmax(axis=0).A1  # the index of the max value in each column
            candidate_mapping = {original_idx: coarsened_idx[i]+coarsen_node for i, original_idx in enumerate(keep)}
            mapping.update(candidate_mapping)
            coarsen_node += Gc.W.shape[0]

        elif torch.sum(H_train_mask)+torch.sum(H_val_mask)>0:

            coarsen_features = torch.cat([coarsen_features, H_features], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, H_labels.float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, H_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, H_labels.float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, H_val_mask], dim=0)
            coarsen_labels = torch.cat([coarsen_labels, H_labels.float()], dim=0)

            if coarsen_row is None:
                raise Exception('The graph does not need coarsening.')
            else:
                current_row = H.W.tocoo().row + coarsen_node
                current_col = H.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            candidate_mapping = {original_idx: i+coarsen_node for i, original_idx in enumerate(keep)}
            mapping.update(candidate_mapping)
            coarsen_node += H.W.shape[0]
        number += 1

    # print('the size of coarsen graph features:', coarsen_features.shape)

    coarsen_edge = torch.LongTensor([coarsen_row, coarsen_col])
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()
    coarsen_labels = coarsen_labels.long()

    return coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, coarsen_labels, mapping


def load_data(num_classes, candidate, C_list, Gc_list, poison_x, poison_labels, idx_train, idx_val, verbose=True):
    n_classes = num_classes

    N = poison_x.shape[0]
    train_mask = torch.zeros(N, dtype=torch.bool).cpu()
    val_mask = torch.zeros(N, dtype=torch.bool).cpu()
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    mask = train_mask + val_mask

    poison_labels = poison_labels.cpu()
    labels = torch.zeros(N, dtype=poison_labels.dtype)
    labels[:len(poison_labels)] = poison_labels
    features = poison_x.cpu()

    coarsen_node = 0
    number = 0
    coarsen_row = None
    coarsen_col = None
    coarsen_features = torch.Tensor([])
    coarsen_train_labels = torch.Tensor([])
    coarsen_train_mask = torch.Tensor([]).bool()
    coarsen_val_labels = torch.Tensor([])
    coarsen_val_mask = torch.Tensor([]).bool()
    coarsen_labels = torch.Tensor([])
    

    while number < len(candidate):
        H = candidate[number]
        keep = H.info['orig_idx']
        # keep = [num for num in keep if num <= n_nodes-1]
        H_features = features[keep]
        H_labels = labels[keep]
        H_train_mask = train_mask[keep]
        H_val_mask = val_mask[keep]
        H_mask = mask[keep]
        

        if len(H.info['orig_idx']) > 10 and torch.sum(H_train_mask)+torch.sum(H_val_mask) > 0:
            train_labels = one_hot(H_labels, n_classes)
            train_labels[~H_train_mask] = torch.Tensor([0 for _ in range(n_classes)])
            val_labels = one_hot(H_labels, n_classes)
            val_labels[~H_val_mask] = torch.Tensor([0 for _ in range(n_classes)])
            whole_labels = one_hot(H_labels, n_classes)
            whole_labels[~H_mask] = torch.Tensor([0 for _ in range(n_classes)])

            C = C_list[number]
            Gc = Gc_list[number]

            new_train_mask = torch.BoolTensor(np.sum(C.dot(train_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(train_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_train_mask[mix_mask > 1] = False

            new_val_mask = torch.BoolTensor(np.sum(C.dot(val_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(val_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_val_mask[mix_mask > 1] = False

            coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, torch.argmax(torch.FloatTensor(C.dot(val_labels)), dim=1).float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, new_val_mask], dim=0)
            coarsen_labels = torch.cat([coarsen_labels, torch.argmax(torch.FloatTensor(C.dot(whole_labels)), dim=1).float()], dim=0)


            if coarsen_row is None:
                coarsen_row = Gc.W.tocoo().row
                coarsen_col = Gc.W.tocoo().col
            else:
                current_row = Gc.W.tocoo().row + coarsen_node
                current_col = Gc.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += Gc.W.shape[0]

        elif torch.sum(H_train_mask)+torch.sum(H_val_mask)>0:

            coarsen_features = torch.cat([coarsen_features, H_features], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, H_labels.float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, H_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, H_labels.float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, H_val_mask], dim=0)
            coarsen_labels = torch.cat([coarsen_labels, H_labels.float()], dim=0)

            if coarsen_row is None:
                raise Exception('The graph does not need coarsening.')
            else:
                current_row = H.W.tocoo().row + coarsen_node
                current_col = H.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += H.W.shape[0]
        number += 1

    # print('the size of coarsen graph features:', coarsen_features.shape)

    coarsen_edge = torch.LongTensor([coarsen_row, coarsen_col])
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()
    coarsen_labels = coarsen_labels.long()

    coarsen_val_mask = coarsen_val_mask & ~coarsen_train_mask
    coarsen_val_labels = coarsen_labels[coarsen_val_mask]

    if verbose:
        print('Coarsening finished.')
        print('Number of nodes:', coarsen_features.shape[0])
        print('Number of edges:', coarsen_edge.shape[1])
        print('Number of training nodes:', torch.sum(coarsen_train_mask).item())
        print('Number of validation nodes:', torch.sum(coarsen_val_mask).item())
        print('Number of training + validation nodes:', torch.sum(torch.bitwise_or(coarsen_train_mask, coarsen_val_mask)).item())

    return coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, coarsen_labels


def coarsen_poison(num_classes, num_nodes, poisoned_adj, features, labels, idx_train, idx_val, coarsening_rate, method=None):
    # convert adj sparse matrix to edge_index
    poison_edge_index = from_scipy_sparse_matrix(poisoned_adj)[0]
    # convert features from csr_matrix to tensor
    poisoned_x = torch.tensor(features.todense(), dtype=torch.float)
    poison_labels = torch.tensor(labels, dtype=torch.long)
    if method is not None or coarsening_rate < 1.0:
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
        node_ratio = coarsen_adj.shape[0] / num_nodes

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



def coarsen_poison_mapping(num_classes, num_nodes, poisoned_adj, features, labels, idx_train, idx_val, coarsening_rate, method=None):
    # convert adj sparse matrix to edge_index
    poison_edge_index = from_scipy_sparse_matrix(poisoned_adj)[0]
    # convert features from csr_matrix to tensor
    poisoned_x = torch.tensor(features.todense(), dtype=torch.float)
    poison_labels = torch.tensor(labels, dtype=torch.long)
    if method is not None or coarsening_rate < 1.0:
        candidate, C_list, Gc_list = coarsening(poison_edge_index, 1-coarsening_rate, method)
        coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, coarsen_labels, mapping = load_data_mapping(num_classes, candidate, C_list, Gc_list, poisoned_x, poison_labels, idx_train, idx_val)
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
        node_ratio = coarsen_adj.shape[0] / num_nodes

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
    
    return coarsen_adj, coarsen_features, coarsen_labels, coarsen_poison_idx_train, coarsen_poison_idx_val, node_ratio, edge_ratio, mapping
