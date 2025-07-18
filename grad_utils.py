import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
import torch.sparse as ts

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_npz(file_name, is_sparse=True):
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    with np.load(file_name) as loader:
        
        if is_sparse:

            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                        loader['adj_indptr']), shape=loader['adj_shape'])

            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                             loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                features = None

            labels = loader.get('labels')

        else:
            adj = loader['adj_data']

            if 'attr_data' in loader:
                features = loader['attr_data']
            else:
                features = None

            labels = loader.get('labels')

    return adj, features, labels

def get_adj(dataset, require_lcc=True):
    print('reading %s...' % dataset)
    _A_obs, _X_obs, _z_obs = load_npz(r'data/%s.npz' % dataset)
    _A_obs = _A_obs + _A_obs.T
    _A_obs = _A_obs.tolil()
    _A_obs[_A_obs > 1] = 1

    if _X_obs is None:
        _X_obs = np.eye(_A_obs.shape[0])

    require_lcc= False
    if require_lcc:
        lcc = largest_connected_components(_A_obs)

        _A_obs = _A_obs[lcc][:,lcc]
        _X_obs = _X_obs[lcc]
        _z_obs = _z_obs[lcc]

        assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    _A_obs.setdiag(0)
    _A_obs = _A_obs.astype("float32").tocsr()
    _A_obs.eliminate_zeros()

    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"

    return _A_obs, _X_obs, _z_obs

def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    Parameters
    """
    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep



def load_data(dataset="cora", val_size=0.1, test_size=0.1):

    print('Loading {} dataset...'.format(dataset))
    adj, features, labels = get_adj(dataset)
    features = sp.csr_matrix(features, dtype=np.float32)

    return adj, features, labels


def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False):
    if preprocess_adj == True:
        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

    if preprocess_feature:
        features = normalize_feature(features)

    labels = torch.LongTensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features.todense()))
        adj = torch.FloatTensor(adj.todense())

    return adj, features, labels


def normalize_feature(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

def normalize_adj_tensor(adj, sparse=False):
    if sparse:
        adj = to_scipy(adj)
        mx = normalize_adj(adj.tolil())
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = torch.eye(adj.shape[0]).to(device) + adj
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def to_scipy(sparse_tensor):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    values = sparse_tensor._values()
    indices = sparse_tensor._indices()
    return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()))

def get_train_val_test(idx, train_size, val_size, test_size, stratify):

    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


def unravel_index(index, array_shape):
    rows = index // array_shape[1]
    cols = index % array_shape[1]
    return rows, cols

def likelihood_ratio_filter(node_pairs, modified_adjacency, original_adjacency, d_min, threshold=0.004):
    """
    Filter the input node pairs based on the likelihood ratio test proposed by Zügner et al. 2018, see
    https://dl.acm.org/citation.cfm?id=3220078. In essence, for each node pair return 1 if adding/removing the edge
    between the two nodes does not violate the unnoticeability constraint, and return 0 otherwise. Assumes unweighted
    and undirected graphs.
    """

    N = int(modified_adjacency.shape[0])

    original_degree_sequence = original_adjacency.sum(0)
    current_degree_sequence = modified_adjacency.sum(0)

    # Concatenate the degree sequences
    concat_degree_sequence = torch.cat((current_degree_sequence, original_degree_sequence))

    # Compute the log likelihood values of the original, modified, and combined degree sequences.
    ll_orig, alpha_orig, n_orig, sum_log_degrees_original = degree_sequence_log_likelihood(original_degree_sequence, d_min)
    ll_current, alpha_current, n_current, sum_log_degrees_current = degree_sequence_log_likelihood(
        current_degree_sequence, d_min)

    ll_comb, alpha_comb, n_comb, sum_log_degrees_combined = degree_sequence_log_likelihood(concat_degree_sequence, d_min)

    # Compute the log likelihood ratio
    current_ratio = -2 * ll_comb + 2 * (ll_orig + ll_current)

    # Compute new log likelihood values that would arise if we add/remove the edges corresponding to each node pair.

    new_lls, new_alphas, new_ns, new_sum_log_degrees = updated_log_likelihood_for_edge_changes(node_pairs,
                                                                                               modified_adjacency, d_min)

    # Combination of the original degree distribution with the distributions corresponding to each node pair.
    n_combined = n_orig + new_ns
    new_sum_log_degrees_combined = sum_log_degrees_original + new_sum_log_degrees
    alpha_combined = compute_alpha(n_combined, new_sum_log_degrees_combined, d_min)

    new_ll_combined = compute_log_likelihood(n_combined, alpha_combined, new_sum_log_degrees_combined, d_min)
    new_ratios = -2 * new_ll_combined + 2 * (new_lls + ll_orig)

    # Allowed edges are only those for which the resulting likelihood ratio measure is < than the threshold
    allowed_edges = new_ratios < threshold
    try:
        filtered_edges = node_pairs[allowed_edges.cpu().numpy().astype(np.bool)]
    except:
        filtered_edges = node_pairs[allowed_edges.numpy().astype(np.bool)]

    allowed_mask = torch.zeros(modified_adjacency.shape)
    allowed_mask[filtered_edges.T] = 1
    allowed_mask += allowed_mask.t()
    return allowed_mask, current_ratio


def degree_sequence_log_likelihood(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.

    """
    # Determine which degrees are to be considered, i.e. >= d_min.

    D_G = degree_sequence[(degree_sequence >= d_min.item())]
    try:
        sum_log_degrees = torch.log(D_G).sum()
    except:
        sum_log_degrees = np.log(D_G).sum()
    n = len(D_G)

    alpha = compute_alpha(n, sum_log_degrees, d_min)
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)
    return ll, alpha, n, sum_log_degrees

def updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):

    # For each node pair find out whether there is an edge or not in the input adjacency matrix.

    edge_entries_before = adjacency_matrix[node_pairs.T]

    degree_sequence = adjacency_matrix.sum(1)

    D_G = degree_sequence[degree_sequence >= d_min.item()]
    sum_log_degrees = torch.log(D_G).sum()
    n = len(D_G)


    deltas = -2 * edge_entries_before + 1
    d_edges_before = degree_sequence[node_pairs]

    d_edges_after = degree_sequence[node_pairs] + deltas[:, None]

    # Sum the log of the degrees after the potential changes which are >= d_min
    sum_log_degrees_after, new_n = update_sum_log_degrees(sum_log_degrees, n, d_edges_before, d_edges_after, d_min)

    # Updated estimates of the Powerlaw exponents
    new_alpha = compute_alpha(new_n, sum_log_degrees_after, d_min)
    # Updated log likelihood values for the Powerlaw distributions
    new_ll = compute_log_likelihood(new_n, new_alpha, sum_log_degrees_after, d_min)

    return new_ll, new_alpha, new_n, sum_log_degrees_after


def update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    # Find out whether the degrees before and after the change are above the threshold d_min.
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = d_old * old_in_range.float()
    d_new_in_range = d_new * new_in_range.float()

    # Update the sum by subtracting the old values and then adding the updated logs of the degrees.
    sum_log_degrees_after = sum_log_degrees_before - (torch.log(torch.clamp(d_old_in_range, min=1))).sum(1) \
                                 + (torch.log(torch.clamp(d_new_in_range, min=1))).sum(1)

    # Update the number of degrees >= d_min
    new_n = n_old - (old_in_range!=0).sum(1) + (new_in_range!=0).sum(1)
    new_n = new_n.float()
    return sum_log_degrees_after, new_n

def compute_alpha(n, sum_log_degrees, d_min):

    try:
        alpha =  1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))
    except:
        alpha =  1 + n / (sum_log_degrees - n * np.log(d_min - 0.5))
    return alpha

def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    # Log likelihood under alpha
    try:
        ll = n * torch.log(alpha) + n * alpha * torch.log(d_min) + (alpha + 1) * sum_log_degrees
    except:
        ll = n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * sum_log_degrees

    return ll

def ravel_multiple_indices(ixs, shape, reverse=False):
    """
    "Flattens" multiple 2D input indices into indices on the flattened matrix, similar to np.ravel_multi_index.
    Does the same as ravel_index but for multiple indices at once.
    Parameters
    """
    if reverse:
        return ixs[:, 1] * shape[1] + ixs[:, 0]

    return ixs[:, 0] * shape[1] + ixs[:, 1]

