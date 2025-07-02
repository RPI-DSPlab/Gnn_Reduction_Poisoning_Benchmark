import torch
from deeprobust.graph.defense import GCN
from deeprobust.graph.defense import RGCN
from model.gnn_guard import GNNGuard
from model.median_gcn import MedianGCN
from model.sage import GraphSage
from model.gat import GAT
from model.strg import STRG

from torch_geometric.utils.convert import from_scipy_sparse_matrix

def train_gnn(adj, features, labels, idx_train, idx_val, args, test_gnn='gcn', device='cpu'):
    if test_gnn == 'gcn':
        gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)
        gcn = gcn.to(device)
        gcn.fit(features, adj, labels, idx_train, idx_val)
    elif test_gnn == 'strg':
        gcn = STRG(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
        gcn = gcn.to(device)
        gcn.fit(features, adj, labels, idx_train, idx_val)
    elif test_gnn == 'gnnguard':
        gcn = GNNGuard(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
        gcn = gcn.to(device)
        gcn.fit(features, adj, labels, idx_train, idx_val)
    elif test_gnn == 'rgcn':
        gcn = RGCN(nnodes=adj.shape[0], 
                   nfeat=features.shape[1], 
                   nclass=labels.max()+1,
                   nhid=32, device=device)
        gcn = gcn.to(device)
        gcn.fit(features, adj, labels, idx_train, idx_val)
    elif test_gnn == 'median':
        features_tensor = torch.FloatTensor(features.todense()).to(device)
        edge_index = from_scipy_sparse_matrix(adj)[0]
        edge_index = edge_index.to(device)
        labels_tensor = torch.LongTensor(labels).to(device)

        gcn = MedianGCN(nfeat=features.shape[1],
                    nhid=16,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout, device=device)
        gcn = gcn.to(device)
        gcn.fit(features_tensor, edge_index, labels_tensor, idx_train, idx_val)
    elif test_gnn == 'sage':
        features_tensor = torch.FloatTensor(features.todense()).to(device)
        edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        labels_tensor = torch.LongTensor(labels).to(device)

        gcn = GraphSage(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout, device=device)
        gcn = gcn.to(device)
        gcn.fit(features_tensor, edge_index, edge_weight, labels_tensor, idx_train, idx_val)
    elif test_gnn == 'gat':
        features_tensor = torch.FloatTensor(features.todense()).to(device)
        edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        labels_tensor = torch.LongTensor(labels).to(device)

        gcn = GAT(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout, device=device)
        gcn = gcn.to(device)
        gcn.fit(features_tensor, edge_index, edge_weight, labels_tensor, idx_train, idx_val)
    else:
        raise ValueError('Invalid GNN model name')
    return gcn