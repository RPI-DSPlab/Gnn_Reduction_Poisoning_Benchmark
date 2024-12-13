from deeprobust.graph.defense import GCN
from deeprobust.graph.defense import RGCN
from model.gnn_guard import GNNGuard
from model.median_gcn import MedianGCN
from model.sage import GraphSage
from model.gat import GAT

from deeprobust.graph.utils import accuracy
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import torch


def calc_acc(args, train_adj, train_feature, train_label, 
                train_idx_train, train_idx_val, 
                predict_adj, predict_feature, predict_label, predict_idx_test,
                test_gnn='gcn', device=None):
    if test_gnn == 'gcn':
        gcn = GCN(nfeat=train_feature.shape[1],
              nhid=args.hidden,
              nclass=train_label.max().item() + 1,
              dropout=args.dropout, device=device)
        gcn = gcn.to(device)
        gcn.fit(train_feature, train_adj, train_label, train_idx_train, train_idx_val)

        gcn.eval()
        output = gcn.predict(predict_feature, predict_adj)
    elif test_gnn == 'gnnguard':
        gcn = GNNGuard(nfeat=train_feature.shape[1],
              nhid=args.hidden,
              nclass=train_label.max().item() + 1,
              dropout=args.dropout, device=device)
        gcn = gcn.to(device)
        gcn.fit(train_feature, train_adj, train_label, train_idx_train, train_idx_val)

        gcn.eval()
        output = gcn.predict(predict_feature, predict_adj)
    elif test_gnn == 'rgcn':
        if train_adj.shape[0] != predict_adj.shape[0]:
            acc_test = -1
            return acc_test
        gcn = RGCN(nnodes=train_adj.shape[0], 
                   nfeat=train_feature.shape[1], 
                   nclass=train_label.max()+1,
                   nhid=32, device=device)
        gcn = gcn.to(device)
        gcn.fit(train_feature, train_adj, train_label, train_idx_train, train_idx_val)

        gcn.eval()
        output = gcn.predict(predict_feature, predict_adj)
    elif test_gnn == 'median':
        features_tensor = torch.FloatTensor(train_feature.todense()).to(device)
        edge_index = from_scipy_sparse_matrix(train_adj)[0]
        edge_index = edge_index.to(device)
        labels_tensor = torch.LongTensor(train_label).to(device)

        gcn = MedianGCN(nfeat=train_feature.shape[1],
                    nhid=16,
                    nclass=train_label.max().item() + 1,
                    dropout=args.dropout, device=device)
        gcn = gcn.to(device)
        gcn.fit(features_tensor, edge_index, labels_tensor, train_idx_train, train_idx_val)

        features_tensor = torch.FloatTensor(predict_feature.todense()).to(device)
        edge_index, edge_weight = from_scipy_sparse_matrix(predict_adj)
        edge_index = edge_index.to(device)

        gcn.eval()
        output = gcn(features_tensor, edge_index)
    elif test_gnn == 'sage':
        features_tensor = torch.FloatTensor(train_feature.todense()).to(device)
        edge_index, edge_weight = from_scipy_sparse_matrix(train_adj)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        labels_tensor = torch.LongTensor(train_label).to(device)

        gcn = GraphSage(nfeat=train_feature.shape[1],
                    nhid=args.hidden,
                    nclass=train_label.max().item() + 1,
                    dropout=args.dropout, device=device)
        gcn = gcn.to(device)
        gcn.fit(features_tensor, edge_index, edge_weight, labels_tensor, train_idx_train, train_idx_val)

        features_tensor = torch.FloatTensor(predict_feature.todense()).to(device)
        edge_index, edge_weight = from_scipy_sparse_matrix(predict_adj)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        gcn.eval()
        output = gcn(features_tensor, edge_index, edge_weight)
    elif test_gnn == 'gat':
        features_tensor = torch.FloatTensor(train_feature.todense()).to(device)
        edge_index, edge_weight = from_scipy_sparse_matrix(train_adj)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        labels_tensor = torch.LongTensor(train_label).to(device)

        gcn = GAT(nfeat=train_feature.shape[1],
                    nhid=args.hidden,
                    nclass=train_label.max().item() + 1,
                    dropout=args.dropout, device=device)
        gcn = gcn.to(device)
        gcn.fit(features_tensor, edge_index, edge_weight, labels_tensor, train_idx_train, train_idx_val)

        features_tensor = torch.FloatTensor(predict_feature.todense()).to(device)
        edge_index, edge_weight = from_scipy_sparse_matrix(predict_adj)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        gcn.eval()
        output = gcn(features_tensor, edge_index, edge_weight)
        print('gat')
    else:
        gcn = GCN(nfeat=train_feature.shape[1],
                    nhid=args.hidden,
                    nclass=train_label.max().item() + 1,
                    dropout=args.dropout, device=device)
        gcn = gcn.to(device)
        gcn.fit(train_feature, train_adj, train_label, train_idx_train, train_idx_val)

        gcn.eval()
        output = gcn.predict(predict_feature, predict_adj)
    
    acc_test = accuracy(output[predict_idx_test], predict_label[predict_idx_test])
    return acc_test.item()