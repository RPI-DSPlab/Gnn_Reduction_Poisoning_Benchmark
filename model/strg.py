import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
# from sklearn.metrics import f1_score
import numpy as np

import torch.utils.data as Data


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden=1024, n_layers=2, drop_out=0.5, batchnorm=True):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, n_hidden))
        if batchnorm:
            self.layers.append(nn.BatchNorm1d(n_hidden))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(drop_out))
        for _ in range(n_layers - 2):
            self.layer.append(nn.Linear(n_hidden, n_hidden))
            if batchnorm:
                self.layers.append(nn.BatchNorm1d(n_hidden))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(drop_out))
        self.layers.append(nn.Linear(n_hidden, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train_MLP(model, epochs, optimizer, train_loader, val_loader, loss, device, verbose=True):
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            if x.shape[0] == 1:
                continue
            y = y.to(device)
            # print('X:', x.shape)
            output = model(x)
            optimizer.zero_grad()
            l = loss(output, y)
            l.backward()
            optimizer.step()
        n_acc = 0
        loss_total = 0
        n = 0
        best_acc_val = 0
        best_loss_val = 0
        model.eval()
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            n += len(y)
            acc = (pred == y).sum().item()
            n_acc += acc
            l = loss(output, y)
            loss_total += l
        acc_total = n_acc / n
        val_loss = loss_total /n
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
        if acc_total > best_acc_val:
            best_acc_val = acc_total
            weights = deepcopy(model.state_dict())
        if verbose:
            if epoch % 10 == 0:
                print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}"
                      .format(epoch, l.item(), acc_total))
    model.load_state_dict(weights)


def get_psu_labels(logits, pseudo_labels, idx_train, idx_test, k=30, append_idx=True):
    # idx_train = np.array([], dtype='int32')
    if append_idx:
        idx_train = idx_train
    else:
        idx_train = np.array([], dtype='int64')
    pred_labels = torch.argmax(logits, dim=1)
    pred_labels_test = pred_labels[idx_test]
    for label in range(pseudo_labels.max().item() + 1):
        idx_label = idx_test[pred_labels_test==label]
        logits_label = logits[idx_label][:, label]
        if len(logits_label) > k:
            _, idx_topk = torch.topk(logits_label, k)
        else:
            idx_topk = np.arange(len(logits_label))
        idx_topk = idx_label[idx_topk]
        pseudo_labels[idx_topk] = label
        idx_train = np.concatenate((idx_train, idx_topk))
    return idx_train, pseudo_labels


class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class STRG(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None):

        super(STRG, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def forward(self, x, adj):
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features

        # train a surrogate MLP model to generate pseudo labels
        features_tensor = features.to_dense() if utils.is_sparse_tensor(features) else features
        labels_tensor = labels.to_dense() if utils.is_sparse_tensor(labels) else labels
        epochs = 200
        n_hidden = 1024
        # dropout = 0.5
        weight_decay = 5e-4
        lr = 1e-2
        loss = nn.CrossEntropyLoss()
        n_class = labels.max().item() + 1
        batch_size = 64
        train_dataset = Data.TensorDataset(features_tensor[idx_train], labels_tensor[idx_train])
        train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = Data.TensorDataset(features_tensor[idx_val], labels_tensor[idx_val])
        val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model = MLP(features_tensor.shape[1], n_class, n_hidden)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_MLP(model, epochs, optimizer, train_loader, val_loader, loss, self.device)
        # print('MLP model accuracy: {:.4f}'.format(acc))
        logits = model(features_tensor).detach().cpu()

        pseudo_labels = labels.clone().detach().cpu()
        idx_train, pseudo_labels = get_psu_labels(logits, pseudo_labels, idx_train, idx_val, k=80, append_idx=False)
        self.labels = pseudo_labels.to(self.device)
        print('pseudo labels:', self.labels.shape)

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            # def eval_class(output, labels):
            #     preds = output.max(1)[1].type_as(labels)
            #     return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro') + \
            #         f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # perf_sum = eval_class(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)



