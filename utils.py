from deeprobust.graph.utils import classification_margin
from deeprobust.graph.defense import GCN
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI,CitationFull,Coauthor

from datetime import datetime
import torch
import logging
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', 
                        choices=['Cora', 'Pubmed', 'Flickr', 'Polblogs'])
    parser.add_argument('--gnn', type=str, default='gcn',
                        choices=['gcn', 'gat', 'sage'])
    parser.add_argument('--attack', type=str, default='fga',
                        choices=['fga', 'nettack'])
    parser.add_argument('--nhid', type=int, default=16, help='Hidden layer size.')
    parser.add_argument('--defense', type=str, default='coarsening', 
                        choices=['coarsening', 'sparsification', 'sparsification_svd', 'jaccard', 'svd', 'rgcn', 'median', 'airgnn'])
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods',
                        choices=['variation_neighborhoods_degree', 'variation_neighborhoods','variation_edges_degree','variation_edges', 'variation_cliques_degree', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'kron'],
                        help="Method of coarsening")
    parser.add_argument('--sparsification_method', type=str, default='random_node_edge',
                        choices=['random_node_edge', 'random_edge', 'local_degree', 'forest_fire', 'local_similarity', 'scan', 'simmelian'],
                        help="Method of sparsification")
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--ratio_number', type=int, default=9, help='Number of ratios.')

    args = parser.parse_args()
    if args.defense != 'coarsening' and args.defense != 'sparsification' and args.defense != 'sparsification_svd':
        args.ratio_number = 1
    args.cuda =  not args.no_cuda and torch.cuda.is_available()

    return args


def get_logger(logpath, filepath, package_files=[],
			   displaying=True, saving=True, debug=False):
	logger = logging.getLogger()
	if debug:
		level = logging.DEBUG
	else:
		level = logging.INFO
	logger.setLevel(level)
	if saving:
		info_file_handler = logging.FileHandler(logpath, mode='w')
		info_file_handler.setLevel(level)
		logger.addHandler(info_file_handler)
	if displaying:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		logger.addHandler(console_handler)
	logger.info(filepath)

	for f in package_files:
		logger.info(f)
		with open(f, 'r') as package_f:
			logger.info(package_f.read())

	return logger

def setup_logger(args, path=None):
    log_path = f'./log/{datetime.now().strftime("%b_%d")}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if path is not None:
        log_path += f'/{path}'
    else:
        log_path += f'/{args.technique}_{args.attack}_{args.seed}_{args.dataset}_{args.coarsening_method}.log'
    logger = get_logger(
        logpath=log_path,
        filepath=os.path.abspath(__file__)
    )
	
    return logger


def select_nodes(clean_labels, clean_idx_test, target_gcn, device=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''
    target_gcn.eval()
    output = target_gcn.predict()

    margin_dict = {}
    for idx in clean_idx_test:
        margin = classification_margin(output[idx], clean_labels[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high, other, low


def get_split(data, device):
    rs = np.random.RandomState(10)
    perm = rs.permutation(data.num_nodes)
    train_number = int(0.2*len(perm))
    idx_train = torch.tensor(sorted(perm[:train_number])).to(device)
    data.train_mask = torch.zeros_like(data.train_mask)
    data.train_mask[idx_train] = True

    val_number = int(0.1*len(perm))
    idx_val = torch.tensor(sorted(perm[train_number:train_number+val_number])).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True


    test_number = int(0.2*len(perm))
    idx_test = torch.tensor(sorted(perm[train_number+val_number:train_number+val_number+test_number])).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True

    return data, idx_train, idx_val


def load_data(args, datapath, device='cpu'):
    transform = T.Compose([T.NormalizeFeatures()])
    if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
        dataset = Planetoid(root=datapath, \
                            name=args.dataset,\
                            transform=transform)
    elif(args.dataset == 'Flickr'):
        dataset = Flickr(datapath, transform=transform)
    elif(args.dataset == 'DBLP'):
        dataset = CitationFull(root=datapath, \
                                name=args.dataset,\
                        transform=transform)
    elif(args.dataset == 'Physics'):
        dataset = Coauthor(root=datapath, \
                                name=args.dataset,\
                        transform=transform)
    elif(args.dataset == 'ogbn-arxiv'):
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root=datapath)
        split_idx = dataset.get_idx_split() 

    data = dataset[0].to(device)

    if(args.dataset == 'ogbn-arxiv' or args.dataset == 'DBLP' or args.dataset == 'Physics'):
        nNode = data.x.shape[0]
        setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
        data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
        data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)

    if(args.dataset == 'ogbn-arxiv'):
        data.y = data.y.squeeze(1).flatten()
    
    data, idx_train, idx_val = get_split(data, device)


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



