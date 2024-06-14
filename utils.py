from deeprobust.graph.utils import classification_margin
from deeprobust.graph.defense import GCN
import numpy as np

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
                        choices=['gcn', 'gat', 'gcn_jaccard'])
    parser.add_argument('--nhid', type=int, default=16, help='Hidden layer size.')
    parser.add_argument('--technique', type=str, default='coarsening', 
                        choices=['coarsening', 'sparsification'])
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

def setup_logger(args):
    log_path = f'./log/{datetime.now().strftime("%b_%d")}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path += f'/{args.technique}_FGA_{args.seed}_{args.dataset}_{args.coarsening_method}.log'
    logger = get_logger(
        logpath=log_path,
        filepath=os.path.abspath(__file__)
    )
	
    return logger


def select_nodes(clean_adj, clean_features, clean_labels, clean_idx_train, clean_idx_val, clean_idx_test, target_gcn=None, device=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    if target_gcn is None:
        target_gcn = GCN(nfeat=clean_features.shape[1],
                  nhid=16,
                  nclass=clean_labels.max().item() + 1,
                  dropout=0.5, device=device)
        target_gcn = target_gcn.to(device)
        target_gcn.fit(clean_features, clean_adj, clean_labels, clean_idx_train, clean_idx_val, patience=30)
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