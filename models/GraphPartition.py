import torch
from torch_geometric.nn import GCNConv
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy
from utils import load_data, coarsening
from utils import accuracy


"""
coarsening
"""
class CoarseningMachine(object):
    """
    Coarsening the graph.
    """
    def __init__(self, args, clusters, cluster_degree,
                    sg_nodes,sg_edges, sg_features, sg_targets, 
                    sg_train_nodes, sg_test_nodes, 
                    coarsening_ratio, coarsening_method):
        """
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.args = args
        self.clusters = clusters
        self.cluster_degree = cluster_degree
        self.sg_nodes = sg_nodes
        self.sg_edges = sg_edges
        self.sg_features = sg_features
        self.sg_targets = sg_targets
        self.sg_train_nodes = sg_train_nodes
        self.sg_test_nodes = sg_test_nodes
        self.coarsening_ratio = 1-coarsening_ratio
        self.coarsening_method = coarsening_method
        self.coarsen()

    def coarsen(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        self.calculate_ratio()
        self.cluster_coarsening()
    
    def calculate_ratio(self):
        """
        Calculate coarsening ratio for each cluster.
        """
        # degree_based_calculation
        self.cluster_coarsening_ratio = [(degree / (sum(self.cluster_degree)/len(self.cluster_degree)))**2 * self.coarsening_ratio for degree in self.cluster_degree]
        print(f"cluster coarsening ratio: {self.cluster_coarsening_ratio}")


    def cluster_coarsening(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_coarsening_nodes = {}
        self.sg_coarsening_edges = {}
        self.sg_coarsening_train_mask = {}
        self.sg_coarsening_val_mask = {}
        self.sg_coarsening_features = {}
        self.sg_coarsening_labels = {}

        for cluster in self.clusters:

            edges = self.sg_edges[cluster]
            features = self.sg_features[cluster]
            target = self.sg_targets[cluster].squeeze()

            candidate, C_list, Gc_list = coarsening(self.args.dataset, edges, 1-self.cluster_coarsening_ratio[cluster], self.coarsening_method)
            bkd_tn_nodes = self.sg_train_nodes[cluster]
            idx_val = self.sg_test_nodes[cluster]
            data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, coarsen_labels = load_data(
                self.args.dataset, candidate, C_list, Gc_list, features, target, bkd_tn_nodes, idx_val)

            self.sg_coarsening_features[cluster] = coarsen_features
            self.sg_coarsening_edges[cluster] = coarsen_edge
            self.sg_coarsening_labels[cluster] = coarsen_labels
            self.sg_coarsening_train_mask[cluster] = coarsen_train_mask
            self.sg_coarsening_val_mask[cluster] = coarsen_val_mask
            













"""
clustering
"""
class ClusteringMachine(object):
    """
    Clustering the graph, feature set and target.
    """
    def __init__(self, args, graph, features, target, cluster_number, idx_train, idx_val):
        """
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self.cluster_number = cluster_number
        self.idx_train = idx_train
        self.idx_val = idx_val
        self._set_sizes()

    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        self.feature_count = self.features.shape[1] 
        self.class_count = np.max(self.target)+1

    def decompose(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        self.random_clustering()
        self.general_data_partitioning()
        self.transfer_edges_and_nodes()

        return self.sg_nodes, self.sg_edges, self.sg_features, self.sg_targets, self.sg_train_nodes, self.sg_test_nodes, self.clusters, self.cluster_degree_order, self.connecting_edge

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}
       

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}
        self.edge_index = []
        cluster_edge = []
        connecting_edge = []
        self.cluster_degree_order = []
        edge = list(self.graph.edges())
        
        # for cluster in self.clusters:
        #     subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if node in self.cluster_membership and self.cluster_membership[node] == cluster])
        #     self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
        #     # mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
        #     # mapper = {node: node for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
        #     self.sg_edges[cluster] = [[edge[0], edge[1]] for edge in subgraph.edges()] + [[edge[1], edge[0]] for edge in subgraph.edges()]
            
        #     # add all the edges
        #     cluster_edge += self.sg_edges[cluster]
        #     # get the average degree of the cluster
        #     cluster_avg_degree = np.mean([val for (node, val) in subgraph.degree()])
        #     self.cluster_degree_order.append(cluster_avg_degree)
        #     print(f"cluster {cluster} has {len(self.sg_nodes[cluster])} nodes, {len(self.sg_edges[cluster])} edges, and average degree {cluster_avg_degree}")
            
            
        #     self.sg_train_nodes[cluster] = []
        #     for node in self.idx_train.cpu().numpy():
        #         if node in self.sg_nodes[cluster]:
        #             self.sg_train_nodes[cluster].append(node)
        #     self.sg_test_nodes[cluster] = []
        #     for node in self.idx_val.cpu().numpy():
        #         if node in self.sg_nodes[cluster]:
        #             self.sg_test_nodes[cluster].append(node)
        #     self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
        #     self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])

        # # self.edge_index = torch.LongTensor(cluster_edge).t()
        # self.features = torch.FloatTensor(self.features)
        # self.target = torch.LongTensor(self.target).flatten()
        # # self.idx_train = torch.LongTensor(self.idx_train)
        # # self.idx_val = torch.LongTensor(self.idx_val)
        


        for cluster in self.clusters:
            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if node in self.cluster_membership and self.cluster_membership[node] == cluster])
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] + [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            """
            calculate average degree of the cluster
            """
            cluster_avg_degree = np.mean([val for (node, val) in subgraph.degree()])
            self.cluster_degree_order.append(cluster_avg_degree)
            print(f"cluster {cluster} has {len(self.sg_nodes[cluster])} nodes, {len(self.sg_edges[cluster])} edges, and average degree {cluster_avg_degree}")
            """"""
            self.sg_train_nodes[cluster] = []
            for node in self.idx_train.cpu().numpy():
                if node in self.sg_nodes[cluster]:
                    self.sg_train_nodes[cluster].append(mapper[node])
            self.sg_test_nodes[cluster] = []
            for node in self.idx_val.cpu().numpy():
                if node in self.sg_nodes[cluster]:
                    self.sg_test_nodes[cluster].append(mapper[node])
            self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
            self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])
            self.sg_features[cluster] = self.features[self.sg_nodes[cluster],:]
            # self.sg_targets[cluster] = self.target[self.sg_nodes[cluster],:]
            temp = []
            for node in self.sg_nodes[cluster]:
                if node >= len(self.target):
                    continue
                temp.append(self.target[node])
            self.sg_targets[cluster] = np.array(temp)

        

        # get the edges connecting different clusters
        edges_list = [list(edge) for edge in self.graph.edges()]
        connecting_edge = [item for item in edges_list if item not in cluster_edge]+[item[::-1] for item in edges_list if item[::-1] not in cluster_edge]
        connecting_edge = sorted(connecting_edge)
        self.connecting_edge = connecting_edge
        self.edge_index = torch.LongTensor(cluster_edge+connecting_edge).t()
        
        

        


        
       

    def transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format.
        """
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster])
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])



"""
ClusterGCN
"""
class GraphPartition(object):
    """
    Training a ClusterGCN.
    """
    def __init__(self, args, graph, features, target, idx_train, idx_val, device=None, cluster_num=5, coarsening_ratio=1, coarsening_method=None, test_model='GCN'):
        """
        :param ags: Arguments object.
        :param graph: Networkx graph.
        :param features: Feature matrix.
        :param target: Target vector.
        :param device: CPU or GPU.
        """  
        self.args = args
        self.clustering_machine = ClusteringMachine(args, graph, features, target, cluster_num, idx_train, idx_val)
        sg_nodes, sg_edges, sg_features, sg_targets, sg_train_nodes, sg_test_nodes, clusters, cluster_degree_order, connecting_edge = self.clustering_machine.decompose()
        self.device = device
        self.cluster_number = cluster_num
        self.coarsening_ratio = coarsening_ratio
        self.coarsening_machine = CoarseningMachine(args, clusters, cluster_degree_order,
                                                    sg_nodes,sg_edges, sg_features, sg_targets, 
                                                    sg_train_nodes, sg_test_nodes, 
                                                    coarsening_ratio, coarsening_method)
         
 

     