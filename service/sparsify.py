import networkit as nk
import numpy as np
import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix


def sparsify_score(poisoned_adj, method=None, target_ratio=1.0):
    if method is None or target_ratio >= 1.0:
        return poisoned_adj, 1
    else:
        poison_edge_index = from_scipy_sparse_matrix(poisoned_adj)[0]
        print(poison_edge_index.shape)
        G = nk.graph.Graph(weighted=True)
        G.addNodes(poisoned_adj.shape[0])
        check = set()
        for i in range(poison_edge_index.shape[1]):
            # print(poison_edge_index[0, i].item(), poison_edge_index[1, i].item())
            # print(poison_edge_index[0, i], poison_edge_index[1, i])
            if (poison_edge_index[0, i].item(), poison_edge_index[1, i].item()) not in check:
                G.addEdge(poison_edge_index[0, i].item(), poison_edge_index[1, i].item())
                check.add((poison_edge_index[0, i].item(), poison_edge_index[1, i].item()))
                check.add((poison_edge_index[1, i].item(), poison_edge_index[0, i].item()))
        # print(f"Original graph: {G.numberOfNodes()} nodes, {G.numberOfEdges()} edges")
        G.indexEdges()
        print('num of edges:', G.numberOfEdges())
        if method == "local_degree":
            lds = nk.sparsification.LocalDegreeScore(G)
            lds.run()

            ldsScores = lds.scores()
            ldsScores = np.array(ldsScores)


            sort_idx = np.argsort(ldsScores)
            print(ldsScores.shape, ldsScores[sort_idx[:5]])
            


def reverse_sparsify(poisoned_adj, method=None, target_ratio=1.0):
    if method is None or target_ratio >= 1.0:
        return poisoned_adj, 1
    else:
        poison_edge_index = from_scipy_sparse_matrix(poisoned_adj)[0]
        print(poison_edge_index.shape)
        G = nk.graph.Graph(weighted=True)
        G.addNodes(poisoned_adj.shape[0])
        check = set()
        for i in range(poison_edge_index.shape[1]):
            # print(poison_edge_index[0, i].item(), poison_edge_index[1, i].item())
            # print(poison_edge_index[0, i], poison_edge_index[1, i])
            if (poison_edge_index[0, i].item(), poison_edge_index[1, i].item()) not in check:
                G.addEdge(poison_edge_index[0, i].item(), poison_edge_index[1, i].item())
                check.add((poison_edge_index[0, i].item(), poison_edge_index[1, i].item()))
                check.add((poison_edge_index[1, i].item(), poison_edge_index[0, i].item()))
        # print(f"Original graph: {G.numberOfNodes()} nodes, {G.numberOfEdges()} edges")
        G.indexEdges()
        print('num of edges:', G.numberOfEdges())
        if method == "local_degree":
            lds = nk.sparsification.LocalDegreeScore(G)
            lds.run()

            ldsScores = lds.scores()

            ldsScores = sorted(ldsScores, key=lambda x: x[1], reverse=True)
            print(len(ldsScores), ldsScores[:5])


def sparsify(poisoned_adj, method=None, target_ratio=1.0):
    if method is None or target_ratio >= 1.0:
        return poisoned_adj, 1
    else:
        poison_edge_index = from_scipy_sparse_matrix(poisoned_adj)[0]
        G = nk.graph.Graph(weighted=True)
        G.addNodes(poisoned_adj.shape[0])
        check = set()
        for i in range(poison_edge_index.shape[1]):
            # print(poison_edge_index[0, i].item(), poison_edge_index[1, i].item())
            # print(poison_edge_index[0, i], poison_edge_index[1, i])
            if (poison_edge_index[0, i].item(), poison_edge_index[1, i].item()) not in check:
                G.addEdge(poison_edge_index[0, i].item(), poison_edge_index[1, i].item())
                check.add((poison_edge_index[0, i].item(), poison_edge_index[1, i].item()))
                check.add((poison_edge_index[1, i].item(), poison_edge_index[0, i].item()))
        # print(f"Original graph: {G.numberOfNodes()} nodes, {G.numberOfEdges()} edges")
        G.indexEdges()
        if method == "local_degree":
            local_degree_sparsifier = nk.sparsification.LocalDegreeSparsifier()
            sparsified_G = local_degree_sparsifier.getSparsifiedGraphOfSize(G, target_ratio)
            # nk.writeGraph(sparsified_G, "./sparsified_G.graph", nk.Format.EdgeListTabOne)
            # print(f"ratio: {ratio} Sparsified graph: {sparsified_G.numberOfNodes()} nodes, {sparsified_G.numberOfEdges()} edges")
        elif method == "forest_fire":
            forest_fire_sparsifier = nk.sparsification.ForestFireSparsifier(0.6, 5.0)
            sparsified_G = forest_fire_sparsifier.getSparsifiedGraphOfSize(G, target_ratio)
        elif method == "local_similarity":
            similaritySparsifier = nk.sparsification.LocalSimilaritySparsifier()
            sparsified_G = similaritySparsifier.getSparsifiedGraphOfSize(G, target_ratio)
        elif method == "random_node_edge":
            randomNodeEdgeSparsifier = nk.sparsification.RandomNodeEdgeSparsifier()
            sparsified_G = randomNodeEdgeSparsifier.getSparsifiedGraphOfSize(G, target_ratio)
        elif method == "random_edge":
            randomEdgeSparsifier = nk.sparsification.RandomEdgeSparsifier()
            sparsified_G = randomEdgeSparsifier.getSparsifiedGraphOfSize(G, target_ratio)
        elif method == "scan":
            scanSparsifier = nk.sparsification.SCANSparsifier()
            sparsified_G = scanSparsifier.getSparsifiedGraphOfSize(G, target_ratio)
        elif method == "simmelian":
            simmelianSparsifier = nk.sparsification.SimmelianSparsifierNonParametric()
            sparsified_G = simmelianSparsifier.getSparsifiedGraphOfSize(G, target_ratio)
        edge_rate = sparsified_G.numberOfEdges() / G.numberOfEdges()

        sparsified_edges = [[], []]
        for u, v in sparsified_G.iterEdges():
            sparsified_edges[0].append(u)
            sparsified_edges[1].append(v)
            sparsified_edges[0].append(v)
            sparsified_edges[1].append(u)
        sparsified_edges = np.array(sparsified_edges)
        sparsified_adj = torch.zeros((poisoned_adj.shape[0], poisoned_adj.shape[0]))
        for i in range(sparsified_edges.shape[1]):
            sparsified_adj[sparsified_edges[0, i], sparsified_edges[1, i]] = 1
            sparsified_adj[sparsified_edges[1, i], sparsified_edges[0, i]] = 1
        return csr_matrix(sparsified_adj), edge_rate