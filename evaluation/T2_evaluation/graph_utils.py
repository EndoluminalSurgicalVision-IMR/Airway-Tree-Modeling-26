# -*- coding: utf-8 -*-
"""
Graph utilities for Track-2 branch-wise anatomical labeling evaluation.

Builds the shortest-path-distance matrix (`spd`) and the ancestor->descendant
subtree membership matrix (`mask_top`) from the GT edge list `airway_graph.npy`.

Adapted from the baseline repo:
  airway_labeling_baseline_LCY_20260302/airwaynet/dataset.py
    - floyd(edge_index)        (L70-85)
    - get_mask(edge, node_num) (L20-31) + dfs (L13-17)
"""
import numpy as np


def build_spd(edge_index):
    """Undirected shortest-path (hop) distance matrix via vectorized Floyd-Warshall.

    edge_index: int array (2, E), node indices in [0, node_num).
    Returns: float array (node_num, node_num); np.inf for unreachable pairs.
    """
    node_num = int(edge_index.max()) + 1
    a = np.full((node_num, node_num), np.inf, dtype=np.float64)
    np.fill_diagonal(a, 0.0)
    u = edge_index[0].astype(np.int64)
    v = edge_index[1].astype(np.int64)
    a[u, v] = 1.0
    a[v, u] = 1.0  # undirected
    # Floyd-Warshall, one O(n^2) vectorized relaxation per intermediate node k.
    for k in range(node_num):
        a = np.minimum(a, a[:, k][:, None] + a[k, :][None, :])
    return a


def build_mask_top(edge_index):
    """Ancestor->descendant matrix M; M[a, d] == 1 iff d is in the subtree of a
    (including a itself). Iterative DFS (no recursion-depth risk).

    edge_index: int array (2, E); edge[0]=parent, edge[1]=child.
    Returns: int8 array (node_num, node_num).
    """
    node_num = int(edge_index.max()) + 1
    children = [[] for _ in range(node_num)]
    for i in range(edge_index.shape[1]):
        children[int(edge_index[0, i])].append(int(edge_index[1, i]))

    M = np.zeros((node_num, node_num), dtype=np.int8)
    for ancestor in range(node_num):
        stack = [ancestor]
        while stack:
            node = stack.pop()
            M[ancestor, node] = 1
            stack.extend(children[node])
    return M
