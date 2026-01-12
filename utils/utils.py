from ete3 import Tree
import numpy as np
import random
import math
import os
import networkx as nx
from typing import List, Set, Tuple, Optional, Iterable, Dict
from collections import deque

def get_possible_ids(nexus_root):
    ids = []
    for name in os.listdir(nexus_root):
        base, ext = os.path.splitext(name)
        ids.append(base)
    ids.sort()
    return ids

def remove_bit(mask: int, d: int) -> int:
    """
    Remove bit position d from 'mask' and compress higher bits down by 1.
    Example: remove_bit(0b101001, d=3) removes the 8's place.
    """
    low = mask & ((1 << d) - 1)     # bits [0..d-1]
    high = mask >> (d + 1)         # bits [d+1..] shifted down
    return low | (high << d)

def find_polytomy_nodes(G: nx.Graph, min_degree: int = 4) -> List[int]:
    """
    Internal nodes with degree >= 4 are polytomies (unrooted).
    """
    polys = []
    for u in G.nodes():
        if G.degree[u] >= min_degree:
            polys.append(u)
    return polys

def polytomy_components_at_node(G: nx.Graph, node: int, n_leaves: int, return_comps: bool = False) -> List[int]:
    """
    For a multifurcating node, return component masks of each incident branch.
    Assumes leaf nodes are labeled 0..n_leaves-1 (true in your build_tree_from_splits output).
    """
    leaf_nodes = [str(i) for i in range(n_leaves)]
    comps = []
    for nb in G.neighbors(node):
        m = leaves_in_component(G, nb, node, leaf_nodes)
        if m != 0:
            if return_comps:
                comps.append((nb, m))
            else:
                comps.append(m)
    return comps

def leaves_in_component(G: nx.Graph, start: int, forbidden: int, leaf_nodes: Set[int]) -> int:
    """Bitmask of leaves reachable from start without passing through forbidden."""
    seen = {forbidden}
    q = deque([start])
    mask = 0
    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        if u in leaf_nodes:
            mask |= (1 << int(u))
        for v in G.neighbors(u):
            if v not in seen:
                q.append(v)
    return mask

def bucket_by_overlap(splits: Iterable[int]) -> List[Set[int]]:
    """
    Buckets are connected components under the relation 'overlap' (a&b != 0).
    O(n^2) in #splits; usually fine for boundary-size sets.
    """
    splits = list(set(splits))
    n = len(splits)
    seen = set()
    buckets: List[Set[int]] = []

    for i in range(n):
        s = splits[i]
        if s in seen:
            continue

        comp = set([s])
        seen.add(s)
        q = deque([s])

        while q:
            cur = q.popleft()
            for t in splits:
                if t in seen:
                    continue
                if (cur & t) != 0:
                    seen.add(t)
                    comp.add(t)
                    q.append(t)

        buckets.append(comp)

    # sort buckets (largest "max split" first) just for nicer printing
    buckets.sort(key=lambda B: max(x.bit_count() for x in B), reverse=True)
    return buckets
