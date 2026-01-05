from ete3 import Tree
import numpy as np
import random
import math
import os
import networkx as nx
from typing import List, Set, Tuple, Optional
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

def polytomy_components_at_node(G: nx.Graph, node: int, n_leaves: int) -> List[int]:
    """
    For a multifurcating node, return component masks of each incident branch.
    Assumes leaf nodes are labeled 0..n_leaves-1 (true in your build_tree_from_splits output).
    """
    leaf_nodes = [str(i) for i in set(range(n_leaves))]
    comps = []
    for nb in G.neighbors(node):
        m = leaves_in_component(G, nb, node, leaf_nodes)
        if m != 0:
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

def popcount(x: int) -> int:
    return x.bit_count()

def build_target_clusters_from_splits(
    target_canonical_splits: Set[int],
    full_mask: int,
    region_mask: int,
) -> Set[int]:
    """
    Convert target unrooted splits into cluster masks within region.
    For each split A|A^c, both A and A^c are possible clusters.
    Keep only those contained in region_mask and nontrivial.
    """
    clusters: Set[int] = set()
    for s in target_canonical_splits:
        a = s
        b = full_mask ^ s
        if (a & ~region_mask) == 0 and a not in (0, region_mask):
            clusters.add(a)
        if (b & ~region_mask) == 0 and b not in (0, region_mask):
            clusters.add(b)
    return clusters

def all_merge_actions(k: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(k) for j in range(i + 1, k)]

def apply_merge(components: List[int], i: int, j: int) -> List[int]:
    """Merge components i and j -> union, return new component list."""
    if i > j:
        i, j = j, i
    out = []
    for t, c in enumerate(components):
        if t not in (i, j):
            out.append(c)
    out.append(components[i] | components[j])
    return out

def pick_teacher_merge(
    components: List[int],
    target_clusters_in_region: Set[int],
    region_mask: int,
) -> Optional[Tuple[int, int]]:
    """
    Teacher forcing on merges: choose a pair (i,j) such that U=Ci∪Cj is
    a minimal target cluster (within region) containing U, i.e. U is a "cherry"
    in the induced target refinement over current components.

    If multiple, pick smallest |U|.
    """
    best = None
    best_size = None

    for i, j in all_merge_actions(len(components)):
        U = components[i] | components[j]
        # minimal target cluster containing U (within region)
        candidates = [c for c in target_clusters_in_region if (U & ~c) == 0 and (c & ~region_mask) == 0]
        if not candidates:
            continue
        cmin = min(candidates, key=popcount)
        if cmin == U:
            u_sz = popcount(U)
            if best is None or u_sz < best_size:
                best = (i, j)
                best_size = u_sz
    return best

def teacher_force_merge_sequence(
    components: List[int],
    target_canonical_splits: Set[int],
    full_mask: int,
) -> List[Tuple[int, int]]:
    """
    Produce merge labels until unresolved region is resolved enough.
    For an unrooted refinement, once k<=3 you can attach without adding internal splits.
    """
    comps = components[:]
    region_mask = 0
    for c in comps:
        region_mask |= c

    target_clusters = build_target_clusters_from_splits(target_canonical_splits, full_mask, region_mask)

    labels: List[Tuple[int, int]] = []
    while len(comps) > 3:
        a = pick_teacher_merge(comps, target_clusters, region_mask)
        if a is None:
            raise RuntimeError(
                f"No valid teacher merge found for region with k={len(comps)} components. "
                f"Likely mismatch of leaf indexing/mapping or target splits."
            )
        labels.append(a)
        comps = apply_merge(comps, *a)

    return labels