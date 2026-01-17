import os
import networkx as nx
from typing import List, Set, Tuple, Optional, Iterable, Dict
from collections import defaultdict, deque
import torch
import re
from scipy.special import rel_entr
import numpy as np
from scipy.spatial.distance import jensenshannon
from ete3 import Tree as EteTree
import random
import hashlib

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

def get_batch_polytomy_indices(
    edge_split_masks: List[torch.Tensor],  # [B, T_raw] int64 (bitmask per edge-token)
    edge_mask: torch.Tensor,         # [B, T_raw] bool or {0,1} (valid edge-token positions)
    min_children: int = 3,
    include_root: bool = True,
    num_leaves: int = 50,
) -> List[List[torch.LongTensor]]:
    """
    Groups edge-token indices into overlap-buckets (polytomy "regions") per batch element.

    Returns:
      batch_polytomy_index:
        List over b, each is a List of 1D LongTensors of token indices (positions in [0..T_raw-1]).
        Each tensor corresponds to one "polytomy group" bucket.
    """

    full_mask = (1 << num_leaves) - 1
    root_bit  = 1 << (num_leaves - 1)  # last bit is p_root

    # if edge_split_masks.dim() != 2:
    #     raise ValueError(f"edge_split_masks must be [B,T], got {tuple(edge_split_masks.shape)}")
    if edge_mask.dim() != 2:
        raise ValueError(f"edge_mask must be [B,T], got {tuple(edge_mask.shape)}")

    B = len(edge_split_masks)
    device = edge_mask.device

    batch_polytomy_index: List[List[torch.LongTensor]] = []
    batch_polytomy_splits: List[List[List[int]]] = []

    for b in range(B):
        valid_pos = torch.nonzero(edge_mask[b], as_tuple=False).squeeze(1)  # positions in [0..T-1]

        # splits for valid edge tokens
        splits_b = edge_split_masks[b] #This only contains the valid splits it is not indexed for the full lenght of input 
        if len(splits_b) != edge_mask[b].sum().item():
            raise ValueError("Length mismatch between splits and valid edge mask. This SHOULD NOT HAPPEN.")

        # Map split_mask -> list of token positions that have that split.
        # (Important: keep duplicates! don't lose indices.)
        split_to_positions: Dict[int, List[int]] = defaultdict(list)
        for pos, sm in zip(valid_pos.tolist(), splits_b):
            # You can choose to ignore 0 masks if those mean "no split"
            # (often 0 is padding or placeholder)
            if sm == 0:
                continue
            split_to_positions[int(sm)].append(int(pos))

        unique_splits = list(split_to_positions.keys())

        #ADD in the root nodes
        # candidates = list(unique_splits)
        # if include_root:
        #     candidates.append(full_mask ^ root_bit)  # p_root
        unique_splits.append(full_mask ^ root_bit)

        polytomy_groups: List[torch.LongTensor] = []
        polytomy_splits: List[List[int]] = []

        def is_subset(sub: int, sup: int) -> bool:
            return (sub & ~sup) == 0

        n = len(unique_splits)
        for pi in range(n):
            p = unique_splits[pi]

            # Proper subsets of p (exclude p)
            subs = [s for s in unique_splits if s!= p and is_subset(s, p)]
            if len(subs) < min_children:
                continue

            # Maximal proper subsets within p:
            # s is maximal if there is NO t in subs such that s ⊂ t ⊂ p
            maximal_subs = []
            for s in subs:
                dominated = False
                for t in subs:
                    if s != t and is_subset(s, t):  # s ⊆ t
                        # if t strictly larger than s, s is not maximal
                        if t.bit_count() > s.bit_count():
                            dominated = True
                            break
                if not dominated:
                    maximal_subs.append(s)

            if len(maximal_subs) >= min_children:
                # Collect token positions for this polytomy region
                idxs: List[int] = []
                for s in maximal_subs:
                    idxs.extend(split_to_positions[int(s)])

                # Dedup + sort for stable indexing
                idxs = sorted(set(idxs))
                polytomy_groups.append(torch.tensor(idxs, dtype=torch.long, device=device))
                polytomy_splits.append(maximal_subs)

        batch_polytomy_index.append(polytomy_groups)
        batch_polytomy_splits.append(polytomy_splits)

    return batch_polytomy_index, batch_polytomy_splits

def pick_group(W, tau=0.5):
    # W: symmetric, diag=-inf
    G = W.size(0)
    i, j = divmod(torch.argmax(W).item(), G)
    if torch.sigmoid(W[i, j]) < tau:
        return None  # nothing confident

    S = {i, j}

    while True:
        best_k, best_score = None, None
        for k in range(G):
            if k in S: 
                continue
            # score to join group: conservative = min link, or average link
            score = torch.sigmoid(torch.stack([W[k, s] for s in S]).min())
            # alternatively: score = torch.sigmoid(torch.stack([W[k,s] for s in S]).mean())
            if best_score is None or score > best_score:
                best_k, best_score = k, score
        if best_score is None or best_score < tau:
            break
        S.add(best_k)

    return sorted(S)

def number_to_name_newick(newick: str, mapping: Dict[int, str], zero_indexed_tree: bool) -> str:
    # Replace digits that are immediately followed by ':' (branch length delimiter)
    pat = re.compile(r'\b(\d+)\b(?=:)')
    def repl(m):
        num = int(m.group(1))
        if not zero_indexed_tree:
            num = num-1
        if num not in mapping:
            raise Exception(f"Mapping missing for leaf number {num} in newick.")
        return mapping.get(num, m.group(1))  # NO colon here
    
    return pat.sub(repl, newick)

def jensenshannon_loglh_divergence(
    true_loglhs: List[float], 
    sampled_loglhs: List[float], 
    bins: int = 50
) -> float:
    """Compute Jensen-Shannon divergence between two log-likelihood distributions."""
    all_vals = true_loglhs + sampled_loglhs
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)
    p, _ = np.histogram(true_loglhs, bins=bin_edges, density=True)
    q, _ = np.histogram(sampled_loglhs, bins=bin_edges, density=True)
    # Add small epsilon to avoid zero probabilities
    p = p + 1e-10
    q = q + 1e-10
    return jensenshannon(p, q)


def kl_loglh_divergence(
    true_loglhs: List[float], 
    sampled_loglhs: List[float], 
    bins: int = 50
) -> float:
    """Compute KL divergence D(true || sampled) between two log-likelihood distributions."""
    all_vals = true_loglhs + sampled_loglhs
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)
    p, _ = np.histogram(true_loglhs, bins=bin_edges, density=True)
    q, _ = np.histogram(sampled_loglhs, bins=bin_edges, density=True)
    # Normalize to proper probability distributions and add epsilon
    p = (p + 1e-10) / (p + 1e-10).sum()
    q = (q + 1e-10) / (q + 1e-10).sum()
    return rel_entr(p, q).sum()

def return_total_tree_length(newick: str) -> float:
    """
    Computes the total tree length from a Newick string.
    Assumes branch lengths are provided in the Newick format.
    """
    length_pattern = re.compile(r':([\d\.eE+-]+)')
    lengths = length_pattern.findall(newick)
    total_length = sum(float(length) for length in lengths)
    return total_length

def _stable_seed(newick: str, salt: str = "") -> int:
    """
    Deterministic 64-bit seed from (salt | newick). Stable across runs/machines.
    """
    h = hashlib.blake2b((salt + "|" + newick).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)

def resolve_polytomies_random_deterministic(
    newick: str,
    *,
    dummy_name: str = "ROOT_DUMMY",
    min_len: float = 1e-6,
    salt: str = "",
) -> str:
    """
    Make a Newick strictly bifurcating by deterministically-randomly resolving polytomies.
    - Drops a dummy leaf if present.
    - Contracts unary nodes (degree-2 after rooting representation).
    - Resolves any node with >2 children by repeatedly grouping 2 children under a new internal node.
    - Clamps all branch lengths to >= min_len.

    Determinism: seed = hash(salt | newick). Use salt to get K different but reproducible refinements.
    """
    rng = random.Random(_stable_seed(newick, salt))

    # Parse (format=1 expects branch lengths if present; still works if absent)
    t = EteTree(newick, format=1)

    # Drop dummy leaf if present
    dummies = t.search_nodes(name=dummy_name)
    if dummies:
        dummies[0].detach()

    def clamp_lengths(tree: EteTree):
        for n in tree.traverse():
            if n.is_root():
                continue
            if n.dist is None:
                n.dist = float(min_len)
            else:
                n.dist = max(float(n.dist), float(min_len))

    def contract_unary(tree: EteTree) -> EteTree:
        """
        Remove nodes with a single child by contracting them (add lengths).
        Returns (possibly new) root.
        """
        changed = True
        while changed:
            changed = False
            # Postorder so we contract from leaves upward
            for n in list(tree.traverse("postorder")):
                if n.is_leaf():
                    continue
                if len(n.children) == 1:
                    child = n.children[0]
                    # accumulate length into child
                    if not n.is_root():
                        child.dist = (child.dist or 0.0) + (n.dist or 0.0)
                        parent = n.up
                        child.detach()
                        n.detach()
                        parent.add_child(child)
                        changed = True
                    else:
                        # root with one child: promote child to root
                        child.dist = 0.0
                        child.detach()
                        tree = child
                        changed = True
        return tree

    # Clean up and clamp
    t = contract_unary(t)
    clamp_lengths(t)

    # Resolve polytomies
    # Postorder so deeper polytomies are resolved first
    for n in list(t.traverse("postorder")):
        while not n.is_leaf() and len(n.children) > 2:
            # Pick two children in a deterministic-random way
            c1 = rng.choice(n.children); c1.detach()
            c2 = rng.choice(n.children); c2.detach()

            mid = EteTree()
            mid.dist = float(min_len)
            mid.add_child(c1)
            mid.add_child(c2)

            n.add_child(mid)

    # Contract any unary nodes created by dummy removal / restructuring, clamp again
    t = contract_unary(t)
    clamp_lengths(t)

    # Write back
    return t.write(format=1)

def has_polytomy_fast(newick: str, unrooted_ok: bool = True) -> bool:
    comma_stack = []  # top-level comma count per open '(' group

    for ch in newick:
        if ch == '(':
            comma_stack.append(0)
        elif ch == ',':
            if comma_stack:
                comma_stack[-1] += 1
        elif ch == ')':
            if not comma_stack:
                continue
            commas = comma_stack.pop()

            # if stack is empty AFTER pop => this was the OUTERMOST group (the Newick "root")
            is_root_group = (len(comma_stack) == 0)

            children = commas + 1
            limit = 3 if (unrooted_ok and is_root_group) else 2
            if children > limit:
                return True

    return False
