from ete3 import Tree
import numpy as np
import random
import math
import os

def get_possible_ids(nexus_root):
    ids = []
    for name in os.listdir(nexus_root):
        base, ext = os.path.splitext(name)
        ids.append(base)
    ids.sort()
    return ids

def get_splits(newick, leaf_set=None):
    """
    Extract nontrivial splits from a Newick tree and return a dict:
        split_tuple -> branch_length
    Splits are canonicalized: always the smaller side first.
    """
    tree = Tree(newick, format=1)

    # Get all leaf names
    leaves = leaf_set if leaf_set is not None else sorted(tree.get_leaf_names())
    leaf_index = {leaf: i for i, leaf in enumerate(leaves)}
    n = len(leaves)
    
    splits = {}

    for node in tree.traverse():
        if node.is_root() or node.is_leaf():
            continue

        # Collect leaf names under this node
        clade = sorted(node.get_leaf_names())
        complement = [x for x in leaves if x not in clade]

        # Skip trivial splits
        if len(clade) < 2 or len(complement) < 2:
            continue

        # Canonicalize: always store smaller partition first
        A, B = (tuple(clade), tuple(complement))
        split = tuple(sorted([A, B], key=lambda x: (len(x), x)))
        
        splits[split] = node.dist  # branch length

    return splits

def build_global_split_index(newick_list):
    global_splits = {}
    for newick in newick_list:
        splits = get_splits(newick)
        for split in splits:
            if split not in global_splits:
                global_splits[split] = len(global_splits)
    return global_splits


def tree_to_bhv_vector(newick, global_splits, leaf_set=None):
    vec = np.zeros(len(global_splits), dtype=np.float32)
    splits = get_splits(newick, leaf_set=leaf_set)

    for split, length in splits.items():
        if split in global_splits:
            idx = global_splits[split]
            vec[idx] = length
    return vec

def random_topology(leaves):
    """
    Generate a random binary tree topology over given leaf labels.
    """
    nodes = [Tree(name=leaf) for leaf in leaves]

    while len(nodes) > 1:
        # pick two nodes to merge
        a = nodes.pop(random.randrange(len(nodes)))
        b = nodes.pop(random.randrange(len(nodes)))

        parent = Tree()
        parent.add_child(a)
        parent.add_child(b)

        nodes.append(parent)

    return nodes[0]

def random_bhv_tree(leaves, length_scale=0.1):
    t = random_topology(leaves)

    for node in t.traverse():
        if not node.is_root():
            node.dist = math.exp(length_scale * random.gauss(0, 1))
    
    return t



def test():
    trees = [
        "(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);",
        "((A:0.2,C:0.1):0.3,B:0.7,D:0.2);"
    ]

    global_splits = build_global_split_index(trees)

    vec1 = tree_to_bhv_vector(trees[0], global_splits)
    vec2 = tree_to_bhv_vector(trees[1], global_splits)

    print("dim =", len(global_splits))
    print(vec1)
    print(vec2)
