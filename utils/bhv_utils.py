from collections import defaultdict, deque
import random
from utils.bhv_distance import bhv_geodesic_with_support
from utils.random_tree import RandomTree, Tree
from utils.bhv_movie import make_bhv_topology_movie, sample_tree_along_geodesic

class BHVEncoder():

    def _choose_root(self, tree, root=None):
        """Choose a root for an (unrooted) tree.

        If `root` is provided, use it. Otherwise, pick a random leaf in 1..n.
        Falls back to the smallest node id if no leaves are found.
        """
        if root is not None:
            return root

        # Prefer a random leaf among 1..n_leaves
        leaves = [u for u in tree.adj if 1 <= u <= getattr(tree, "n_leaves", 0)]
        if leaves:
            return random.choice(leaves)

        # Fallback: arbitrary node
        return next(iter(tree.adj))

    def compute_edge_masks(self, tree, root=None):
        """
        Returns:
        edge_masks: dict[(u,v)] -> mask over leaves below v, for directed edges u->v
                    Only for edges that correspond to nontrivial splits.
        Assumes leaves are labeled 1..n_leaves, internal nodes >= n_leaves.
        """
        root = self._choose_root(tree, root)
        n = tree.n_leaves
        full = (1 << n) - 1

        parent = {}
        order = []

        # iterative DFS to get postorder
        stack = [root]
        parent[root] = None
        while stack:
            u = stack.pop()
            order.append(u)
            for v in tree.adj[u]:
                if v not in parent:
                    parent[v] = u
                    stack.append(v)

        # postorder accumulation of leaf masks
        #So every leaf has a bit mask associated with just that identity from 0 to n
        #Every internal node then gets ORd with that bit mask to make a bitmask that represents all nodes it is a parent of

        node_mask = {u: 0 for u in tree.adj}
        for u in reversed(order):
            if 1 <= u <= n:  # leaf
                node_mask[u] = (1 << (u - 1))
            else:
                m = 0
                for v in tree.adj[u]:
                    if parent.get(v) == u:  # child
                        m |= node_mask[v]
                node_mask[u] = m

        # build edge masks for internal edges
        # For each internal edge take the sort of anti-not somehow like if both 1 then 0, this represents the split of the tree
        edge_masks = []
        edge_lengths = []
        for v in tree.adj:
            p = parent.get(v)
            if p is not None:
                A = node_mask[v]
                if A != 0 and A != full:  # nontrivial split
                    # canonical side
                    Ac = full ^ A
                    canon = min(A, Ac)
                    edge_masks.append(canon)
                    edge_lengths.append(tree.length(p,v))

        return edge_masks, edge_lengths

    def return_BHV_encoding(self, tree):
        #Find root of the tree
        
        edge_masks, edge_lengths = self.compute_edge_masks(tree)
        return edge_masks, edge_lengths
    
    def BHV_length(self, one, two):
        edge_mask_1, edge_length_1 = one 
        edge_mask_2, edge_length_2 = two 

        t1 = {x:y for x,y in zip(edge_mask_1, edge_length_1)}
        t2 = {x:y for x,y in zip(edge_mask_2, edge_length_2)}
        result = bhv_geodesic_with_support(t1, t2, n_leaves=self.n_leaves)

        print("BHV distance:", result["distance"])
        for i, seg in enumerate(result["segments"]):
            print(f"Segment {i}:")
            print("  Ai (collapsed):", seg["Ai"])
            print("  Bi (grown):    ", seg["Bi"])
            print("  ratio:", seg["ratio"])
            # seg["start_splits"], seg["end_splits"] give you orthant topology at each step
    
def return_sampled_tree_velocity(newick_tree_one, newick_tree_two, time_point):
    t1 = Tree(newick_tree_one)
    t2 = Tree(newick_tree_two)

    enc = BHVEncoder()
    t1_edge_mask, t1_edge_length = enc.return_BHV_encoding(t1)
    t2_edge_mask, t2_edge_length = enc.return_BHV_encoding(t2)

    tree1 = {m: l for m, l in zip(t1_edge_mask, t1_edge_length)}
    tree2 = {m: l for m, l in zip(t2_edge_mask, t2_edge_length)}

    geodesic_result = bhv_geodesic_with_support(tree1, tree2, n_leaves=t1.n_leaves)
    G, newick, info = sample_tree_along_geodesic(geodesic_result, t1.n_leaves, u=time_point)

    return newick, info['velocity']


def test_bhv_on_two_random_20_leaf_trees():
    n = 20
    print("Generating random trees...")
    T1 = RandomTree(n)
    T2 = RandomTree(n)

    enc = BHVEncoder()

    print("Encoding trees into bitmask form...")
    root = 11
    edge_masks_1, edge_lengths_1 = enc.compute_edge_masks(T1, root = root)
    edge_masks_2, edge_lengths_2 = enc.compute_edge_masks(T2, root = root)

    tree1 = {m: l for m, l in zip(edge_masks_1, edge_lengths_1)}
    tree2 = {m: l for m, l in zip(edge_masks_2, edge_lengths_2)}

    print("Computing BHV geodesic with support pairs...")
    result = bhv_geodesic_with_support(tree1, tree2, n_leaves=n)

    print("\n======================")
    print("BHV DISTANCE =", result["distance"])
    print("======================\n")

    print("Common-edge contribution squared =", result["common_sq"])
    print("Disjoint-edge contribution squared =", result["disjoint_sq"])
    print("Number of support pairs =", len(result["A_support"]))
    print()

    for i, seg in enumerate(result["segments"]):
        print(f"--- Segment {i} ---")
        print("Ai (collapse):", seg["Ai"])
        print("Bi (grow):    ", seg["Bi"])
        print("||A||=", seg["normA"], "||B||=", seg["normB"], "ratio=", seg["ratio"])
        print("#start splits =", len(seg["start_splits"]))
        print("#end splits   =", len(seg["end_splits"]))
        print()

    print("Test completed.")

    make_bhv_topology_movie(
        result,
        n_leaves=n,
        filename="bhv_topology_20leaf.gif",
        fps=1,   # 1 frame per second (one per step)
    )


##############################################################################
# Run the test
##############################################################################

if __name__ == "__main__":
    test_bhv_on_two_random_20_leaf_trees()

        
       
        



