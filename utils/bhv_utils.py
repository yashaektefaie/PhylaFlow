from collections import defaultdict, deque
from utils.bhv_distance import bhv_geodesic_with_support
from utils.random_tree import RandomTree

class BHVEncoder():
    def __init__(self, n_leaves):
        self.n_leaves = n_leaves 
    
    def compute_edge_masks(self, tree, root=0):
        """
        Returns:
        edge_masks: dict[(u,v)] -> mask over leaves below v, for directed edges u->v
                    Only for edges that correspond to nontrivial splits.
        Assumes leaves are labeled 1..n_leaves, internal nodes >= n_leaves.
        """
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


def test_bhv_on_two_random_20_leaf_trees():
    n = 20
    print("Generating random trees...")
    T1 = RandomTree(n)
    T2 = RandomTree(n)

    enc = BHVEncoder(n)

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


##############################################################################
# Run the test
##############################################################################

if __name__ == "__main__":
    test_bhv_on_two_random_20_leaf_trees()

        
       
        



