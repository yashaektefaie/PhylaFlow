from collections import defaultdict, deque

class BHVEncoder():
    def __init__(self, n_leaves):
        self.n_leaves = n_leaves 
    
    def compute_edge_masks(tree: Tree, root=0):
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

        return edge_masks

    def return_BHV_encoding(self, tree):
        edge_masks, edge_lengths = self.compute_edge_masks(tree)
        return edge_masks, edge_lengths
