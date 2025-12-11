import random 
from collections import defaultdict
import networkx as nx
from ete3 import Tree as eteTree
from utils.bhv_movie import tree_to_newick

class RandomTree:
    """
    Build a random unrooted binary tree with n_leaves numbered 1..n.
    Internal nodes numbered n+1, n+2, ...
    """
    def __init__(self, n_leaves):
        self.n_leaves = n_leaves
        self.adj = defaultdict(list)
        self.lengths = {}   # store edge lengths: key=(u,v) in sorted order

        # Build random unrooted binary tree via random split growth.
        # (This is not Yule or coalescent, just a simple binary random shape.)
        #
        # Start with a cherry: 1 -- (n+1) -- 2
        internal_id = n_leaves + 1

        # adjacency for starting structure
        self._add_edge(1, internal_id)
        self._add_edge(2, internal_id)

        # Internal nodes used so far
        current_internal = [internal_id]

        # Add leaf 3..n
        for leaf in range(3, n_leaves + 1):
            # Pick a random existing edge to subdivide
            # Choose a random internal or leaf, then one of its neighbors
            u = random.choice(list(self.adj.keys()))
            v = random.choice(self.adj[u])

            # Remove edge u--v
            self._remove_edge(u, v)

            # Create new internal node
            internal_id += 1
            w = internal_id

            # New edges u--w, v--w, w--leaf
            self._add_edge(u, w)
            self._add_edge(v, w)
            self._add_edge(w, leaf)

        # Randomize lengths
        for u in self.adj:
            for v in self.adj[u]:
                if (u,v) not in self.lengths and (v,u) not in self.lengths:
                    L = random.uniform(0.1, 1.0)
                    self.lengths[(u,v)] = self.lengths[(v,u)] = L

    def _add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)

    def _remove_edge(self, u, v):
        self.adj[u].remove(v)
        self.adj[v].remove(u)

    def length(self, u, v):
        # symmetric
        return self.lengths.get((u, v), self.lengths.get((v, u)))
    
    def to_newick(self, name_map=None):
        """
        Return Newick string.
        If name_map is provided, it should map integer leaf IDs to string labels.
        Internal nodes keep integer IDs (or you can omit labels).
        """
        def build_newick(node, parent):
            children = [n for n in self.adj[node] if n != parent]
            if not children:
                # leaf
                if name_map is not None and node in name_map:
                    label = name_map[node]
                else:
                    label = str(node)
                return label
            else:
                subtrees = [
                    build_newick(c, node) + f":{self.length(node, c):.4f}"
                    for c in children
                ]
                return "(" + ",".join(subtrees) + ")"

        # Arbitrarily root at leaf 1 just for Newick representation
        return build_newick(1, None) + ";"

    def __str__(self):
        return self.to_newick()



class Tree:
    """Build a tree object from a Newick string.

    Leaves are assumed to be labeled with integers (1..n). Internal nodes are
    assigned integer IDs greater than the maximum leaf label. Edge lengths are
    read from the Newick branch lengths; if missing, a small default (0.1) is
    used.
    """

    def __init__(self, newick: str):
        self.adj = defaultdict(list)
        self.lengths = {}  # symmetric edge lengths: key=(u,v) or (v,u)
        self.n_leaves = 0
        self.id_to_name = {}

        self._build_from_newick(newick)

    def _build_from_newick(self, newick: str):
        t = eteTree(newick)

        current_leaves = list(t.iter_leaves())
        max_id = 0
        for l in current_leaves:
            try:
                val = int(l.name)
                if val > max_id:
                    max_id = val
            except ValueError:
                pass # Ignore non-integer names if any

        # Create the ID for the new leaf (e.g., if leaves are 1..4, this is 5)
        dummy_id = max_id + 1

        # Create a new "Super Root"
        # We move the original tree to be a child of this new node, 
        # and add the dummy leaf as the second child.
        new_root = eteTree()
        new_root.add_child(t, dist=0.0) # Original tree attached here
        new_root.add_child(name=dummy_id, dist=0.0) # Dummy anchor attached here
        
        # Point our tree reference to this new super structure
        t = new_root

        # 1) Collect leaves and assume their names are integers 1..n
        leaf_nodes = list(t.iter_leaves())
        self.n_leaves = len(leaf_nodes)+1  # +1 for dummy leaf

        # 2) Map ete3 nodes -> integer IDs
        mapping = {}
        mapping_to_rename = {}

        num = 0
        # leaves keep their numeric labels
        for n in leaf_nodes:
            lid = int(n.name)
            mapping[n] = lid
            mapping_to_rename[n] = num
            self.id_to_name[num] = n.name
            num += 1

        next_internal_id = self.n_leaves

        # internal nodes get new IDs
        for n in t.traverse("postorder"):
            if not n.is_leaf():
                if n not in mapping:
                    mapping[n] = next_internal_id
                    mapping_to_rename[n] = next_internal_id
                    next_internal_id += 1

        self.root = mapping_to_rename[t]

        # 3) Build adjacency and lengths
        for parent in t.traverse():
            u = mapping_to_rename[parent]
            for child in parent.children:
                v = mapping_to_rename[child]
                L = child.dist if child.dist is not None else 0.1
                self.adj[u].append(v)
                self.adj[v].append(u)
                self.lengths[(u, v)] = self.lengths[(v, u)] = L

    def length(self, u, v):
        return self.lengths.get((u, v), self.lengths.get((v, u)))

    def __str__(self):
        # Build a NetworkX graph from adjacency/lengths and convert to Newick
        G = nx.Graph()
        # add nodes
        for u in self.adj:
            G.add_node(u)
        # add edges with length
        for u in self.adj:
            for v in self.adj[u]:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, length=self.length(u, v))

        return tree_to_newick(G, root=None)