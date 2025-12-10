import random 
from collections import defaultdict

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

    def __str__(self):
        #Build newick representation
        def build_newick(node, parent):
            children = [n for n in self.adj[node] if n != parent]
            if not children:
                return str(node)
            else:
                subtrees = [build_newick(c, node) + f":{self.length(node,c):.4f}" for c in children]
                return "(" + ",".join(subtrees) + ")"
        return build_newick(1, None) + ";"


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

        self._build_from_newick(newick)

    def _build_from_newick(self, newick: str):
        s = newick.strip()
        if s.endswith(";"):
            s = s[:-1]

        idx = 0
        temp_edges = []  # (parent, child, length)
        leaves = set()
        internal_counter = 0

        def parse_branch_length():
            nonlocal idx
            if idx < len(s) and s[idx] == ":":
                idx += 1
                start = idx
                while idx < len(s) and s[idx] not in ",)":
                    idx += 1
                try:
                    return float(s[start:idx])
                except ValueError:
                    return None
            return None

        def parse_subtree():
            nonlocal idx, internal_counter
            if s[idx] == "(":
                idx += 1  # consume '('
                node_id = f"I{internal_counter}"
                internal_counter += 1

                while True:
                    child, blen = parse_subtree()
                    temp_edges.append((node_id, child, blen))

                    if idx < len(s) and s[idx] == ",":
                        idx += 1
                        continue
                    elif idx < len(s) and s[idx] == ")":
                        idx += 1  # consume ')'
                        break
                    else:
                        break

                # optional internal label (ignored)
                while idx < len(s) and s[idx] not in ",):":
                    idx += 1

                blen = parse_branch_length()
                return node_id, blen
            else:
                start = idx
                while idx < len(s) and s[idx] not in ",):":
                    idx += 1
                label = s[start:idx].strip()
                try:
                    node_id = int(label)
                except ValueError:
                    node_id = label  # fallback to string label
                leaves.add(node_id)
                blen = parse_branch_length()
                return node_id, blen

        root_id, _ = parse_subtree()

        # Map nodes to integer IDs: leaves keep their numeric label; internals get new ints
        leaf_ints = [x for x in leaves if isinstance(x, int)]
        max_leaf = max(leaf_ints) if leaf_ints else 0
        next_internal_id = max_leaf + 1

        mapping = {}
        for leaf in leaves:
            if isinstance(leaf, int):
                mapping[leaf] = leaf
            else:
                mapping[leaf] = next_internal_id
                next_internal_id += 1

        # ensure the root placeholder has an id
        mapping.setdefault(root_id, next_internal_id)
        if root_id not in mapping:
            next_internal_id += 1

        # assign IDs for internal placeholders
        for parent, child, _ in temp_edges:
            if isinstance(parent, str) and parent not in mapping:
                mapping[parent] = next_internal_id
                next_internal_id += 1
            if isinstance(child, str) and child not in mapping:
                mapping[child] = next_internal_id
                next_internal_id += 1

        # rebuild adjacency and lengths with mapped integer IDs
        for parent, child, blen in temp_edges:
            u = mapping[parent]
            v = mapping[child]
            L = blen if blen is not None else 0.1
            self.adj[u].append(v)
            self.adj[v].append(u)
            self.lengths[(u, v)] = self.lengths[(v, u)] = L

        self.n_leaves = len(leaves)
        self.root = mapping.get(root_id)

    def length(self, u, v):
        return self.lengths.get((u, v), self.lengths.get((v, u)))

    def __str__(self):
        # choose the smallest leaf as the display root if available
        leaves = [n for n in self.adj if 1 <= n <= self.n_leaves]
        display_root = min(leaves) if leaves else next(iter(self.adj))

        def build_newick(node, parent):
            children = [n for n in self.adj[node] if n != parent]
            if not children:
                return str(node)
            subtrees = [build_newick(c, node) + f":{self.length(node, c):.4f}" for c in children]
            return "(" + ",".join(subtrees) + ")"

        return build_newick(display_root, None) + ";"