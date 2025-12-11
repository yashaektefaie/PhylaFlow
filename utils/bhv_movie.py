import networkx as nx
import math
import io
import matplotlib.pyplot as plt
from matplotlib import animation
from Bio import Phylo
import random

def popcount(x: int) -> int:
    return x.bit_count()  # Python 3.8+: or bin(x).count("1")

def tree_to_newick(G: nx.Graph, root=None, leaf_label_attr="label") -> str:
    """
    Convert a NetworkX tree into a Newick string.

    Assumptions:
      - G is an undirected tree.
      - Edge lengths (branch lengths) are stored in edge attribute 'length'.
      - Leaves have some label (default attr 'label'); if missing, node id is used.
      - 'root' is either a node in G or None.
        If None, we'll try to find a node with label 'root', else pick an arbitrary node.

    Returns:
      Newick string ending with a semicolon.
    """

    if root is None:
        # Prefer the node labeled 'root' (like your C_root)
        for n, data in G.nodes(data=True):
            if data.get(leaf_label_attr, None) == "root":
                root = n
                break
        else:
            # fallback: arbitrary node
            root = next(iter(G.nodes))

    # Recursive DFS to build Newick
    def build_subtree(node, parent):
        # children are neighbors except the parent
        children = [nbr for nbr in G.neighbors(node) if nbr != parent]

        if not children:
            # Leaf: return its label only (branch length added by parent)
            label = G.nodes[node].get(leaf_label_attr, str(node))
            return label
        else:
            # Internal node: recursively build children
            parts = []
            for child in children:
                child_subtree = build_subtree(child, node)
                blen = G.edges[node, child].get("length", 1.0)
                parts.append(f"{child_subtree}:{blen}")
            return "(" + ",".join(parts) + ")"

    newick_body = build_subtree(root, parent=None)
    return newick_body + ";"

def build_tree_from_splits(split_set, length_map, n_leaves, root_leaf=1):
    """
    Build a NetworkX tree from a compatible set of splits (bitmasks).

    split_set: iterable of bitmasks (each is one side of an unrooted split)
    length_map: dict[bitmask -> branch_length] for those splits
    n_leaves: number of leaves (labels 1..n_leaves)
    root_leaf: leaf index used as "outgroup" to orient clusters

    Returns:
        G: networkx.Graph with nodes:
            - 1..n_leaves are leaves (attr is_leaf=True)
            - internal nodes (cluster nodes) (attr is_leaf=False)
          and edges have 'length' attribute.
        newick: string
    """
    full_mask = (1 << n_leaves) - 1
    root_bit = 1 << (root_leaf - 1)

    G = nx.Graph()

    # Add leaf nodes
    for i in range(1, n_leaves + 1):
        G.add_node(i, is_leaf=True, label=str(i))

    # Special case: no splits => star tree; use leaf-lengths if present
    if not split_set:
        root_node = "R"
        G.add_node(root_node, is_leaf=False, label="R")
        for i in range(1, n_leaves + 1):
            m_leaf = 1 << (i - 1)
            length = float(length_map.get(m_leaf, 0.1))
            G.add_edge(root_node, i, length=length)
        return G, tree_to_newick(G, root=None)

    # ------------------------------------------------------------------
    # STEP 0: separate internal splits from pendant (leaf) splits
    # ------------------------------------------------------------------
    leaf_lengths = {i: 0.0 for i in range(1, n_leaves + 1)}
    internal_splits = []

    for m in split_set:
        A = m
        B = full_mask ^ m
        sizeA = popcount(A)
        sizeB = popcount(B)
        length = float(length_map.get(m, 0.5))

        if sizeA == 1 or sizeB == 1:
            # Pendant edge: one side is a single leaf
            if sizeA == 1:
                leaf_mask = A
            else:
                leaf_mask = B
            leaf_index = (leaf_mask.bit_length() - 1) + 1  # 1-based index
            leaf_lengths[leaf_index] = length
            # Do NOT include this in internal split system
        else:
            internal_splits.append(m)

    # If there are no internal splits, just connect leaves to a root
    if not internal_splits:
        root_node = "R"
        G.add_node(root_node, is_leaf=False, label="R")
        for i in range(1, n_leaves + 1):
            G.add_edge(root_node, i, length=leaf_lengths.get(i, 0.0))
        return G, tree_to_newick(G, root=None)

    # ------------------------------------------------------------------
    # STEP 1: convert internal splits -> oriented clusters (away from root)
    # ------------------------------------------------------------------
    cluster_masks = set()
    cluster_to_split = {}

    for m in internal_splits:
        if m & root_bit:
            # take side that does NOT contain root_leaf
            cluster = full_mask ^ m
        else:
            cluster = m
        cluster_masks.add(cluster)
        cluster_to_split[cluster] = m

    # Include full cluster (all leaves) as root cluster
    root_cluster = full_mask
    all_clusters = list(cluster_masks)
    all_clusters.append(root_cluster)

    # STEP 2: sort clusters (excluding root) by size (descending)
    cluster_list = sorted(cluster_masks, key=lambda c: popcount(c), reverse=True)

    # STEP 3: create nodes for clusters & connect to parent cluster
    cluster_nodes = {}

    # Root node
    root_node = "C_root"
    G.add_node(root_node, is_leaf=False, label="root")
    cluster_nodes[root_cluster] = root_node

    # For each cluster, find minimal parent cluster that strictly contains it
    for C in cluster_list:
        # identify parent cluster P: smallest cluster with C ⊂ P
        parent = root_cluster
        parent_size = popcount(root_cluster)
        sizeC = popcount(C)
        for P in all_clusters:
            if P == C:
                continue
            # C subset P?
            if (C & ~P) == 0:
                sizeP = popcount(P)
                if sizeC < sizeP < parent_size:
                    parent = P
                    parent_size = sizeP

        # make node for this cluster
        nodeC = f"C_{C}"
        G.add_node(nodeC, is_leaf=False, label=f"C({sizeC})")
        cluster_nodes[C] = nodeC

        # ensure parent node exists
        nodeP = cluster_nodes[parent]

        # internal edge length determined by split that induced this cluster
        split_mask = cluster_to_split[C]
        length = float(length_map.get(split_mask, 0.5))
        G.add_edge(nodeP, nodeC, length=length)

    # ------------------------------------------------------------------
    # STEP 4: connect leaves to smallest cluster that contains them
    #         using the *pendant* lengths
    # ------------------------------------------------------------------
    all_clusters_with_root = [root_cluster] + cluster_list
    for leaf in range(1, n_leaves + 1):
        leaf_bit = 1 << (leaf - 1)
        candidates = [C for C in all_clusters_with_root if (C & leaf_bit) != 0]
        if not candidates:
            parent_cluster = root_cluster
        else:
            parent_cluster = min(candidates, key=lambda c: popcount(c))
        parent_node = cluster_nodes[parent_cluster]
        length = float(leaf_lengths.get(leaf, 0.0))
        G.add_edge(parent_node, leaf, length=length)

    return G, tree_to_newick(G, root=None)

def make_bhv_topology_movie(
    geodesic_result,
    n_leaves,
    filename="bhv_topology.mp4",
    F=10,
    fps=1,
    dpi=150,
):
    """
    Make a simple movie where each frame is a NetworkX drawing of the tree
    topology at a different step along the BHV geodesic.

    One frame per segment boundary:
      - frame 0: start topology (tree1-like)
      - frame i>0: topology after segment i-1
    """
    snapshots = []
    for k in range(F):
        u = k / (F - 1)
        G, newick, info = sample_tree_along_geodesic(geodesic_result, n_leaves, u=u)
        snapshots.append(newick)

    # snapshots = build_geodesic_snapshots(tree1, tree2, geodesic_result, n_leaves)

    fig, ax = plt.subplots(figsize=(6, 6))

    def init():
        ax.clear()
        ax.axis("off")
        return []

    def draw_snapshot(idx):
        ax.clear()
        ax.axis("off")
        snap = snapshots[idx]
        newick = snap
        desc = f"Frame {idx} (u={idx/(len(snapshots)-1):.2f})"
        print(newick)

        # Render the Newick tree with Biopython
        tree = Phylo.read(io.StringIO(newick), "newick")
        # Label each edge with its branch length if present
        Phylo.draw(
            tree,
            axes=ax,
            do_show=False,
            branch_labels=lambda clade: (f"{clade.branch_length:.1f}" if clade.branch_length not in (None, 0.0) else None),
        )
        ax.set_title(desc)
        return []

    anim = animation.FuncAnimation(
        fig,
        draw_snapshot,
        init_func=init,
        frames=len(snapshots),
        interval=1000.0 / fps,
        blit=False,
    )

    # if filename.endswith(".gif"):
    anim.save(filename, writer="pillow", fps=fps, dpi=dpi)
    # else:
    #     Writer = animation.writers["ffmpeg"]
    #     writer = Writer(fps=fps, bitrate=1800)
    #     anim.save(filename, writer=writer, dpi=dpi)

    plt.close(fig)
    print(f"Saved BHV topology movie to {filename}")

def sample_tree_along_geodesic(geodesic_result, n_leaves, u=None):
    """
    Sample a tree at a *continuous* position along a BHV geodesic.

    geodesic_result["segments"] must be a list of dicts with keys:
        - "length": float BHV length of the segment
        - "splits": set of split bitmasks present in this orthant
        - "start_lengths": dict[split -> length at start of segment]
        - "end_lengths":   dict[split -> length at end of segment]

    n_leaves: number of leaves in the tree
    u: scalar in [0,1]; if None, sampled uniformly

    Returns:
        G: NetworkX tree at that point
        newick: Newick string for that tree
        info: dict with where we are along the path
    """
    segments = geodesic_result["segments"]
    if u is None:
        u = random.random()

    # 1) total BHV length
    total_L = sum(seg["length"] for seg in segments)
    if total_L == 0:
        # degenerate: return start topology
        seg0 = segments[0]
        split_set = set(seg0["splits"])
        length_map = dict(seg0["start_lengths"])
        G, newick = build_tree_from_splits(split_set, length_map, n_leaves)
        return G, newick, {"u": u, "segment_index": 0, "alpha": 0.0}

    # 2) convert u -> arc length
    s = u * total_L

    # 3) find segment
    cum = 0.0
    seg_idx = None
    offset = 0.0
    for i, seg in enumerate(segments):
        if s <= cum + seg["length"] or i == len(segments) - 1:
            seg_idx = i
            offset = s - cum
            break
        cum += seg["length"]

    seg = segments[seg_idx]
    L_seg = seg["length"]
    alpha = 0.0 if L_seg == 0 else offset / L_seg

    # 4) interpolate lengths for splits in this orthant
    curr_lengths = {}
    for m in list(seg["start_lengths"].keys()):
        l0 = seg["start_lengths"][m]
        l1 = seg["end_lengths"][m]
        curr_lengths[m] = (1.0 - alpha) * l0 + alpha * l1

    # 5) drop ~zero edges
    eps = 1e-8
    split_set = {m for m, L in curr_lengths.items() if L > eps}
    length_map = {m: L for m, L in curr_lengths.items() if L > eps}

    # 6) build tree
    G, newick = build_tree_from_splits(split_set, length_map, n_leaves)

    info = {
        "u": u,
        "s": s,
        "segment_index": seg_idx,
        "alpha": alpha,
        "total_length": total_L,
        "velocity": seg['velocity']
    }

    return G, newick, info


if __name__ == "__main__":
    # Simple test with two random trees
    n_leaves = 5
    from utils.random_tree import RandomTree, Tree
    from utils.bhv_utils import BHVEncoder

    rt1 = RandomTree(n_leaves)
    newick1 = rt1.to_newick()
    print("Generated random tree 1:", newick1)
    # rt2 = RandomTree(n_leaves)
    # newick2 = rt2.to_newick()

    T1 = Tree(newick1)
    print("Parsed tree 1 now see it as:", T1)
    # T2 = Tree(newick2)

    # print("Tree 1 Newick:", newick1)
    # print("Tree 2 Newick:", newick2)

    enc = BHVEncoder()
    t1_edge_mask, t1_edge_length = enc.return_BHV_encoding(T1)
    # t2_edge_mask, t2_edge_length = enc.return_BHV_encoding(T2)

    tree1 = {m: l for m, l in zip(t1_edge_mask, t1_edge_length)}
    # tree2 = {m: l for m, l in zip(t2_edge_mask, t2_edge_length)}

    recovered_newick_1 = build_tree_from_splits(t1_edge_mask, tree1, n_leaves)[1]
    # recovered_newick_2 = build_tree_from_splits(t2_edge_mask, tree2, n_leaves)[1]
    print("Recovered Tree 1 Newick from BHV:", recovered_newick_1)
    import pdb; pdb.set_trace()