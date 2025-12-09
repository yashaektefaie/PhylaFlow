import networkx as nx
import math
import io
import matplotlib.pyplot as plt
from matplotlib import animation
from Bio import Phylo

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

    split_set: iterable of bitmasks (each is one side of unrooted split)
    length_map: dict[bitmask -> branch_length] for *those* splits
    n_leaves: number of leaves (labels 1..n_leaves)
    root_leaf: leaf index used as "outgroup" to orient clusters

    Returns:
        G: networkx.Graph with nodes:
            - 1..n_leaves are leaves (attr is_leaf=True)
            - internal nodes (cluster nodes) (attr is_leaf=False)
          and edges have 'length' attribute.
    """
    full_mask = (1 << n_leaves) - 1
    root_bit = 1 << (root_leaf - 1)

    G = nx.Graph()

    # Add leaf nodes
    for i in range(1, n_leaves + 1):
        G.add_node(i, is_leaf=True, label=str(i))

    if not split_set:
        # Star tree: root node connected to all leaves
        root_node = f"R"
        G.add_node(root_node, is_leaf=False, label="R")
        for i in range(1, n_leaves + 1):
            G.add_edge(root_node, i, length=0.1)
        return G

    # Step 1: convert splits -> oriented clusters that do not contain root_leaf
    cluster_masks = set()
    cluster_to_split = {}  # cluster_mask -> original split mask
    for m in split_set:
        if m & root_bit:
            cluster = full_mask ^ m  # choose side not containing root
        else:
            cluster = m
        cluster_masks.add(cluster)
        cluster_to_split[cluster] = m  # remember which split gave this cluster

    # Include full cluster (all leaves) as root cluster
    root_cluster = full_mask
    all_clusters = list(cluster_masks)
    all_clusters.append(root_cluster)

    # Step 2: sort clusters (excluding root) by size (descending)
    cluster_list = sorted(cluster_masks, key=lambda c: popcount(c), reverse=True)

    # Step 3: create nodes for clusters & connect to parent cluster
    cluster_nodes = {}

    # Root node
    root_node = f"C_root"
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

    # Step 4: connect leaves to smallest cluster that contains them
    # For each leaf ℓ, find minimal cluster C (or root_cluster) with ℓ ∈ C
    all_clusters_with_root = [root_cluster] + cluster_list
    for leaf in range(1, n_leaves + 1):
        leaf_bit = 1 << (leaf - 1)
        candidates = [C for C in all_clusters_with_root if (C & leaf_bit) != 0]
        if not candidates:
            parent_cluster = root_cluster
        else:
            parent_cluster = min(candidates, key=lambda c: popcount(c))
        parent_node = cluster_nodes[parent_cluster]
        G.add_edge(parent_node, leaf, length=0.0)

    return G, tree_to_newick(G, root=None)

def build_geodesic_snapshots(tree1, tree2, geodesic_result, n_leaves):
    """
    Build a list of (G, description) pairs, one per topological "step" along the
    BHV geodesic, where G is a NetworkX tree for that step.

    Returns:
        snapshots: list of dicts with keys:
            - 'graph': NetworkX Graph
            - 'desc': str description (e.g. "start", "after segment 0", ...)
    """
    segments = geodesic_result["segments"]

    E1 = set(tree1.keys())
    E2 = set(tree2.keys())
    common = E1 & E2
    X = E1 - common
    Y = E2 - common

    snapshots = []

    if not segments:
        # just a single topology
        split_set = E1  # == E2 if common-only
        length_map = {}
        for m in split_set:
            if m in X:
                length_map[m] = tree1[m]
            elif m in Y:
                length_map[m] = tree2[m]
            else:
                length_map[m] = 0.5 * (tree1[m] + tree2[m])
        G, newick = build_tree_from_splits(split_set, length_map, n_leaves)
        snapshots.append({"graph": G, "desc": "single topology", "newick": newick})
        return snapshots

    # Snapshot 0: start_splits of first segment (this is tree1 topology)
    first_splits = segments[0]["start_splits"]
    length_map0 = {}
    for m in first_splits:
        if m in X:
            length_map0[m] = tree1[m]
        elif m in Y:
            length_map0[m] = tree2[m]
        else:  # common
            length_map0[m] = 0.5 * (tree1[m] + tree2[m])
    G0, newick0 = build_tree_from_splits(first_splits, length_map0, n_leaves)
    snapshots.append({"graph": G0, "desc": "start (tree1 topology)", "newick": newick0})

    # Snapshots after each segment: use end_splits
    for i, seg in enumerate(segments):
        split_set = seg["end_splits"]
        length_map = {}
        for m in split_set:
            if m in X:
                length_map[m] = tree1[m]
            elif m in Y:
                length_map[m] = tree2[m]
            else:  # common
                length_map[m] = 0.5 * (tree1[m] + tree2[m])
        G, newick = build_tree_from_splits(split_set, length_map, n_leaves)
        snapshots.append({"graph": G, "desc": f"after segment {i}", "newick": newick})

    return snapshots

def make_bhv_topology_movie(
    tree1,
    tree2,
    geodesic_result,
    n_leaves,
    filename="bhv_topology.mp4",
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
    snapshots = build_geodesic_snapshots(tree1, tree2, geodesic_result, n_leaves)

    fig, ax = plt.subplots(figsize=(6, 6))

    def init():
        ax.clear()
        ax.axis("off")
        return []

    def draw_snapshot(idx):
        ax.clear()
        ax.axis("off")
        snap = snapshots[idx]
        newick = snap["newick"]
        desc = snap["desc"]
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