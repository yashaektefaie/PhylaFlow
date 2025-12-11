import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ete3 import Tree as EteTree


class TreeFeatureTokenizer(nn.Module):
    """
    TokenGT-style tokenizer for trees.
    Each token = sum of [node/edge attribute embedding, pairwise Laplacian PE, type embedding]
    """

    def __init__(
        self,
        num_node_types,
        num_edge_types,
        hidden_dim,
        n_layers=6,
        lap_dim=16,
        lap_dropout=0.2,
        orf_dim=16,
        max_nodes=100,
        identifier=["orf", "lap"],
        concat_features=False,
    ):
        super().__init__()
        self.encoder_embed_dim = hidden_dim
        self.node_encoder = nn.Embedding(num_node_types, hidden_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(num_edge_types, hidden_dim, padding_idx=0)

        self.branch_length_encoder = nn.Linear(1, hidden_dim, bias=False)

        self.lap_dim = lap_dim
        self.lap_dropout = lap_dropout
        self.lap_encoder = nn.Linear(
            2 * lap_dim, hidden_dim, bias=False
        )  # Projects [PE_u, PE_v] to hidden_dim

        self.orf_dim = orf_dim
        self.orf_encoder = nn.Linear(2 * orf_dim, hidden_dim, bias=False)
        self.type_encoder = nn.Embedding(2, hidden_dim)  # 0=node, 1=edge
        self.identifier = identifier
        self.concat_features = concat_features
        if self.concat_features:
            self.feature_combiner = nn.Linear(3 * hidden_dim, hidden_dim)
        else:
            self.feature_combiner = nn.Identity()

        m = max(max_nodes, orf_dim)
        random_matrix = torch.randn(m, m)
        q, _ = torch.linalg.qr(random_matrix)
        self.register_buffer("orf_matrix", q)

        self.apply(lambda module: self.init_params(module, n_layers=n_layers))

    @staticmethod
    def init_params(module, n_layers):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02 / n_layers**0.5)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def sinusoidal_pos_enc(self, n_positions: int, dim: int, device):
        position = torch.arange(n_positions, device=device).float()  # [n]
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device).float() * -(math.log(10000.0) / dim)
        )  # [dim/2]
        pe = torch.zeros(n_positions, dim, device=device)  # [n, d]
        pe[:, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
        return pe
    
    def _ensure_lap_dim(self, lap: torch.Tensor) -> torch.Tensor:
        # Ensure lap has exactly self.lap_dim columns
        N, d = lap.size(0), lap.size(1)
        k = self.lap_dim
        if d == k:
            return lap
        if d > k:
            return lap[:, :k]
        # d < k
        pad = torch.zeros((N, k - d), dtype=lap.dtype, device=lap.device)
        return torch.cat([lap, pad], dim=1)

    def lap_pe_scipy(self, children: torch.Tensor, k: int, device=None) -> torch.Tensor:
        """
        children: Long[N,2], -1 for missing. Returns LapPE: Float[N,k].
        Always returns exactly k columns by requesting k+1 smallest and dropping the trivial.
        """
        dev = device or children.device
        N = int(children.size(0))
        if N == 0 or k == 0:
            return torch.zeros((N, k), dtype=torch.float32, device=dev)

        c0 = children[:, 0].detach().cpu().numpy()
        c1 = children[:, 1].detach().cpu().numpy()
        rows, cols = [], []
        if np.any(c0 >= 0):
            u = np.nonzero(c0 >= 0)[0]; v = c0[c0 >= 0].astype(np.int64)
            rows += [u, v]; cols += [v, u]
        if np.any(c1 >= 0):
            u = np.nonzero(c1 >= 0)[0]; v = c1[c1 >= 0].astype(np.int64)
            rows += [u, v]; cols += [v, u]

        if not rows:
            return torch.zeros((N, k), dtype=torch.float32, device=dev)

        row = np.concatenate(rows); col = np.concatenate(cols)
        data = np.ones_like(row, dtype=np.float64)

        A = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
        deg = np.asarray(A.sum(axis=1)).ravel()
        L = sp.diags(deg) - A

        # number of non-trivial eigenvectors the graph can actually provide
        target = min(k, max(N - 1, 0))
        if target == 0:
            return torch.zeros((N, k), dtype=torch.float32, device=dev)

        # Request one extra (k+1) to safely drop the trivial eigenvector,
        # but eigsh requires k < N; clamp accordingly.
        k_req = min(target + 1, N - 1)  # N-1 smallest (excluding dimension issues)
        # For very small N (e.g., N=2), k_req may equal target==1; that's fine.

        vals, vecs = spla.eigsh(L, k=k_req, which='SM')  # smallest magnitude

        # Sort, then drop the smallest (≈0), keep next 'target'
        order = np.argsort(vals)
        vecs = vecs[:, order]
        # It’s possible numerical ordering puts the zero not strictly first;
        # the sort ensures we drop the smallest.
        vecs = vecs[:, 1:1 + target]

        # Pad to exactly k columns
        if target < k:
            vecs = np.pad(vecs, ((0, 0), (0, k - target)), mode='constant')

        out = torch.from_numpy(vecs.astype(np.float32)).to(dev)
        # Optional TokenGT random sign flip + dropout (training only)
        if self.training and out.numel() > 0:
            sign = (torch.randint(0, 2, (1, out.size(1)), device=dev) * 2 - 1).float()
            out = out * sign
            if self.lap_dropout > 0:
                out = out * (torch.rand_like(out) > self.lap_dropout)
        return out

    def compute_laplacian_eigvecs(self, tree, k=None, device=None):
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        else:
            device = (
                torch.device(device) if not isinstance(device, torch.device) else device
            )
        k = k if k is not None else self.lap_dim
        node_list = list(tree.traverse("preorder"))

        # Check if all node names can be converted to integers
        try:
            node_list = sorted(node_list, key=lambda n: int(n.name))
        except ValueError as e:
            raise ValueError(
                f"Tree nodes must have numeric names, but found non-numeric name. "
                f"This usually means you need to call tree_numbering() function from treeVector.py first. "
                f"Original error: {e}"
            ) from e
        n = len(node_list)
        if n == 0:
            return torch.zeros((0, k), dtype=torch.float32, device=device)

        idx_map = {node: i for i, node in enumerate(node_list)}
        adj = np.zeros((n, n), dtype=np.float32)
        for node in node_list:
            for child in node.children:
                i, j = idx_map[node], idx_map[child]
                adj[i, j] = 1
                adj[j, i] = 1
        D_val = np.diag(adj.sum(axis=1))
        L = D_val - adj
        if n == 1:
            eigvecs = np.zeros((1, 0), dtype=np.float32)
        else:
            try:
                eigvals, eigvecs = np.linalg.eigh(L)
                max_eigs = min(k, n - 1 if n > 0 else 0)
                idx = np.argsort(eigvals)
                start_idx = 1 if n > 1 and max_eigs > 0 else 0
                idx_to_take = idx[start_idx : start_idx + max_eigs]
                eigvecs = (
                    eigvecs[:, idx_to_take]
                    if max_eigs > 0
                    else np.zeros((n, 0), dtype=np.float32)
                )
            except np.linalg.LinAlgError:
                eigvecs = np.zeros((n, 0), dtype=np.float32)
        if eigvecs.shape[1] < k:
            pad_width = k - eigvecs.shape[1]
            eigvecs = np.pad(eigvecs, ((0, 0), (0, pad_width)), mode="constant")
        if eigvecs.shape[1] > k:
            eigvecs = eigvecs[:, :k]
        eigvecs = torch.tensor(eigvecs, dtype=torch.float32, device=device)
        if eigvecs.size(0) > 0 and eigvecs.size(1) > 0 and self.training:
            # TokenGT: random sign flip and LapPE dropout, only during training
            sign_flip = torch.randint(0, 2, (1, eigvecs.size(1)), device=device) * 2 - 1
            eigvecs = eigvecs * sign_flip
            if self.lap_dropout > 0:
                dropout_mask = torch.rand_like(eigvecs) > self.lap_dropout
                eigvecs = eigvecs * dropout_mask
        return eigvecs
    
    def compute_laplacian_eigvecs_from_children(self, children: torch.Tensor, k: int, device=None):
        """
        children: Long[N,2], -1 for missing child.
        Returns LapPE: Float[N, k] (skipping the trivial eigenvector; padded if N-1 < k).
        Uses torch (GPU-capable). For large N, uses LOBPCG; else eigh.
        """
        device = device or next(self.parameters()).device
        children = children.to(device)
        N = children.size(0)
        if N == 0 or k == 0:
            return torch.zeros((N, k), dtype=torch.float32, device=device)

        # Build symmetric adjacency from children
        rows = []
        cols = []
        c0 = children[:, 0]
        c1 = children[:, 1]
        u = torch.arange(N, device=device)

        mask0 = c0 >= 0
        mask1 = c1 >= 0
        if mask0.any():
            rows.append(u[mask0]); cols.append(c0[mask0])
            rows.append(c0[mask0]); cols.append(u[mask0])
        if mask1.any():
            rows.append(u[mask1]); cols.append(c1[mask1])
            rows.append(c1[mask1]); cols.append(u[mask1])

        if rows:
            row = torch.cat(rows)
            col = torch.cat(cols)
            A = torch.zeros((N, N), dtype=torch.float32, device=device)
            A.index_put_((row, col), torch.ones_like(row, dtype=torch.float32), accumulate=True)
        else:
            A = torch.zeros((N, N), dtype=torch.float32, device=device)

        deg = A.sum(dim=1)
        L = torch.diag(deg) - A

        # We want the first k non-trivial eigenvectors (skip the constant one)
        need = min(k, max(N - 1, 0))
        if need == 0:
            return torch.zeros((N, k), dtype=torch.float32, device=device)

        # Choose solver
        use_lobpcg = (N >= 800)  # heuristic; tune if needed

        if use_lobpcg:
            # LOBPCG needs a symmetric positive semidefinite operator
            # Compute a few smallest eigenpairs; add tiny shift to improve stability
            X = torch.randn(N, need, device=device)
            vals, vecs = torch.lobpcg(L + 1e-6 * torch.eye(N, device=device), k=need, B=None, X=X, largest=False, tol=1e-3, maxiter=200)
            # Remove (near-)zero eigenvector if included
            # We’ll sort by eigenvalue and skip the smallest one
            order = torch.argsort(vals)
            vals = vals[order]; vecs = vecs[:, order]
            vecs_nontriv = vecs[:, 1:1+need]
        else:
            vals, vecs = torch.linalg.eigh(L)  # sorted ascending
            vecs_nontriv = vecs[:, 1:1+need]

        # Pad to k if need < k
        if need < k:
            pad = torch.zeros((N, k - need), device=device, dtype=torch.float32)
            vecs_k = torch.cat([vecs_nontriv, pad], dim=1)
        else:
            vecs_k = vecs_nontriv[:, :k]

        # TokenGT random sign flip + dropout during training
        if self.training and vecs_k.numel() > 0:
            sign = (torch.randint(0, 2, (1, vecs_k.size(1)), device=device) * 2 - 1).to(vecs_k.dtype)
            vecs_k = vecs_k * sign
            if self.lap_dropout > 0:
                vecs_k = vecs_k * (torch.rand_like(vecs_k) > self.lap_dropout)

        return vecs_k.to(torch.float32)
    
    def is_structural_tree(self, x):
        # Single structural tree: (children[N,2], root_idx[, ...])
        if not isinstance(x, (tuple, list)) or len(x) < 2:
            return False
        ch = x[0]
        return isinstance(ch, torch.Tensor) and ch.dim() == 2 and ch.size(-1) == 2

    def is_structural_pack(self, x):
        # Batched structural pack: (children[BK,Nmax,2], root_BK[BK][, ...])
        if not isinstance(x, (tuple, list)) or len(x) < 2:
            return False
        ch, rt = x[0], x[1]
        return (
            isinstance(ch, torch.Tensor) and ch.dim() == 3 and ch.size(-1) == 2
            and isinstance(rt, torch.Tensor) and rt.dim() == 1
        )
    
    def tree_to_graph_from_children(
        self,
        children: torch.Tensor,
        root_idx: int,
        branch_lengths: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
    ):
        """
        Inputs:
        children      : Long[N,2] with -1 for missing
        root_idx      : int
        branch_lengths: Optional Float[E] aligned with edge_index after creation; if None, set 1.0; root self-edge 0.0
        edge_types    : Optional Long[E]; default 1; root self-edge 0

        Returns (matches your tree_to_graph outputs):
        node_data, edge_index, edge_data, branch_lengths, node_num, edge_num,
        lap_eigvecs, leaf_mask, leaf_idx, sin_embed_node, sin_embed_edge
        """
        device = next(self.parameters()).device
        children = children.to(device)
        N = children.size(0)

        # Node types: 1 leaf, 2 internal (your convention), but zero out 0 and 1 later in leaf_mask
        is_leaf = (children[:, 0] < 0) & (children[:, 1] < 0)
        node_data = torch.where(is_leaf, torch.ones(N, device=device, dtype=torch.long), torch.full((N,), 2, device=device, dtype=torch.long))

        # Build edges (parent -> child), order by child id
        parents = []
        childs = []
        for slot in (0, 1):
            c = children[:, slot]
            mask = c >= 0
            if mask.any():
                u = torch.nonzero(mask, as_tuple=True)[0]
                v = c[mask]
                parents.append(u)
                childs.append(v)
        if parents:
            parent_cat = torch.cat(parents)
            child_cat  = torch.cat(childs)
            # order by child id
            order = torch.argsort(child_cat)
            parent_cat = parent_cat[order]
            child_cat  = child_cat[order]
            edge_index = torch.stack([parent_cat, child_cat], dim=0)  # [2, E]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        # Root self-edge
        edge_index = torch.cat([edge_index, torch.tensor([[root_idx],[root_idx]], device=device, dtype=torch.long)], dim=1)

        E = edge_index.size(1)

        # Edge data / branch lengths
        if edge_types is None:
            edge_data = torch.ones(E, dtype=torch.long, device=device)
            if E > 0:
                edge_data[-1] = 0  # root self-edge
        else:
            edge_data = edge_types.to(device)

        if branch_lengths is None:
            branch_lengths = torch.ones(E, dtype=torch.float32, device=device)
            if E > 0:
                branch_lengths[-1] = 0.0
        else:
            branch_lengths = branch_lengths.to(device)

        node_num = [N]
        edge_num = [E]

        # Positional encodings
        sin_embed_node = self.sinusoidal_pos_enc(N, self.encoder_embed_dim, device)
        sin_embed_edge = self.sinusoidal_pos_enc(E, self.encoder_embed_dim, device)

        # Laplacian PE (N x lap_dim) from children
        #lap_eigvecs = self.compute_laplacian_eigvecs_from_children(children, k=self.lap_dim, device=device)
        lap_eigvecs = self.lap_pe_scipy(children, k=self.lap_dim, device=device)
        lap_eigvecs = self._ensure_lap_dim(lap_eigvecs)

        # Leaf mask for tokens: you had nodes first then edges.
        leaf_mask_nodes = is_leaf.clone()
        # force ids 0 and 1 not to be treated as leaves in your convention
        if N > 1:
            leaf_mask_nodes[0] = False
            leaf_mask_nodes[1] = False

        leaf_mask = torch.cat([leaf_mask_nodes, torch.zeros(E, dtype=torch.bool, device=device)], dim=0)
        # leaf_idx = preorder indices for tips 2..num_leaves (here: node IDs themselves are already integers)
        # If your IDs are consistent (0..), then leaf indices are simply the node ids of leaves excluding 0,1.
        leaf_ids = torch.nonzero(leaf_mask_nodes, as_tuple=True)[0]
        leaf_ids = leaf_ids[leaf_ids >= 2]
        leaf_idx = leaf_ids.to(torch.long)

        return (
            node_data,
            edge_index,
            edge_data,
            branch_lengths,
            node_num,
            edge_num,
            lap_eigvecs,
            leaf_mask,
            leaf_idx,
            sin_embed_node,
            sin_embed_edge,
        )

    def tree_to_graph(self, tree):
        """Node list is the tree in preorder"""
        node_list = list(tree.traverse("preorder"))
        node_list = sorted(node_list, key=lambda n: int(n.name))
        device = torch.device('cpu')

        name_to_type = {
            int(node.name): (1 if node.is_leaf() else 2) for node in node_list
        }
        num_leaves = sum(1 for node in node_list if node.is_leaf())
        name_to_preorder_idx = {int(node.name): i for i, node in enumerate(node_list)}
        node_data = torch.tensor(
            [name_to_type[int(i.name)] for i in node_list],
            dtype=torch.long,
            device=device,
        )

        # Build edge list: each edge is (parent, child), ordered by child node name
        edge_tuples = []
        edge_data_list = []
        branch_length_list = []
        for node in node_list:
            for child in node.children:
                edge_tuples.append((int(node.name), int(child.name)))
                edge_type = getattr(child, "edge_type_id", 1)
                edge_data_list.append(edge_type)
                branch_length = getattr(child, "dist", 1.0)
                branch_length_list.append(branch_length)
        # Add root node connection: root is node with no parent
        root_nodes = [n for n in node_list if n.up is None]

        for root in root_nodes:
            # Convention: connect root to itself, length 0
            edge_tuples.append((int(root.name), int(root.name)))
            edge_data_list.append(0)  # or a special edge type for root
            branch_length_list.append(0.0)

        # Order edges by child node name
        edge_order = sorted(range(len(edge_tuples)), key=lambda i: edge_tuples[i][1])
        edge_tuples = [edge_tuples[i] for i in edge_order]
        edge_data_list = [edge_data_list[i] for i in edge_order]
        branch_length_list = [branch_length_list[i] for i in edge_order]
        edge_index_list = [
            [name_to_preorder_idx[parent], name_to_preorder_idx[child]]
            for parent, child in edge_tuples
        ]
        edge_index = (
            torch.tensor(edge_index_list, dtype=torch.long, device=device).t()
            if edge_index_list
            else torch.zeros(2, 0, dtype=torch.long, device=device)
        )
        edge_data = (
            torch.tensor(edge_data_list, dtype=torch.long, device=device)
            if edge_data_list
            else torch.zeros(0, dtype=torch.long, device=device)
        )
        branch_lengths = (
            torch.tensor(branch_length_list, dtype=torch.float32, device=device)
            if branch_length_list
            else torch.zeros(0, dtype=torch.float32, device=device)
        )
        node_num = [len(node_list)]
        edge_num = [edge_index.size(1)]
        sin_embed_node = self.sinusoidal_pos_enc(
            node_num[0], self.encoder_embed_dim, node_data.device
        )
        sin_embed_edge = self.sinusoidal_pos_enc(
            edge_num[0], self.encoder_embed_dim, edge_data.device
        )
        lap_eigvecs = self.compute_laplacian_eigvecs(
            tree, k=self.lap_dim, device=node_data.device
        )
        leaf_mask_nodes = torch.tensor(
            [name_to_type[int(node.name)] == 1 for node in node_list],
            device=device,
        )
        leaf_mask_nodes[name_to_preorder_idx[0]] = 0
        leaf_mask_nodes[name_to_preorder_idx[1]] = 0
        leaf_mask = torch.cat(
            [
                leaf_mask_nodes,
                torch.zeros(edge_num[0], dtype=torch.bool, device=node_data.device),
            ]
        )
        leaf_idx = torch.tensor(
            [name_to_preorder_idx[i] for i in range(2, num_leaves)],
            dtype=torch.long,
            device=device,
        )
        return (
            node_data,
            edge_index,
            edge_data,
            branch_lengths,
            node_num,
            edge_num,
            lap_eigvecs,
            leaf_mask,
            leaf_idx,
            sin_embed_node,
            sin_embed_edge,
        )

    def forward(self, trees):

        if isinstance(trees, (str, EteTree)):
            trees = [trees]

        # List of Newick strings / ETE3 Trees
        if isinstance(trees, list) and len(trees) > 0:
            if all(isinstance(t, (str, EteTree)) for t in trees):
                trees = [self._newick_to_structural(t) for t in trees]

        # We take in a list of trees where each entry is this format (children, root_idx[, branch_lengths][, edge_types])
        if self.is_structural_pack(trees):
            children_BK = trees[0]
            root_BK     = trees[1]
            branch_BK   = trees[2] if len(trees) >= 3 else None
            edge_BK     = trees[3] if len(trees) >= 4 else None

            BK, Nmax, _ = children_BK.shape
            # Expand the packed tensors into a Python list of per-item tuples.
            # (Keeping the simple loop here is fine—heavy work is still tensorized inside.)
            trees = []
            for i in range(BK):
                ch = children_BK[i]              # [Nmax, 2] with padding rows at bottom as [-1, -1]
                rt = int(root_BK[i].item())

                # Infer actual N: any row that has a child, or is referenced as a child, or is the root.
                used = (ch[:,0] >= 0) | (ch[:,1] >= 0)  # rows that parent at least one child
                kids0 = ch[ch[:,0] >= 0, 0]
                kids1 = ch[ch[:,1] >= 0, 1]
                child_refs = torch.zeros(ch.size(0), dtype=torch.bool, device=ch.device)
                if kids0.numel(): child_refs[kids0] = True
                if kids1.numel(): child_refs[kids1] = True
                used = used | child_refs
                used[rt] = True

                if used.any():
                    N_i = int(used.nonzero(as_tuple=True)[0].max().item()) + 1
                else:
                    N_i = 0  # degenerate (shouldn’t happen for trees)

                ch = ch[:N_i]                     # <-- CRITICAL: drop padded rows
                tup = [ch, rt]
                if branch_BK is not None: tup.append(branch_BK[i][:, :])  # keep aligned if you carry per-edge BLs
                if edge_BK   is not None: tup.append(edge_BK[i][:])
                trees.append(tuple(tup))
        elif not isinstance(trees, list):
            trees = [trees]

        batch_size = len(trees)
        if batch_size == 0:
            device = next(self.parameters()).device
            return (
                torch.zeros(
                    (0, 0, self.encoder_embed_dim), device=device, dtype=torch.float
                ),
                torch.zeros((0, 0), device=device, dtype=torch.bool),
                torch.zeros((0, 0, 2), device=device, dtype=torch.long),
                torch.zeros((0, 0), device=device, dtype=torch.bool),
                [],
            )

        if batch_size == 1:
            return self._forward_single_tree(trees[0])

        # Process each tree and collect results
        batch_features = []
        batch_padding_masks = []
        batch_padded_indices = []
        batch_leaf_masks = []
        batch_leaf_indices = []
        max_tokens = 0

        for tree in trees:
            features, padding_mask, padded_index, leaf_mask_single, leaf_idx_single = (
                self._forward_single_tree(tree)
            )
            # Remove batch dimension from single tree output
            features = features.squeeze(0)
            padding_mask = padding_mask.squeeze(0)
            padded_index = padded_index.squeeze(0)

            batch_features.append(features)
            batch_padding_masks.append(padding_mask)
            batch_padded_indices.append(padded_index)
            batch_leaf_masks.append(leaf_mask_single)
            batch_leaf_indices.append(leaf_idx_single)
            max_tokens = max(max_tokens, features.size(0))

        device = batch_features[0].device

        # Pad sequences to max length
        padded_features = torch.zeros(
            (batch_size, max_tokens, self.encoder_embed_dim), device=device
        )
        padded_masks = torch.ones(
            (batch_size, max_tokens), device=device, dtype=torch.bool
        )  # True = padded
        padded_indices = torch.zeros(
            (batch_size, max_tokens, 2), device=device, dtype=torch.long
        )
        padded_leaf_masks = torch.zeros(
            (batch_size, max_tokens), device=device, dtype=torch.bool
        )

        for i, (
            features,
            mask,
            indices,
            leaf_mask_single,
            leaf_idx_single,
        ) in enumerate(
            zip(
                batch_features,
                batch_padding_masks,
                batch_padded_indices,
                batch_leaf_masks,
                batch_leaf_indices,
            )
        ):
            seq_len = features.size(0)
            if seq_len > 0:
                padded_features[i, :seq_len] = features
                padded_masks[i, :seq_len] = mask
                padded_indices[i, :seq_len] = indices
                padded_leaf_masks[i, :seq_len] = leaf_mask_single

        return (
            padded_features,
            padded_masks,
            padded_indices,
            padded_leaf_masks,
            batch_leaf_indices,
        )

    def _forward_single_tree(self, tree_or_tuple):
        device = next(self.parameters()).device
        if self.is_structural_tree(tree_or_tuple):
            # (children, root_idx[, branch_lengths][, edge_types])
            children = tree_or_tuple[0].to(device)
            root_idx = int(tree_or_tuple[1])
            branch_lengths = tree_or_tuple[2].to(device) if len(tree_or_tuple) >= 3 and tree_or_tuple[2] is not None else None
            edge_types     = tree_or_tuple[3].to(device) if len(tree_or_tuple) >= 4 and tree_or_tuple[3] is not None else None
            (node_data, edge_index, edge_data, branch_lengths, node_num, edge_num,
            lap_pe, leaf_mask, leaf_idx, sin_embed_node, sin_embed_edge) = \
                self.tree_to_graph_from_children(children, root_idx, branch_lengths, edge_types)
        else:
            # Legacy: ETE3 tree (kept for compatibility)
            # Convert once using your new fast builders:
            #   Use tau_to_arrays if you have τ, or write a small extractor from ETE3 to children.
            # If you *must* support ETE3, add a tiny adapter here. Otherwise, raise:
            raise TypeError("Pass structural trees: (children, root_idx[, branch_lengths][, edge_types])")

        if not node_num[0]:
            return (
                torch.zeros((1, 0, self.encoder_embed_dim), device=device, dtype=torch.float),
                torch.zeros((1, 0), device=device, dtype=torch.bool),
                torch.zeros((1, 0, 2), device=device, dtype=torch.long),
                torch.zeros((0,), device=device, dtype=torch.bool),
                torch.zeros((0,), device=device, dtype=torch.long),
            )

        return self._process_and_pack_single(
            node_data, edge_index, edge_data, branch_lengths,
            node_num, edge_num, lap_pe, leaf_mask, leaf_idx,
            sin_embed_node, sin_embed_edge, device
        )

    def _process_and_pack_single(
        self, node_data, edge_index, edge_data, branch_lengths,
        node_num, edge_num, lap_pe, leaf_mask, leaf_idx, sin_embed_node, sin_embed_edge, device
    ):
        # --- (identical to your _process_single_tree, with the vectorized LapPE) ---
        node_indices = torch.arange(node_num[0], device=device)
        node_attr_embedding = self.node_encoder(node_data) + sin_embed_node
        node_pairs = torch.stack([node_indices, node_indices], dim=1)

        edge_attr_embedding = self.edge_encoder(edge_data)
        if edge_attr_embedding.size(0) > 0:
            branch_length_feat = self.branch_length_encoder(branch_lengths.unsqueeze(1).to(device))
            edge_attr_embedding = edge_attr_embedding + branch_length_feat + sin_embed_edge
        edge_pairs = edge_index.t()

        full_attr_embedding = torch.cat([node_attr_embedding, edge_attr_embedding], dim=0)
        full_padded_index   = torch.cat([node_pairs, edge_pairs], dim=0)

        type_ids = torch.cat([
            torch.zeros(node_num[0], dtype=torch.long, device=device),
            torch.ones(edge_num[0], dtype=torch.long, device=device),
        ])
        type_embedding = self.type_encoder(type_ids)

        u = full_padded_index[:, 0]
        v = full_padded_index[:, 1]
        pos_pe_concat = torch.cat([lap_pe[u], lap_pe[v]], dim=1)
        pos_embedding = self.lap_encoder(pos_pe_concat)

        if self.concat_features:
            final_token_features = self.feature_combiner(torch.cat([full_attr_embedding, type_embedding, pos_embedding], dim=1))
        else:
            final_token_features = full_attr_embedding + type_embedding + pos_embedding

        padding_mask = torch.zeros(final_token_features.size(0), dtype=torch.bool, device=device)

        # Pack batch dim = 1
        return (
            final_token_features.unsqueeze(0),
            padding_mask.unsqueeze(0),
            full_padded_index.unsqueeze(0),
            leaf_mask,
            leaf_idx,
        )

    def _process_single_tree(
        self,
        node_data,
        edge_index,
        edge_data,
        branch_lengths,
        node_num,
        edge_num,
        lap_pe,
        leaf_mask,
        leaf_idx,
        sin_embed_node,
        sin_embed_edge,
        device,
    ):
        """Process a single tree and return token features"""
        # Node tokens
        node_indices = torch.arange(node_num[0], device=device)  # 0 … N-1
        node_attr_embedding = self.node_encoder(node_data) + sin_embed_node
        node_pairs = torch.stack([node_indices, node_indices], dim=1)  # [N,2]

        # Edge tokens
        edge_attr_embedding = self.edge_encoder(edge_data)
        if edge_attr_embedding.size(0) > 0:
            branch_length_feat = self.branch_length_encoder(
                branch_lengths.unsqueeze(1).to(device)
            )
            edge_attr_embedding = (
                edge_attr_embedding + branch_length_feat + sin_embed_edge
            )
        edge_pairs = edge_index.t()  # shape [num_edges, 2]

        full_attr_embedding = torch.cat(
            [node_attr_embedding, edge_attr_embedding], dim=0
        )
        full_padded_index = torch.cat(
            [node_pairs, edge_pairs], dim=0
        )  # [num_tokens, 2]

        # Type encoding: 0 for node, 1 for edge
        type_ids = torch.cat(
            [
                torch.zeros(node_num[0], dtype=torch.long, device=device),
                torch.ones(edge_num[0], dtype=torch.long, device=device),
            ]
        )
        type_embedding = self.type_encoder(type_ids)

        # LapPE encoding for each token (projected to hidden_dim)
        u = full_padded_index[:, 0]
        v = full_padded_index[:, 1]
        pos_pe_concat = torch.cat([lap_pe[u], lap_pe[v]], dim=1)  # [num_tokens, 2*lap_dim]
        pos_embedding = self.lap_encoder(pos_pe_concat)

        if self.concat_features:
            final_token_features = self.feature_combiner(
                torch.cat([full_attr_embedding, type_embedding, pos_embedding], dim=1)
            )
        else:
            final_token_features = self.feature_combiner(
                full_attr_embedding + type_embedding + pos_embedding
            )
        #Don't need this line since laready done above...
        # final_token_features = full_attr_embedding + type_embedding + pos_embedding

        # Create padding mask (False = valid tokens, True = padding)
        padding_mask = torch.zeros(
            final_token_features.size(0),
            dtype=torch.bool,
            device=device,
        )

        return (
            final_token_features,
            padding_mask,
            full_padded_index,
            leaf_mask,
            leaf_idx,
        )
    
    def _newick_to_structural(
        self,
        tree_or_str,
        default_edge_type: int = 1,
    ):
        """
        Convert a Newick string or an ETE3 Tree into the structural format:
        (children[N,2], root_idx, branch_lengths[E+1], edge_types[E+1])

        - children[i, :] = [child0, child1] with -1 for missing.
        - root_idx is the index of the root node.
        - branch_lengths & edge_types are aligned to the edge_index that
          tree_to_graph_from_children will construct (i.e., sorted by child id),
          plus a final entry for the root self-edge.
        """
        # Parse if needed
        if isinstance(tree_or_str, str):
            if 'C(0)' in tree_or_str:
                tree_or_str = tree_or_str.replace('C(0)', '"C"')
            t = EteTree(tree_or_str, format=1, quoted_node_names=True)
        else:
            # assume already an ETE3 Tree-like object
            t = tree_or_str

        # Preorder traversal and index assignment
        nodes = list(t.traverse("preorder"))
        idx_map = {node: i for i, node in enumerate(nodes)}
        N = len(nodes)

        device = next(self.parameters()).device

        # children[N,2] initialized to -1
        children = torch.full(
            (N, 2),
            -1,
            dtype=torch.long,
            device=device,
        )

        parent_list = []
        child_list = []
        branch_list = []
        edge_type_list = []

        for parent in nodes:
            p_idx = idx_map[parent]
            child_nodes = list(parent.children)

            if len(child_nodes) > 2:
                raise ValueError(
                    "Non-binary tree encountered in _newick_to_structural; "
                    "children tensor only supports up to 2 children per node."
                )

            for slot, child in enumerate(child_nodes):
                c_idx = idx_map[child]
                children[p_idx, slot] = c_idx

                parent_list.append(p_idx)
                child_list.append(c_idx)

                # branch length stored on child in ETE
                bl = getattr(child, "dist", 1.0)
                branch_list.append(float(bl))

                et = getattr(child, "edge_type_id", default_edge_type)
                edge_type_list.append(int(et))

        # Now match the exact edge ordering used in tree_to_graph_from_children:
        # sort by child id.
        if len(child_list) > 0:
            child_arr = np.array(child_list, dtype=np.int64)
            order = np.argsort(child_arr)

            branch_arr = np.array(branch_list, dtype=np.float32)[order]
            edge_type_arr = np.array(edge_type_list, dtype=np.int64)[order]
        else:
            branch_arr = np.zeros((0,), dtype=np.float32)
            edge_type_arr = np.zeros((0,), dtype=np.int64)

        # Add root self-edge as the last entry, matching tree_to_graph_from_children
        root_idx = idx_map[t]
        branch_arr = np.concatenate([branch_arr, np.array([0.0], dtype=np.float32)])
        edge_type_arr = np.concatenate([edge_type_arr, np.array([0], dtype=np.int64)])

        branch_lengths = torch.tensor(
            branch_arr,
            dtype=torch.float32,
            device=device,
        )
        edge_types = torch.tensor(
            edge_type_arr,
            dtype=torch.long,
            device=device,
        )

        return (children, root_idx, branch_lengths, edge_types)
