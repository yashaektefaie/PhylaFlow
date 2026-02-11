import math
from math import log, sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.tree_transformer_layers import (
    DropPath,
    MultiheadAttention,
    MultiheadPerformerAttention,
)
from model.treeTokenizer import TreeFeatureTokenizer
from utils.utils import get_batch_polytomy_indices


# TokenGT parameter initialization
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02 / sqrt(n_layers))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "padding_idx") and module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


class PairwiseMergeHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        in_dim = 4 * d_model  # [hi, hj, |hi-hj|, hi*hj]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),  # logit
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: [G, D]
        returns logits: [G, G] with -inf on diagonal (no self-merge)
        """
        G, D = H.shape
        hi = H.unsqueeze(1).expand(G, G, D)  # [G, G, D]
        hj = H.unsqueeze(0).expand(G, G, D)  # [G, G, D]
        feats = torch.cat([hi, hj, (hi - hj).abs(), hi * hj], dim=-1)  # [G, G, 4D]
        logits = self.mlp(feats).squeeze(-1)  # [G, G]

        # disallow i==j
        logits = logits.masked_fill(
            torch.eye(G, device=H.device, dtype=torch.bool), float("-inf")
        )
        return logits


class TreeGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        n_heads,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        drop_path=0.0,
        use_performer=False,
        performer_nb_features=None,
        performer_generalized_attention=False,
        layernorm_style="prenorm",
        n_layers=12,
    ):
        super().__init__()
        self.layernorm_style = layernorm_style
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if use_performer and performer_nb_features is not None:
            self.self_attn = MultiheadPerformerAttention(
                embed_dim,
                n_heads,
                performer_nb_features=performer_nb_features,
                performer_generalized_attention=performer_generalized_attention,
                dropout=dropout,
                n_layers=n_layers,
            )
        else:
            self.self_attn = MultiheadAttention(
                embed_dim,
                n_heads,
                dropout=attention_dropout,
                n_layers=n_layers,
            )
        self.dropout_module = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(activation_dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.apply(lambda m: init_params(m, n_layers))

    def forward(self, x, padding_mask=None):  # padding_mask is key_padding_mask
        attn = None
        if self.layernorm_style == "prenorm":
            residual = x
            x_norm = self.self_attn_layer_norm(x)
            x_attn, attn = self.self_attn(x_norm, key_padding_mask=padding_mask)
            x_attn = self.dropout_module(x_attn)
            x = self.drop_path1(x_attn)
            x = residual + x

            residual = x
            x_norm = self.final_layer_norm(x)
            x_ffn = self.feedforward(x_norm)
            x = self.drop_path2(x_ffn)
            x = residual + x
        elif self.layernorm_style == "postnorm":
            residual = x
            x_attn, attn = self.self_attn(x, key_padding_mask=padding_mask)
            x_attn = self.dropout_module(x_attn)
            x = self.drop_path1(x_attn)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x_ffn = self.feedforward(x)
            x = self.drop_path2(x_ffn)
            x = residual + x
            x = self.final_layer_norm(x)
        else:
            raise NotImplementedError
        return x, attn


class TreeDenoiserTokenGT(nn.Module):
    def __init__(
        self,
        num_node_types,
        num_edge_types,
        embed_dim=768,
        n_layers=12,
        n_heads=32,
        output_dim=1,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        drop_path_rate=0.1,
        use_performer=True,
        performer_nb_features=64,
        performer_generalized_attention=True,
        layernorm_style="prenorm",
        tokenizer_lap_dim=16,  # TreeFeatureTokenizer
        tokenizer_lap_dropout=0.2,  # TreeFeatureTokenizer
        tokenizer_n_layers=6,  # TreeFeatureTokenizer
        phyla_dim=256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.tokenizer = TreeFeatureTokenizer(
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            hidden_dim=embed_dim,
            n_layers=tokenizer_n_layers,
            lap_dim=tokenizer_lap_dim,
            lap_dropout=tokenizer_lap_dropout,
            # concat_features=True,  # Use concatenation of features
        )
        # [graph] token and [null] token
        self.graph_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.graph_token, mean=0.0, std=0.02)
        nn.init.normal_(self.null_token, mean=0.0, std=0.02)

        # self.embed_proj = nn.Linear(embed_dim, embed_dim)

        # Phyla projection
        self.phyla_proj = nn.Linear(phyla_dim, embed_dim)

        # Time projection
        self.time_embed_dim = embed_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, self.time_embed_dim),
            nn.GELU(),
            nn.Linear(self.time_embed_dim, embed_dim),
        )

        # Transformer encoder
        dprates = [
            drop_path_rate * i / (n_layers - 1) if n_layers > 1 else 0.0
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(
            [
                TreeGraphEncoderLayer(
                    embed_dim=embed_dim,
                    ffn_dim=embed_dim * 4,
                    n_heads=n_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    drop_path=dprates[i],
                    use_performer=use_performer,
                    performer_nb_features=performer_nb_features,
                    performer_generalized_attention=performer_generalized_attention,
                    layernorm_style=layernorm_style,
                    n_layers=n_layers,
                )
                for i in range(n_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.pairwise_head = PairwiseMergeHead(
            d_model=embed_dim, hidden=embed_dim, dropout=dropout
        )
        self.group_head = nn.Linear(embed_dim, 1)

        self.apply(lambda m: init_params(m, n_layers))

    def create_sinusoidal_embedding(self, t, dim):
        """
        Creates a stable sinusoidal embedding based on the original Transformer paper.
        """
        # t is a tensor of shape [B]
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32)
        else:
            t = t.float()

        device = self.graph_token.device
        t = t.to(device)

        half_dim = dim // 2

        # Denominator term: 10000^(2i/d)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / (half_dim - 1))
        )

        # Argument to sin/cos: t / 10000^(2i/d)
        emb = t.unsqueeze(-1) / emb.unsqueeze(0)

        # Final embedding
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))  # Pad last dimension

        return emb

    def forward(
        self,
        tokenized_tree_batch,
        t=None,
        phyla_embeddings=None,
        return_all_tokens=True,
        return_leafs_only=False,
        return_edges_only=False,
        autoregressive=False,
    ):
        # Tree is this format now: (children, root_idx[, branch_lengths][, edge_types])
        # Handle both single tree and batch of trees
        # is_single_tree = not isinstance(tree, list)
        # if is_single_tree:
        #     tree = [tree]

        (
            padded_feature,
            padding_mask,
            padded_index,
            leaf_mask,
            leaf_idx,
            edge_mask,
            edge_split_masks,
        ) = tokenized_tree_batch

        # for i in edge_split_masks[0]:
        #     print(
        #                 [j for j in range(int(i).bit_length()) if (int(i) >> j) & 1],
        #             )

        # import pdb; pdb.set_trace()

        x = padded_feature
        B, T_raw, D = x.shape

        # Add phyla embedding to leaf nodes only
        if phyla_embeddings is not None:
            # Handle list of embeddings with different shapes
            if isinstance(phyla_embeddings, list):
                # Find max length across all embeddings in the list
                max_len = max(emb.shape[0] for emb in phyla_embeddings)

                padded_embeddings = []

                for emb in phyla_embeddings:
                    # Pad each embedding to max_len
                    if emb.shape[0] < max_len:
                        padding = torch.zeros(
                            max_len - emb.shape[0],
                            emb.shape[1],
                            device=emb.device,
                            dtype=emb.dtype,
                        )
                        padded_emb = torch.cat([emb, padding], dim=0)
                    else:
                        padded_emb = emb
                    padded_embeddings.append(padded_emb)

                # Stack into tensor (B, max_len, embed_dim)
                phyla_embeddings = torch.stack(padded_embeddings, dim=0)
            # Expected shapes:
            #   phyla_embedding: (B, N_leaves_max, phyla_dim) OR (1, N_leaves, phyla_dim)
            elif phyla_embeddings.dim() == 2:  # (N,D) -> treat as (1,N,D) and broadcast
                phyla_embeddings = phyla_embeddings.unsqueeze(0)

            if phyla_embeddings.size(0) == 1 and B > 1:
                # Broadcast same embedding set across batch
                phyla_embeddings = phyla_embeddings.expand(B, -1, -1)
                if phyla_embeddings.size(0) != B:
                    raise ValueError(
                        f"phyla_embeddings batch mismatch: got {phyla_embeddings.size(0)} expected {B}"
                    )
            # Project all embeddings once: (B, N, D_model)
            phyla_proj_full = self.phyla_proj(phyla_embeddings)  # (B,N,D)

            # We'll store phyla_proj_full and leaf_idx_list for use post-concat
            phyla_info = (phyla_proj_full, leaf_idx)
        else:
            phyla_info = None

        # Prepend [graph] token
        graph_token = self.graph_token.expand(B, 1, D)
        x = torch.cat([graph_token, x], dim=1)  # [B, T_raw+1, D]

        # If we have phyla embeddings, add them now at adjusted indices
        if phyla_info is not None:
            phyla_proj_full, leaf_idx_list = phyla_info
            for b in range(B):
                if len(leaf_idx_list[b]) == 0:
                    continue
                leaf_indices = leaf_idx_list[b]
                if leaf_indices.numel() == 0:
                    continue
                L_b = leaf_indices.numel()
                max_available = phyla_proj_full.size(1)
                adjusted = leaf_indices + 1  # +1 for [graph] token

                # Ensure we have at least L_b phyla embeddings
                if phyla_proj_full.size(1) < L_b:
                    import pdb

                    pdb.set_trace()
                    raise ValueError(
                        f"Need {L_b} phyla embeddings, got {phyla_proj_full.size(1)}"
                    )

                # Scatter-add the first L_b phyla embeddings onto those leaf token positions
                # x[b]: [T_raw+1, D]; adjusted: [L_b]; phyla_proj_full[b, :L_b]: [L_b, D]
                x[b, adjusted] += phyla_proj_full[b, :L_b]

        if padding_mask is not None:
            special_tokens_mask = torch.zeros(
                B, 1, dtype=padding_mask.dtype, device=padding_mask.device
            )
            padding_mask = torch.cat(
                [special_tokens_mask, padding_mask],
                dim=1,
            )  # [B, T_raw+1]

            # Update leaf_mask for batch format
            if leaf_mask.dim() == 2:  # Already batched
                leaf_mask_special = torch.zeros(
                    B, 1, dtype=leaf_mask.dtype, device=leaf_mask.device
                )
                leaf_mask = torch.cat([leaf_mask_special, leaf_mask], dim=1)
            else:  # Single tree case, need to expand
                leaf_mask_expanded = torch.zeros(
                    B, T_raw, dtype=leaf_mask.dtype, device=leaf_mask.device
                )
                leaf_mask_expanded[0, : leaf_mask.size(0)] = leaf_mask
                leaf_mask_special = torch.zeros(
                    B, 1, dtype=leaf_mask.dtype, device=leaf_mask.device
                )
                leaf_mask = torch.cat([leaf_mask_special, leaf_mask_expanded], dim=1)

        # Timestep conditioning via addition
        if t is not None:
            time_sin_emb = self.create_sinusoidal_embedding(t, self.embed_dim)  # [B, D]
            time_emb = self.time_embed(time_sin_emb)  # [B, D]
            x = x + time_emb.unsqueeze(1)
        x = self.dropout(x)

        # Transformer encoder
        for layer in self.layers:
            x, _ = layer(x, padding_mask=padding_mask)
        x = self.final_layer_norm(x)

        if return_leafs_only:
            # Handle batch of leaf indices
            if is_single_tree and B == 1:
                # Return single tree format for backward compatibility
                if len(leaf_idx_list[0]) > 0:
                    adjusted_indices = leaf_idx_list[0] + 1
                    valid_mask = adjusted_indices < x.size(1)
                    valid_indices = adjusted_indices[valid_mask]
                    if valid_indices.numel() > 0:
                        return x[0, valid_indices].unsqueeze(0)
                    else:
                        return torch.zeros(1, 0, D, device=x.device)
                else:
                    return torch.zeros(1, 0, D, device=x.device)
            else:
                # Batch format
                batch_leaf_outputs = []
                for b in range(B):
                    if len(leaf_idx_list[b]) > 0:
                        # Add 1 to account for [graph] token
                        adjusted_indices = leaf_idx_list[b] + 1
                        valid_mask = adjusted_indices < x.size(1)
                        valid_indices = adjusted_indices[valid_mask]
                        if valid_indices.numel() > 0:
                            batch_leaf_outputs.append(x[b, valid_indices])
                        else:
                            batch_leaf_outputs.append(
                                torch.zeros(0, D, device=x.device)
                            )
                    else:
                        batch_leaf_outputs.append(torch.zeros(0, D, device=x.device))

                # Pad to same length for batch processing
                if batch_leaf_outputs:
                    max_leaf_len = max(out.size(0) for out in batch_leaf_outputs)
                    if max_leaf_len > 0:
                        padded_leaf_outputs = torch.zeros(
                            B, max_leaf_len, D, device=x.device
                        )
                        for b, out in enumerate(batch_leaf_outputs):
                            if out.size(0) > 0:
                                padded_leaf_outputs[b, : out.size(0)] = out
                        return padded_leaf_outputs
                    else:
                        return torch.zeros(B, 0, D, device=x.device)
                else:
                    return torch.zeros(B, 0, D, device=x.device)
        elif autoregressive:
            # Remove graph token; edge_mask assumed shape [B, T_raw]
            x_no_graph = x[:, 1:, :]

            all_group_logits = []

            # Derive per-sample split universe from actual edge masks.
            # This keeps polytomy grouping in the exact same bit-space used by tokenizer splits.
            num_leaves = []
            for splits_b in edge_split_masks:
                max_bit = 0
                for s in splits_b:
                    s_int = int(s)
                    if s_int != 0:
                        max_bit = max(max_bit, s_int.bit_length())
                num_leaves.append(max_bit)

            batch_polytomy_index, batch_polytomy_splits = get_batch_polytomy_indices(
                edge_split_masks,
                edge_mask,
                min_children=3,
                include_root=True,
                num_leaves=num_leaves,
            )
                                                                                    
            for b, groups in enumerate(batch_polytomy_index):
                for num, group in enumerate(groups):
                    if group.size(0) <= 1:
                        continue
                    # Get the embeddings for this group
                    group_embeddings = x_no_graph[b, group, :]  # [G, D]
                    logits = self.pairwise_head(group_embeddings)  # [G, G]
                    splits_represented = batch_polytomy_splits[b][num]  # [G]

                    # for split in splits_represented:
                    #     print(
                    #     [j for j in range(int(split).bit_length()) if (int(split) >> j) & 1],
                    # )

                    all_group_logits.append({
                        "batch_index": b,
                        "group_indices": group,
                        "polytomy_pred": self.group_head(group_embeddings.mean(dim=0)),
                        "logits": logits,
                        "splits_represented": splits_represented,
                        "group_embeddings": group_embeddings,
                    })
            # if not all_group_logits:
            #     raise ValueError("No polytomies found for autoregressive processing.")
            # import pdb; pdb.set_trace()
            return all_group_logits

        elif return_edges_only:
            # Remove graph token; edge_mask assumed shape [B, T_raw]
            x_no_graph = x[:, 1:, :]

            # Collect per-batch edge embeddings using the provided mask
            edge_mask_bool = edge_mask.bool()
            edge_lists = [x_no_graph[b][edge_mask_bool[b]] for b in range(B)]
            max_edges = max((e.size(0) for e in edge_lists), default=0)

            # Pad to max_edges so outputs can be batched
            if max_edges == 0:
                padded_edges = torch.zeros(B, 0, D, device=x.device, dtype=x.dtype)
                edge_pad_mask = torch.ones(B, 0, device=x.device, dtype=torch.bool)
            else:
                padded_edges = torch.zeros(
                    B, max_edges, D, device=x.device, dtype=x.dtype
                )
                edge_pad_mask = torch.ones(
                    B, max_edges, device=x.device, dtype=torch.bool
                )
                for b, edges_b in enumerate(edge_lists):
                    n_b = edges_b.size(0)
                    if n_b == 0:
                        continue
                    padded_edges[b, :n_b] = edges_b
                    edge_pad_mask[b, :n_b] = False  # False = real, True = pad

            # Optional: pass through output layer before returning
            edge_outputs = self.output_layer(padded_edges)  # [B, max_edges, output_dim]

            return (
                edge_outputs,
                edge_pad_mask,
            )  # return mask so caller can ignore padding
        elif return_all_tokens:
            return x  # [B, T, D]
        else:
            out = self.output_layer(x[:, 0])
            return out


def return_model(config):
    model = TreeDenoiserTokenGT(
        num_node_types=config["model"]["num_node_types"],
        num_edge_types=config["model"]["num_edge_types"],
        embed_dim=config["model"]["embed_dim"],
        output_dim=config["model"]["output_dim"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        dropout=config["model"]["dropout"],
        attention_dropout=config["model"]["attention_dropout"],
        activation_dropout=config["model"]["activation_dropout"],
        drop_path_rate=config["model"]["drop_path_rate"],
        use_performer=config["model"]["use_performer"],
        performer_nb_features=config["model"]["performer_nb_features"],
        performer_generalized_attention=config["model"][
            "performer_generalized_attention"
        ],
        layernorm_style=config["model"]["layernorm_style"],
        tokenizer_lap_dim=config["model"]["tokenizer_lap_dim"],
        tokenizer_lap_dropout=config["model"]["tokenizer_lap_dropout"],
        tokenizer_n_layers=config["model"]["tokenizer_n_layers"],
        phyla_dim=config["model"]["phyla_dim"],
    )

    return model
