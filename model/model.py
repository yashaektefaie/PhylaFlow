import math
from collections import OrderedDict
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
from utils.bhv_utils import (
    get_batch_explicit_structural_group_indices,
    get_batch_structural_polytomy_indices,
)


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
        self.norm = nn.LayerNorm(d_model)

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
        H = self.norm(H)
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


class StructuredSubsetMergeHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden: int = 256,
        dropout: float = 0.1,
        max_subset_size: int = 64,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.max_subset_size = int(max_subset_size)
        pair_in_dim = 4 * d_model
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pair_logit = nn.Linear(hidden, 1)
        self.pair_context_proj = nn.Sequential(
            nn.Linear(hidden + d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        member_in_dim = 4 * d_model
        self.member_mlp = nn.Sequential(
            nn.Linear(member_in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.subset_size_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.max_subset_size + 1),
        )
        self.stop_after_merge_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, H: torch.Tensor) -> dict:
        """
        H: [G, D]
        Returns:
          starter_pair_logits: [P]
          starter_pair_indices: list[(i, j)] for unordered pairs
          member_logits: [P, G] logits for including each component in the subset
          logits: [G, G] symmetric starter-pair score matrix for compatibility
        """
        H = self.norm(H)
        G, D = H.shape
        if G <= 1:
            empty_logits = H.new_empty((0,))
            return {
                "starter_pair_logits": empty_logits,
                "starter_pair_indices": [],
                "member_logits": H.new_empty((0, G)),
                "logits": H.new_full((G, G), float("-inf")),
                "subset_size_logits": H.new_full(
                    (self.max_subset_size + 1,),
                    float("-inf"),
                ),
                "stop_after_merge_logit": H.new_zeros(()),
            }

        pair_indices = torch.triu_indices(G, G, offset=1, device=H.device)
        left_idx, right_idx = pair_indices[0], pair_indices[1]
        hi = H[left_idx]
        hj = H[right_idx]
        pair_feats = torch.cat([hi, hj, (hi - hj).abs(), hi * hj], dim=-1)
        pair_hidden = self.pair_mlp(pair_feats)
        starter_pair_logits = self.pair_logit(pair_hidden).squeeze(-1)

        pair_summary = 0.5 * (hi + hj)
        pair_context = self.pair_context_proj(
            torch.cat([pair_hidden, pair_summary], dim=-1)
        )

        node_expand = H.unsqueeze(0).expand(pair_context.size(0), G, D)
        pair_expand = pair_context.unsqueeze(1).expand(pair_context.size(0), G, D)
        member_feats = torch.cat(
            [
                node_expand,
                pair_expand,
                (node_expand - pair_expand).abs(),
                node_expand * pair_expand,
            ],
            dim=-1,
        )
        member_logits = self.member_mlp(member_feats).squeeze(-1)

        pair_matrix = H.new_full((G, G), float("-inf"))
        pair_matrix[left_idx, right_idx] = starter_pair_logits
        pair_matrix[right_idx, left_idx] = starter_pair_logits

        return {
            "starter_pair_logits": starter_pair_logits,
            "starter_pair_indices": [
                (int(i.item()), int(j.item()))
                for i, j in zip(left_idx, right_idx)
            ],
            "member_logits": member_logits,
            "logits": pair_matrix,
            "subset_size_logits": self.subset_size_head(H.mean(dim=0)),
            "stop_after_merge_logit": self.stop_after_merge_head(H.mean(dim=0)).squeeze(-1),
        }


class AutoregressiveGroupRefinementBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        n_layers: int = 12,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            n_layers=n_layers,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.apply(lambda m: init_params(m, n_layers))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        if H.size(0) <= 1:
            return H

        attn_input = self.attn_norm(H).unsqueeze(0)
        attn_output, _ = self.attn(attn_input)
        H = H + self.attn_dropout(attn_output.squeeze(0))
        H = H + self.ffn(self.ffn_norm(H))
        return H


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
        tokenizer_branch_length_mode="linear",
        tokenizer_branch_length_num_buckets=64,
        tokenizer_branch_length_log_min=-8.0,
        tokenizer_branch_length_log_max=1.0,
        phyla_dim=256,
        phyla_use_leaf_tokens=True,
        phyla_use_split_tokens=True,
        phyla_leaf_scale=1.0,
        phyla_split_scale=1.0,
        autoregressive_head_mode="pairwise_threshold",
        autoregressive_group_refinement_layers=0,
        autoregressive_max_subset_size=64,
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
            branch_length_mode=tokenizer_branch_length_mode,
            branch_length_num_buckets=tokenizer_branch_length_num_buckets,
            branch_length_log_min=tokenizer_branch_length_log_min,
            branch_length_log_max=tokenizer_branch_length_log_max,
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
        self.phyla_split_proj = nn.Sequential(
            nn.Linear(2 * phyla_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.phyla_use_leaf_tokens = bool(phyla_use_leaf_tokens)
        self.phyla_use_split_tokens = bool(phyla_use_split_tokens)
        self.phyla_leaf_scale = float(phyla_leaf_scale)
        self.phyla_split_scale = float(phyla_split_scale)

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

        self.edge_output_layer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim), # Final layer is LINEAR
        )
        self.first_hit_edge_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.boundary_vanish_edge_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        #This used to be 0.2
        self.edge_output_scale = 2
        self.max_split_bits = 256
        self.split_mask_proj = nn.Sequential(
            nn.Linear(self.max_split_bits, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # self.split_output_layer = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.GELU(),
        #     nn.Linear(embed_dim, output_dim),
        # )
        self.split_identity_scale = 0.75
        self.split_output_scale = 0.5
        self._split_identity_binary_cache = OrderedDict()
        self._split_identity_binary_cache_max = 4096
        self.dropout = nn.Dropout(dropout)
        self.autoregressive_head_mode = str(autoregressive_head_mode)
        self.autoregressive_group_refinement_layers = int(
            autoregressive_group_refinement_layers
        )
        self.autoregressive_max_subset_size = int(autoregressive_max_subset_size)
        self.autoregressive_group_refinement = nn.ModuleList(
            [
                AutoregressiveGroupRefinementBlock(
                    d_model=embed_dim,
                    num_heads=n_heads,
                    dropout=dropout,
                    ffn_mult=4,
                    n_layers=n_layers,
                )
                for _ in range(self.autoregressive_group_refinement_layers)
            ]
        )
        if self.autoregressive_head_mode == "pairwise_threshold":
            self.pairwise_head = PairwiseMergeHead(
                d_model=embed_dim, hidden=embed_dim, dropout=dropout
            )
            self.structured_subset_head = None
        elif self.autoregressive_head_mode == "structured_subset":
            self.pairwise_head = None
            self.structured_subset_head = StructuredSubsetMergeHead(
                d_model=embed_dim,
                hidden=embed_dim,
                dropout=dropout,
                max_subset_size=self.autoregressive_max_subset_size,
            )
        else:
            raise ValueError(
                f"Unknown autoregressive_head_mode={self.autoregressive_head_mode!r}"
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

    def create_split_identity_embedding(self, split_masks, device):
        if not split_masks:
            return torch.zeros(0, self.embed_dim, device=device)

        split_masks_tuple = tuple(int(mask) for mask in split_masks)
        cache_key = (
            split_masks_tuple,
            str(device),
            str(self.graph_token.dtype),
        )
        binary_masks = self._split_identity_binary_cache.get(cache_key)
        cache_writable = not torch.is_inference_mode_enabled()
        if binary_masks is None or (
            not torch.is_inference_mode_enabled() and binary_masks.is_inference()
        ):
            binary_masks = torch.zeros(
                len(split_masks_tuple),
                self.max_split_bits,
                device=device,
                dtype=self.graph_token.dtype,
            )
            for row_idx, mask_int in enumerate(split_masks_tuple):
                bit_idx = 0
                while mask_int and bit_idx < self.max_split_bits:
                    if mask_int & 1:
                        binary_masks[row_idx, bit_idx] = 1.0
                    mask_int >>= 1
                    bit_idx += 1
            if cache_writable:
                self._split_identity_binary_cache[cache_key] = binary_masks
                if (
                    len(self._split_identity_binary_cache)
                    > self._split_identity_binary_cache_max
                ):
                    self._split_identity_binary_cache.popitem(last=False)
        else:
            self._split_identity_binary_cache.move_to_end(cache_key)

        return self.split_mask_proj(binary_masks)

    def _normalize_phyla_embeddings(self, phyla_embeddings, batch_size):
        if phyla_embeddings is None:
            return None
        if isinstance(phyla_embeddings, list):
            max_len = max(emb.shape[0] for emb in phyla_embeddings)
            padded_embeddings = []
            for emb in phyla_embeddings:
                if emb.shape[0] < max_len:
                    padding = torch.zeros(
                        max_len - emb.shape[0],
                        emb.shape[1],
                        device=emb.device,
                        dtype=emb.dtype,
                    )
                    emb = torch.cat([emb, padding], dim=0)
                padded_embeddings.append(emb)
            phyla_embeddings = torch.stack(padded_embeddings, dim=0)
        elif phyla_embeddings.dim() == 2:
            phyla_embeddings = phyla_embeddings.unsqueeze(0)

        if phyla_embeddings.size(0) == 1 and batch_size > 1:
            phyla_embeddings = phyla_embeddings.expand(batch_size, -1, -1)
        if phyla_embeddings.size(0) != batch_size:
            raise ValueError(
                f"phyla_embeddings batch mismatch: got {phyla_embeddings.size(0)} expected {batch_size}"
            )
        return phyla_embeddings

    def _compute_leaf_phyla_token_additions(
        self,
        phyla_proj_full,
        leaf_idx_list,
        num_tokens,
        device,
        dtype,
    ):
        additions = torch.zeros(
            phyla_proj_full.size(0),
            num_tokens,
            self.embed_dim,
            device=device,
            dtype=dtype,
        )
        for b, leaf_indices in enumerate(leaf_idx_list):
            if leaf_indices.numel() == 0:
                continue
            leaf_count = leaf_indices.numel()
            if phyla_proj_full.size(1) < leaf_count:
                raise ValueError(
                    f"Need {leaf_count} phyla embeddings, got {phyla_proj_full.size(1)}"
                )
            additions[b, leaf_indices] = phyla_proj_full[b, :leaf_count].to(dtype)
        return additions

    def _compute_split_phyla_token_additions(
        self,
        phyla_embeddings,
        leaf_idx_list,
        edge_mask,
        edge_split_masks,
        num_tokens,
        device,
        dtype,
    ):
        additions = torch.zeros(
            phyla_embeddings.size(0),
            num_tokens,
            self.embed_dim,
            device=device,
            dtype=dtype,
        )
        zero_raw = torch.zeros(
            phyla_embeddings.size(-1), device=device, dtype=phyla_embeddings.dtype
        )

        for b, leaf_indices in enumerate(leaf_idx_list):
            if leaf_indices.numel() == 0:
                continue

            leaf_count = leaf_indices.numel()
            if phyla_embeddings.size(1) < leaf_count:
                raise ValueError(
                    f"Need {leaf_count} phyla embeddings, got {phyla_embeddings.size(1)}"
                )

            raw_leaf_embeddings = phyla_embeddings[b, :leaf_count]
            leaf_bits = [int(idx.item()) for idx in leaf_indices]
            edge_positions = torch.nonzero(edge_mask[b], as_tuple=True)[0]
            split_masks_b = edge_split_masks[b]
            edge_count = min(edge_positions.numel(), len(split_masks_b))

            for edge_idx in range(edge_count):
                split_mask = int(split_masks_b[edge_idx])
                if split_mask == 0:
                    continue
                select = torch.tensor(
                    [bool((split_mask >> bit) & 1) for bit in leaf_bits],
                    device=device,
                    dtype=torch.bool,
                )
                inside = (
                    raw_leaf_embeddings[select].mean(dim=0)
                    if select.any()
                    else zero_raw
                )
                outside = (
                    raw_leaf_embeddings[~select].mean(dim=0)
                    if (~select).any()
                    else zero_raw
                )
                split_feature = torch.cat([inside, outside], dim=0)
                additions[b, edge_positions[edge_idx]] = self.phyla_split_proj(
                    split_feature
                ).to(dtype)
        return additions

    def _prepare_encoder_inputs(
        self,
        tokenized_tree_batch,
        t=None,
        phyla_embeddings=None,
    ):
        (
            padded_feature,
            padding_mask,
            padded_index,
            leaf_mask,
            leaf_idx,
            edge_mask,
            edge_split_masks,
        ) = tokenized_tree_batch

        x = padded_feature
        B, T_raw, D = x.shape

        phyla_embeddings = self._normalize_phyla_embeddings(phyla_embeddings, B)
        if phyla_embeddings is not None:
            if self.phyla_use_leaf_tokens:
                phyla_proj_full = self.phyla_proj(phyla_embeddings)
                x = x + (
                    self.phyla_leaf_scale
                    * self._compute_leaf_phyla_token_additions(
                        phyla_proj_full,
                        leaf_idx,
                        T_raw,
                        x.device,
                        x.dtype,
                    )
                )
            if self.phyla_use_split_tokens:
                x = x + (
                    self.phyla_split_scale
                    * self._compute_split_phyla_token_additions(
                        phyla_embeddings,
                        leaf_idx,
                        edge_mask.bool(),
                        edge_split_masks,
                        T_raw,
                        x.device,
                        x.dtype,
                    )
                )

        graph_token = self.graph_token.to(device=x.device, dtype=x.dtype).expand(B, 1, D)
        x = torch.cat([graph_token, x], dim=1)

        if padding_mask is not None:
            special_tokens_mask = torch.zeros(
                B, 1, dtype=padding_mask.dtype, device=padding_mask.device
            )
            padding_mask = torch.cat(
                [special_tokens_mask, padding_mask],
                dim=1,
            )

            if leaf_mask.dim() == 2:
                leaf_mask_special = torch.zeros(
                    B, 1, dtype=leaf_mask.dtype, device=leaf_mask.device
                )
                leaf_mask = torch.cat([leaf_mask_special, leaf_mask], dim=1)
            else:
                leaf_mask_expanded = torch.zeros(
                    B, T_raw, dtype=leaf_mask.dtype, device=leaf_mask.device
                )
                leaf_mask_expanded[0, : leaf_mask.size(0)] = leaf_mask
                leaf_mask_special = torch.zeros(
                    B, 1, dtype=leaf_mask.dtype, device=leaf_mask.device
                )
                leaf_mask = torch.cat([leaf_mask_special, leaf_mask_expanded], dim=1)

        if t is not None:
            time_sin_emb = self.create_sinusoidal_embedding(t, self.embed_dim)
            time_emb = self.time_embed(time_sin_emb)
            x = x + time_emb.unsqueeze(1)
        x = self.dropout(x)

        return x, padding_mask, leaf_mask, leaf_idx, edge_mask, edge_split_masks

    def _encode_with_layers(self, x, padding_mask, layers, final_layer_norm):
        for layer in layers:
            x, _ = layer(x, padding_mask=padding_mask)
        return final_layer_norm(x)

    def _decode_outputs(
        self,
        x,
        leaf_mask,
        leaf_idx,
        edge_mask,
        edge_split_masks,
        return_all_tokens=True,
        return_leafs_only=False,
        return_edges_only=False,
        return_edge_features=False,
        return_first_hit_logits=False,
        return_boundary_vanish_logits=False,
        autoregressive=False,
        autoregressive_component_groups=None,
    ):
        B, _T, D = x.shape
        leaf_idx_list = list(leaf_idx) if isinstance(leaf_idx, (list, tuple)) else [leaf_idx]
        is_single_tree = B == 1 and not isinstance(leaf_idx, (list, tuple))

        if return_leafs_only:
            if is_single_tree and B == 1:
                if len(leaf_idx_list[0]) > 0:
                    adjusted_indices = leaf_idx_list[0] + 1
                    valid_mask = adjusted_indices < x.size(1)
                    valid_indices = adjusted_indices[valid_mask]
                    if valid_indices.numel() > 0:
                        return x[0, valid_indices].unsqueeze(0)
                    return torch.zeros(1, 0, D, device=x.device)
                return torch.zeros(1, 0, D, device=x.device)

            batch_leaf_outputs = []
            for b in range(B):
                if len(leaf_idx_list[b]) > 0:
                    adjusted_indices = leaf_idx_list[b] + 1
                    valid_mask = adjusted_indices < x.size(1)
                    valid_indices = adjusted_indices[valid_mask]
                    if valid_indices.numel() > 0:
                        batch_leaf_outputs.append(x[b, valid_indices])
                    else:
                        batch_leaf_outputs.append(torch.zeros(0, D, device=x.device))
                else:
                    batch_leaf_outputs.append(torch.zeros(0, D, device=x.device))

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
            return torch.zeros(B, 0, D, device=x.device)

        if autoregressive:
            x_no_graph = x[:, 1:, :]
            all_group_logits = []
            num_leaves = [int(leaf_mask[b].sum().item()) + 1 for b in range(B)]

            if autoregressive_component_groups is None:
                batch_polytomy_index, batch_polytomy_splits = (
                    get_batch_structural_polytomy_indices(
                        edge_split_masks,
                        edge_mask,
                        min_children=3,
                        num_leaves=num_leaves,
                    )
                )
            else:
                batch_polytomy_index, batch_polytomy_splits = (
                    get_batch_explicit_structural_group_indices(
                        edge_split_masks,
                        edge_mask,
                        autoregressive_component_groups,
                        num_leaves=num_leaves,
                    )
                )

            for b, groups in enumerate(batch_polytomy_index):
                for num, group in enumerate(groups):
                    if group.size(0) <= 1:
                        continue
                    group_embeddings = x_no_graph[b, group, :]
                    group_splits = batch_polytomy_splits[b][num]
                    group_identity = self.create_split_identity_embedding(
                        group_splits, x.device
                    )

                    group_embeddings = group_embeddings + (
                        self.split_identity_scale * group_identity
                    )
                    for refinement_block in self.autoregressive_group_refinement:
                        group_embeddings = refinement_block(group_embeddings)
                    if self.autoregressive_head_mode == "structured_subset":
                        head_outputs = self.structured_subset_head(group_embeddings)
                        logits = head_outputs["logits"]
                    else:
                        head_outputs = {}
                        logits = self.pairwise_head(group_embeddings)

                    group_output = {
                        "batch_index": b,
                        "group_indices": group,
                        "polytomy_pred": self.group_head(
                            self.final_layer_norm(group_embeddings).mean(dim=0)
                        ),
                        "logits": logits,
                        "splits_represented": group_splits,
                        "group_embeddings": group_embeddings,
                        "decoder_mode": self.autoregressive_head_mode,
                    }
                    group_output.update(head_outputs)
                    all_group_logits.append(group_output)
            return all_group_logits

        if return_edges_only:
            x_no_graph = x[:, 1:, :]
            edge_mask_bool = edge_mask.bool()
            edge_lists = [x_no_graph[b][edge_mask_bool[b]] for b in range(B)]
            max_edges = max((e.size(0) for e in edge_lists), default=0)

            if max_edges == 0:
                return torch.zeros(B, 0, D, device=x.device), torch.ones(
                    B, 0, device=x.device, dtype=torch.bool
                )

            padded_edges = torch.zeros(B, max_edges, D, device=x.device, dtype=x.dtype)
            edge_pad_mask = torch.ones(B, max_edges, device=x.device, dtype=torch.bool)

            for b, edges_b in enumerate(edge_lists):
                n_b = edges_b.size(0)
                if n_b == 0:
                    continue
                split_identity = self.create_split_identity_embedding(
                    edge_split_masks[b][:n_b], x.device
                )
                padded_edges[b, :n_b] = edges_b + (
                    self.split_identity_scale * split_identity
                )
                edge_pad_mask[b, :n_b] = False

            edge_outputs = self.edge_output_layer(padded_edges)
            if return_first_hit_logits or return_boundary_vanish_logits:
                first_hit_logits = None
                boundary_vanish_logits = None
                if return_first_hit_logits:
                    first_hit_logits = self.first_hit_edge_head(padded_edges)
                if return_boundary_vanish_logits:
                    boundary_vanish_logits = self.boundary_vanish_edge_head(
                        padded_edges
                    )
                if return_first_hit_logits and return_boundary_vanish_logits:
                    if return_edge_features:
                        return (
                            edge_outputs,
                            edge_pad_mask,
                            padded_edges,
                            first_hit_logits,
                            boundary_vanish_logits,
                        )
                    return (
                        edge_outputs,
                        edge_pad_mask,
                        first_hit_logits,
                        boundary_vanish_logits,
                    )
                if return_first_hit_logits:
                    if return_edge_features:
                        return edge_outputs, edge_pad_mask, padded_edges, first_hit_logits
                    return edge_outputs, edge_pad_mask, first_hit_logits
                if return_edge_features:
                    return edge_outputs, edge_pad_mask, padded_edges, boundary_vanish_logits
                return edge_outputs, edge_pad_mask, boundary_vanish_logits

            if return_edge_features:
                return edge_outputs, edge_pad_mask, padded_edges
            return edge_outputs, edge_pad_mask

        if return_all_tokens:
            return x
        return self.output_layer(x[:, 0])

    def forward(
        self,
        tokenized_tree_batch,
        t=None,
        phyla_embeddings=None,
        return_all_tokens=True,
        return_leafs_only=False,
        return_edges_only=False,
        return_edge_features=False,
        return_first_hit_logits=False,
        return_boundary_vanish_logits=False,
        autoregressive=False,
        autoregressive_component_groups=None,
    ):
        x, padding_mask, leaf_mask, leaf_idx, edge_mask, edge_split_masks = (
            self._prepare_encoder_inputs(
                tokenized_tree_batch,
                t=t,
                phyla_embeddings=phyla_embeddings,
            )
        )
        x = self._encode_with_layers(
            x,
            padding_mask=padding_mask,
            layers=self.layers,
            final_layer_norm=self.final_layer_norm,
        )
        return self._decode_outputs(
            x,
            leaf_mask=leaf_mask,
            leaf_idx=leaf_idx,
            edge_mask=edge_mask,
            edge_split_masks=edge_split_masks,
            return_all_tokens=return_all_tokens,
            return_leafs_only=return_leafs_only,
            return_edges_only=return_edges_only,
            return_edge_features=return_edge_features,
            return_first_hit_logits=return_first_hit_logits,
            return_boundary_vanish_logits=return_boundary_vanish_logits,
            autoregressive=autoregressive,
            autoregressive_component_groups=autoregressive_component_groups,
        )


class TreeDenoiserTokenGTTwoBlock(TreeDenoiserTokenGT):
    def __init__(
        self,
        *args,
        block2_n_layers=None,
        block2_input_mode="direct",
        block2_drop_path_rate=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.block2_n_layers = int(
            self.n_layers if block2_n_layers is None else block2_n_layers
        )
        self.block2_input_mode = str(block2_input_mode)
        block2_drop_path_rate = (
            self.block2_n_layers > 1 and kwargs.get("drop_path_rate", 0.0)
            if block2_drop_path_rate is None
            else float(block2_drop_path_rate)
        )
        dprates = [
            block2_drop_path_rate * i / (self.block2_n_layers - 1)
            if self.block2_n_layers > 1
            else 0.0
            for i in range(self.block2_n_layers)
        ]
        self.block2_layers = nn.ModuleList(
            [
                TreeGraphEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_dim=self.embed_dim * 4,
                    n_heads=kwargs["n_heads"],
                    dropout=kwargs["dropout"],
                    attention_dropout=kwargs["attention_dropout"],
                    activation_dropout=kwargs["activation_dropout"],
                    drop_path=dprates[i],
                    use_performer=kwargs["use_performer"],
                    performer_nb_features=kwargs["performer_nb_features"],
                    performer_generalized_attention=kwargs[
                        "performer_generalized_attention"
                    ],
                    layernorm_style=kwargs["layernorm_style"],
                    n_layers=self.block2_n_layers,
                )
                for i in range(self.block2_n_layers)
            ]
        )
        self.block2_final_layer_norm = nn.LayerNorm(self.embed_dim)
        if self.block2_input_mode == "direct":
            self.block2_bridge = nn.Identity()
        elif self.block2_input_mode == "residual_mlp":
            self.block2_bridge = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Dropout(kwargs["dropout"]),
            )
        elif self.block2_input_mode == "concat_raw_mlp":
            self.block2_bridge = nn.Sequential(
                nn.LayerNorm(2 * self.embed_dim),
                nn.Linear(2 * self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Dropout(kwargs["dropout"]),
            )
        else:
            raise ValueError(
                f"Unknown block2_input_mode={self.block2_input_mode!r}"
            )
        self.apply(lambda m: init_params(m, self.n_layers + self.block2_n_layers))

    def _build_block2_input(self, stage1_x, encoder_input_x):
        if self.block2_input_mode == "direct":
            return stage1_x
        if self.block2_input_mode == "residual_mlp":
            return stage1_x + self.block2_bridge(stage1_x)
        return self.block2_bridge(torch.cat([stage1_x, encoder_input_x], dim=-1))

    def forward(
        self,
        tokenized_tree_batch,
        t=None,
        phyla_embeddings=None,
        return_all_tokens=True,
        return_leafs_only=False,
        return_edges_only=False,
        return_edge_features=False,
        return_first_hit_logits=False,
        return_boundary_vanish_logits=False,
        autoregressive=False,
        autoregressive_component_groups=None,
    ):
        x, padding_mask, leaf_mask, leaf_idx, edge_mask, edge_split_masks = (
            self._prepare_encoder_inputs(
                tokenized_tree_batch,
                t=t,
                phyla_embeddings=phyla_embeddings,
            )
        )
        stage1_x = self._encode_with_layers(
            x,
            padding_mask=padding_mask,
            layers=self.layers,
            final_layer_norm=self.final_layer_norm,
        )
        block2_x = self._build_block2_input(stage1_x, x)
        x = self._encode_with_layers(
            block2_x,
            padding_mask=padding_mask,
            layers=self.block2_layers,
            final_layer_norm=self.block2_final_layer_norm,
        )
        return self._decode_outputs(
            x,
            leaf_mask=leaf_mask,
            leaf_idx=leaf_idx,
            edge_mask=edge_mask,
            edge_split_masks=edge_split_masks,
            return_all_tokens=return_all_tokens,
            return_leafs_only=return_leafs_only,
            return_edges_only=return_edges_only,
            return_edge_features=return_edge_features,
            return_first_hit_logits=return_first_hit_logits,
            return_boundary_vanish_logits=return_boundary_vanish_logits,
            autoregressive=autoregressive,
            autoregressive_component_groups=autoregressive_component_groups,
        )


class TreeDenoiserTokenGTMultiBlock(TreeDenoiserTokenGT):
    def __init__(
        self,
        *args,
        num_model_blocks=3,
        refine_block_n_layers=None,
        refine_block_input_mode="direct",
        refine_block_drop_path_rate=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_model_blocks = int(num_model_blocks)
        if self.num_model_blocks < 2:
            raise ValueError("num_model_blocks must be at least 2.")
        self.refine_block_n_layers = int(
            self.n_layers if refine_block_n_layers is None else refine_block_n_layers
        )
        self.refine_block_input_mode = str(refine_block_input_mode)
        refine_block_drop_path_rate = (
            kwargs.get("drop_path_rate", 0.0)
            if refine_block_drop_path_rate is None
            else float(refine_block_drop_path_rate)
        )
        self.refine_blocks = nn.ModuleList()
        for _ in range(self.num_model_blocks - 1):
            dprates = [
                refine_block_drop_path_rate * i / (self.refine_block_n_layers - 1)
                if self.refine_block_n_layers > 1
                else 0.0
                for i in range(self.refine_block_n_layers)
            ]
            layers = nn.ModuleList(
                [
                    TreeGraphEncoderLayer(
                        embed_dim=self.embed_dim,
                        ffn_dim=self.embed_dim * 4,
                        n_heads=kwargs["n_heads"],
                        dropout=kwargs["dropout"],
                        attention_dropout=kwargs["attention_dropout"],
                        activation_dropout=kwargs["activation_dropout"],
                        drop_path=dprates[i],
                        use_performer=kwargs["use_performer"],
                        performer_nb_features=kwargs["performer_nb_features"],
                        performer_generalized_attention=kwargs[
                            "performer_generalized_attention"
                        ],
                        layernorm_style=kwargs["layernorm_style"],
                        n_layers=self.refine_block_n_layers,
                    )
                    for i in range(self.refine_block_n_layers)
                ]
            )
            final_norm = nn.LayerNorm(self.embed_dim)
            if self.refine_block_input_mode == "direct":
                bridge = nn.Identity()
            elif self.refine_block_input_mode == "residual_mlp":
                bridge = nn.Sequential(
                    nn.LayerNorm(self.embed_dim),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.GELU(),
                    nn.Dropout(kwargs["dropout"]),
                )
            elif self.refine_block_input_mode == "concat_raw_mlp":
                bridge = nn.Sequential(
                    nn.LayerNorm(2 * self.embed_dim),
                    nn.Linear(2 * self.embed_dim, self.embed_dim),
                    nn.GELU(),
                    nn.Dropout(kwargs["dropout"]),
                )
            else:
                raise ValueError(
                    f"Unknown refine_block_input_mode={self.refine_block_input_mode!r}"
                )
            self.refine_blocks.append(
                nn.ModuleDict(
                    {
                        "layers": layers,
                        "final_norm": final_norm,
                        "bridge": bridge,
                    }
                )
            )
        self.apply(
            lambda m: init_params(
                m,
                self.n_layers
                + ((self.num_model_blocks - 1) * self.refine_block_n_layers),
            )
        )

    def _build_refine_block_input(self, prev_x, encoder_input_x, bridge):
        if self.refine_block_input_mode == "direct":
            return prev_x
        if self.refine_block_input_mode == "residual_mlp":
            return prev_x + bridge(prev_x)
        return bridge(torch.cat([prev_x, encoder_input_x], dim=-1))

    def forward(
        self,
        tokenized_tree_batch,
        t=None,
        phyla_embeddings=None,
        return_all_tokens=True,
        return_leafs_only=False,
        return_edges_only=False,
        return_edge_features=False,
        return_first_hit_logits=False,
        return_boundary_vanish_logits=False,
        autoregressive=False,
        autoregressive_component_groups=None,
    ):
        x, padding_mask, leaf_mask, leaf_idx, edge_mask, edge_split_masks = (
            self._prepare_encoder_inputs(
                tokenized_tree_batch,
                t=t,
                phyla_embeddings=phyla_embeddings,
            )
        )
        encoder_input_x = x
        x = self._encode_with_layers(
            x,
            padding_mask=padding_mask,
            layers=self.layers,
            final_layer_norm=self.final_layer_norm,
        )
        for block in self.refine_blocks:
            block_input_x = self._build_refine_block_input(
                x,
                encoder_input_x,
                block["bridge"],
            )
            x = self._encode_with_layers(
                block_input_x,
                padding_mask=padding_mask,
                layers=block["layers"],
                final_layer_norm=block["final_norm"],
            )
        return self._decode_outputs(
            x,
            leaf_mask=leaf_mask,
            leaf_idx=leaf_idx,
            edge_mask=edge_mask,
            edge_split_masks=edge_split_masks,
            return_all_tokens=return_all_tokens,
            return_leafs_only=return_leafs_only,
            return_edges_only=return_edges_only,
            return_edge_features=return_edge_features,
            return_first_hit_logits=return_first_hit_logits,
            return_boundary_vanish_logits=return_boundary_vanish_logits,
            autoregressive=autoregressive,
            autoregressive_component_groups=autoregressive_component_groups,
        )


def return_model(config):
    model_kwargs = dict(
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
        tokenizer_branch_length_mode=config["model"].get(
            "tokenizer_branch_length_mode", "linear"
        ),
        tokenizer_branch_length_num_buckets=config["model"].get(
            "tokenizer_branch_length_num_buckets", 64
        ),
        tokenizer_branch_length_log_min=config["model"].get(
            "tokenizer_branch_length_log_min", -8.0
        ),
        tokenizer_branch_length_log_max=config["model"].get(
            "tokenizer_branch_length_log_max", 1.0
        ),
        phyla_dim=config["model"]["phyla_dim"],
        phyla_use_leaf_tokens=config["model"].get("phyla_use_leaf_tokens", True),
        phyla_use_split_tokens=config["model"].get("phyla_use_split_tokens", True),
        phyla_leaf_scale=config["model"].get("phyla_leaf_scale", 1.0),
        phyla_split_scale=config["model"].get("phyla_split_scale", 1.0),
        autoregressive_head_mode=config["model"].get(
            "autoregressive_head_mode", "pairwise_threshold"
        ),
        autoregressive_group_refinement_layers=config["model"].get(
            "autoregressive_group_refinement_layers", 0
        ),
        autoregressive_max_subset_size=config["model"].get(
            "autoregressive_max_subset_size", 64
        ),
    )
    model_variant = str(config["model"].get("model_variant", "base"))
    if model_variant == "two_block_refine":
        model = TreeDenoiserTokenGTTwoBlock(
            **model_kwargs,
            block2_n_layers=config["model"].get("block2_n_layers"),
            block2_input_mode=config["model"].get("block2_input_mode", "direct"),
            block2_drop_path_rate=config["model"].get("block2_drop_path_rate"),
        )
    elif model_variant == "multi_block_refine":
        model = TreeDenoiserTokenGTMultiBlock(
            **model_kwargs,
            num_model_blocks=config["model"].get("num_model_blocks", 3),
            refine_block_n_layers=config["model"].get("refine_block_n_layers"),
            refine_block_input_mode=config["model"].get(
                "refine_block_input_mode", "direct"
            ),
            refine_block_drop_path_rate=config["model"].get(
                "refine_block_drop_path_rate"
            ),
        )
    else:
        model = TreeDenoiserTokenGT(**model_kwargs)

    return model
