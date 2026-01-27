import random
import time
import torch, torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from deepspeed.ops.adam import FusedAdam
import wandb
import logging
import gc
import torch.distributed
import gc
import torch
import sys
import os
import torch.nn.functional as F

# Ensure the current directory is in sys.path to import 'phyla'
sys.path.append(os.getcwd())
# Import utilities from the provided codebase
from phyla.utils.utils import load_config
from utils.utils import remove_bit, has_polytomy_fast
from phyla.eval.evo_reasoning_eval import (
Config,
load_model,
_encode_sequences_openfold_style,
)

from utils.random_tree import Tree
from utils.bhv_utils import BHVEncoder
from utils.bhv_movie import build_tree_from_splits
from utils.utils import (
pick_group,
find_polytomy_nodes,
number_to_name_newick,
has_polytomy_fast,
resolve_polytomies_random_deterministic,
_pick_knn_pair,
)
from utils.metric_utils import (
kl_divergence_topological_distributions,
split_bipartition_frequency_correlation,
compare_likelihood_distributions,
compare_branch_length_distributions,
calculate_norm_rf,
)
from data.dataset import PhylaDataModule
from model.model import TreeDenoiserTokenGT
import numpy as np
import logging
from tqdm import tqdm
from utils.utils import compute_merge_metrics

logger = logging.getLogger(__name__)


class TrainingModule(LightningModule):
    def __init__(
        self,
        model: TreeDenoiserTokenGT,
        dataset: PhylaDataModule,
        lr: float = 1e-4,
        record=False,
        epochs: int = 5000,
        lr_scheduler: str = "default",
        num_annealing_steps: int = 10000,
        num_warmup_steps: int = 1000,
        deepspeed: bool = False,
        logger=None,
        max_num_timesteps: int = 20,
        training_sampling_frequency: int = 200,
        training_sampling_start: int = 500,
        num_samples: int = 10,
        dt: float = 0.1,
        # Figure out how to do typing here
        global_splits=None,
        random_trees=None,
        verbose: bool = False,
        phyla_checkpoint_path=None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.record = record
        self.epochs = epochs
        self.warmup_steps = 400
        self.current_step_value = 0
        self.lr_scheduler = lr_scheduler
        self.num_annealing_steps = num_annealing_steps
        self.num_warmup_steps = num_warmup_steps
        self.dataset = dataset
        self.max_num_timesteps = max_num_timesteps
        self.global_splits = global_splits
        self.random_trees = random_trees
        self.verbose = verbose
        self.training_sampling_frequency = training_sampling_frequency
        self.training_sampling_start = training_sampling_start
        self.num_samples = num_samples
        self.dt = dt

        self.automatic_optimization = False
        self.deepspeed = deepspeed
        self.logger_ = logger
        if verbose:
            logging.getLogger("filelock").setLevel(logging.WARNING)
            logging.getLogger("fsspec").setLevel(logging.WARNING)
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        self.phyla_checkpoint_path = phyla_checkpoint_path
        self.phyla_model = None
        self.stepper = 1

        phyla_config_path = "configs/sample_eval_config.yaml"

        if self.phyla_checkpoint_path is not None:
            original_argv = sys.argv
            sys.argv = ["script", phyla_config_path]
            try:
                if not os.path.exists(phyla_config_path):
                    logging.warning(
                        f"Phyla configuration file not found at {phyla_config_path}"
                    )

                config = load_config(Config)
                config.trainer.checkpoint_path = self.phyla_checkpoint_path
                config.eval.device = "cuda" if torch.cuda.is_available() else "cpu"
                loaded = load_model(config=config, random_model=False)
                self.phyla_model = loaded["model"]
                self.phyla_model.eval()
                if verbose:
                    logging.info("Phyla model loaded successfully.")
            except Exception as e:
                logging.warning(f"Failed to load Phyla model: {e}")
            finally:
                sys.argv = original_argv

    def compute_phyla_embeddings(self, sequences, names, device="cuda"):
        """
        Generates Phyla embeddings for a batch of sequences.
        """
        if self.phyla_model is None:
            raise ValueError("Phyla model not loaded.")

        # This utility handles tokenization, padding, and CLS token placement
        batch, _ = _encode_sequences_openfold_style(sequences, names)

        # Generate Embeddings
        with torch.no_grad():
            encoded_seqs = batch["encoded_sequences"].to(device)
            sequence_mask = batch["sequence_mask"].to(device)
            cls_positions = batch["cls_positions"].bool().to(device)

            self.phyla_model.to(device)

            # Handle different forward pass signatures depending on model wrapper
            if "TrainingModule" in str(type(self.phyla_model)):
                embeddings = self.phyla_model(
                    encoded_seqs,
                    cls_token_mask=cls_positions,
                    sequence_mask=sequence_mask,
                )
            else:
                embeddings = self.phyla_model(
                    encoded_seqs,
                    sequence_mask,
                    cls_positions,
                )

        return embeddings

    def forward(
        self, batched_tokenized_trees, t, phyla_embeddings, autoregressive=False
    ):
        if not autoregressive:
            velocity, mask = self.model(
                batched_tokenized_trees,
                t,
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=False,
                return_edges_only=True,
            )
            edge_split_masks = batched_tokenized_trees[-1]
            edge_mask = batched_tokenized_trees[-2]
            return velocity, edge_split_masks, edge_mask
        else:
            all_group_logits = self.model(
                batched_tokenized_trees,
                t,
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=False,
                return_edges_only=True,
                autoregressive=True,
            )
            return all_group_logits

    def step(self, batch, eval=False, autoregressive=False):
        logs = {}
        if (
            self.phyla_model is not None
            and batch["phyla_embeddings"] is None
            and "ids" in batch
        ):
            phyla_embeddings_list = []
            for i in range(len(batch["ids"])):
                mapping = batch["mappings"][i]
                num_leaf = batch["num_leaves"][i]
                seqs = []
                names = []
                for idx in range(num_leaf):
                    idx_str = str(idx)
                    taxon_name = mapping.get(idx_str)
                    if taxon_name:
                        seq = self.dataset.name_to_seq.get(taxon_name, "")
                        seqs.append(seq)
                        names.append(taxon_name)
                    else:
                        seqs.append("")
                        names.append("unknown")

                embeddings = self.compute_phyla_embeddings(
                    seqs, names, device=str(self.device)
                )
                # Ensure embeddings are (N, D) not (1, N, D)
                if embeddings.dim() == 3 and embeddings.size(0) == 1:
                    embeddings = embeddings.squeeze(0)
                
                phyla_embeddings_list.append(embeddings)

            batch["phyla_embeddings"] = phyla_embeddings_list

        if not autoregressive:
            v_pred, edge_split_masks, edge_mask = self.forward(
                batch["tokenized_trees"],
                batch["batched_time"],
                batch["phyla_embeddings"],
            )
            velocity_labels = batch["batched_velocity"]
            num_leaves = batch["num_leaves"]
            gathered_velocity_labels = []
            v_pred_indices = []

            for num in range(len(velocity_labels)):
                sub_gathered_velocity_labels = []
                sub_v_pred_indices = []

                num_leave = num_leaves[num]
                real_max_bit = max(m.bit_length() for m in edge_split_masks[num])
                for vel in velocity_labels[num]:
                    if vel.bit_length() == real_max_bit + 1:
                        vel = remove_bit(vel, num_leave - 1)
                    elif vel.bit_length() > real_max_bit + 1:
                        raise Exception(
                            f"Whoa there is a big problem with this split mask {vel} vs real max {real_max_bit}!"
                        )

                    if vel not in edge_split_masks[num]:
                        print(
                            f"This split {vel} from velocity labels is not in edge splits {edge_split_masks[num]}!"
                        )
                        print([i for i in range(vel.bit_length()) if (vel >> i) & 1])
                        raise Exception("Split not found in edge splits")
                    # else:
                    # 	print("WOOO ONE FOUND")
                    sub_gathered_velocity_labels.append(velocity_labels[num][vel])
                    sub_v_pred_indices.append(edge_split_masks[num].index(vel))

                gathered_velocity_labels.append(
                    torch.tensor(sub_gathered_velocity_labels)
                )
                v_pred_indices.append(torch.tensor(sub_v_pred_indices))

            # gathered_velocity_labels = torch.stack(gathered_velocity_labels)
            # v_pred_indices = torch.stack(v_pred_indices)

            # Fix: Flatten tensors to handle variable number of edges per tree
            preds_list = []
            for b_idx in range(len(v_pred_indices)):
                indices = v_pred_indices[b_idx].to(v_pred.device)
                if indices.numel() > 0:
                    preds = v_pred[b_idx].index_select(0, indices)
                    preds_list.append(preds)

            if len(preds_list) > 0:
                v_pred_gathered = torch.cat(preds_list).squeeze(-1)
                gathered_velocity_labels_flat = torch.cat(gathered_velocity_labels).to(
                    v_pred_gathered.device
                )
                loss = ((v_pred_gathered - gathered_velocity_labels_flat) ** 2).mean()
            else:
                loss = torch.tensor(0.0, device=v_pred.device, requires_grad=True)
            # print("Wow congrats")
            logs["loss"] = loss
            logger.info(f"Velocity loss: {loss.item()}")
            if self.record:
                wandb.log({"train/velocity_loss": loss.item()}, step=self.stepper)
            # import pdb

            # pdb.set_trace()
        else:
            all_group_logits = self.forward(
                batch["tokenized_autoregressive_trees"],
                batch["batched_autoregressive_time"],
                batch["phyla_embeddings"],
                autoregressive=True,
            )

            found = {}
            for merge_cluser in batch["batched_autoregressive_labels"]:
                for res_split, components in merge_cluser:
                    found[res_split] = False

            losses = []

            max_logits = []
            total_metrics = []

            chosen_polytomies = []
            polytomy_logits = []
            polytomy_sizes = []  # Track size of each polytomy encountered

            for group in all_group_logits:
                logits = group["logits"]
                labels = batch["batched_autoregressive_labels"]
                splits_in_polytomy = group["splits_represented"]
                
                # Track polytomy size (number of splits in the polytomy)
                polytomy_sizes.append(len(splits_in_polytomy))

                y = torch.zeros(logits.size(0), logits.size(1), dtype=torch.long).to(
                    logits.device
                )

                for labeled_merge_cluster in labels:
                    idxs = None
                    for resulting_split, components in labeled_merge_cluster:
                        res = all([i in splits_in_polytomy for i in components])

                        if res:
                            found[resulting_split] = True
                            idxs = [splits_in_polytomy.index(i) for i in components]

                            for i in idxs:
                                for j in idxs:
                                    if i != j:
                                        y[i, j] = 1.0

                if y.sum() == 0:
                    chosen_polytomies.append(torch.tensor(0.0))
                else:
                    chosen_polytomies.append(torch.tensor(1.0))

                polytomy_logits.append(group["polytomy_pred"])

                if y.sum() > 0:
                    y = y.float()

                    G = logits.size(0)
                    mask = ~torch.eye(
                        G, dtype=torch.bool, device=logits.device
                    )  # off-diagonal only

                    # optionally only use one triangle (avoid double-counting symmetric pairs)
                    tri = torch.triu(mask, diagonal=1)

                    logits_vec = logits[tri]
                    y_vec = y[tri]

                    # ignore any -inf (if any sneak in beyond diagonal)
                    finite = torch.isfinite(logits_vec)
                    logits_vec = logits_vec[finite]
                    y_vec = y_vec[finite]

                    max_logits.append(torch.sigmoid(logits_vec).max().item())
                    # AUC calculation for logging

                    metrics = compute_merge_metrics(
                        logits_vec, y_vec, threshold_logit=0.0, topk=(1, 5, 10)
                    )
                    total_metrics.append(metrics)

                    pos = (y_vec > 0.5).nonzero(as_tuple=False).squeeze(-1)
                    neg = (y_vec < 0.5).nonzero(as_tuple=False).squeeze(-1)
                    # import pdb; pdb.set_trace()
                    k_neg = 512
                    if neg.numel() > k_neg:
                        # take top-k hardest negatives by score
                        neg_scores = logits_vec[neg]
                        topk = torch.topk(neg_scores, k=k_neg, largest=True).indices
                        neg = neg[topk]
                    
                    idx = torch.cat([pos, neg])
                    tau = 1.0
                    s = logits_vec[idx] / tau

                    lse_all = torch.logsumexp(s, dim=0)
                    lse_pos = torch.logsumexp(s[: pos.numel()], dim=0)

                    loss = lse_all - lse_pos

                    # import pdb; pdb.set_trace()
                    # INITIAL LOSS FUNCTION
                    # class imbalance weighting
                    # pos = y_vec.sum().clamp(min=1.0)
                    # neg = (y_vec.numel() - y_vec.sum()).clamp(min=1.0)
                    # pos_weight = (neg / pos).detach()

                    # loss = F.binary_cross_entropy_with_logits(
                    #     logits_vec, y_vec, pos_weight=pos_weight
                    # )

                    losses.append(loss)

            for i in found:
                if not found[i]:
                    import pdb; pdb.set_trace()
                    print(
                        "Missing split: ",
                        [j for j in range(int(i).bit_length()) if (int(i) >> j) & 1],
                    )
                    raise Exception(f"Did not find merge for split {i}!")

            L_polytomy_choosing = None

            if len(chosen_polytomies) > 1:
                polytomy_logits_tensor = torch.stack(polytomy_logits).squeeze(1)
                chosen_polytomies_tensor = torch.stack(chosen_polytomies).to(polytomy_logits_tensor.device)

                L_polytomy_choosing = F.binary_cross_entropy_with_logits(
                    polytomy_logits_tensor,
                    chosen_polytomies_tensor,
                ) 

                logger.info(f"Polytomy choosing loss: {L_polytomy_choosing.item()}")
                if self.record:
                    wandb.log({"train/polytomy_choosing_loss": L_polytomy_choosing.item()}, step=self.stepper)

            L_merging = torch.stack(losses).mean()
            logs["loss"] = L_merging
            logger.info(f"Autoregressive loss: {L_merging.item()}")
            logger.info(f"Max autoregressive logit: {np.mean(max_logits)}")

            aggregated_metrics = {}
            if len(total_metrics) > 0:
                for key in total_metrics[0]:
                    aggregated_metrics[key] = sum(
                        m[key] for m in total_metrics
                    ) / len(total_metrics)

                for key in aggregated_metrics:
                    logger.info(f"{key}: {aggregated_metrics[key]}")

            if L_polytomy_choosing is not None:
                logs["loss"] += L_polytomy_choosing

            # Calculate average polytomy size
            avg_polytomy_size = np.mean(polytomy_sizes) if polytomy_sizes else 0.0
            num_polytomies = len(polytomy_sizes)
            logger.info(f"Average polytomy size: {avg_polytomy_size}")

            if self.record:
                # Batch all metrics into a single wandb.log call to avoid step conflicts
                wandb_metrics = {
                    "train/autoregressive_loss": L_merging.item(),
                    "autoregressive_stats/max_autoregressive_logits": np.mean(max_logits),
                    "autoregressive_stats/avg_polytomy_size": avg_polytomy_size,
                    "autoregressive_stats/num_polytomies": num_polytomies,
                }
                wandb_metrics.update(
                    {f"{key}": aggregated_metrics[key] for key in aggregated_metrics}
                )
                wandb.log(wandb_metrics, step=self.stepper)

        return logs

    def sample(
        self,
        newick_starting_trees: list[str],
        phyla_embeddings,
        num_samples=None,
        mapping=None,
        T=1.0,
        dt_base=0.02,
        eps_len=1e-8,
        hit_tol=1e-10,
        max_events=1000,
        max_steps=20000,
        KNN_TOPM = 32,
        KNN_TAU = 0.05,
        KNN_STOCHASTIC = False,
    ):
        if num_samples is None:
            num_samples = self.num_samples

        self.model.eval()
        max_logits = []

        if (
            phyla_embeddings is None
            and self.phyla_model is not None
            and self.dataset is not None
        ):
            # Calculate embeddings on the fly
            t_temp = Tree(newick_starting_trees[0])
            sorted_names = [t_temp.id_to_name[i] for i in range(t_temp.n_leaves)]

            # Filter out ROOT_DUMMY as it has no sequence
            valid_names = [n for n in sorted_names if n != "ROOT_DUMMY"]
            sorted_seqs = [self.dataset.name_to_seq[name] for name in valid_names]

            raw_emb = self.compute_phyla_embeddings(
                sorted_seqs, valid_names, device=self.device
            )
            # raw_emb is (1, N, D). We want (B, N, D).
            if raw_emb.size(0) == 1:
                phyla_embeddings = raw_emb.expand(len(newick_starting_trees), -1, -1)
            else:
                phyla_embeddings = raw_emb.expand(len(newick_starting_trees), -1, -1)

        # SPEED UP SAMPLING
        # 1) init: parse tree -> {mask: length}
        trees = []
        num_leaves = []
        mapping = []
        # Precompute cache for initial trees
        # Since topology changes in the loop, we will update this cache dynamically
        # Initialize tokenized structure cache
        current_newicks = list(newick_starting_trees)
        token_cache = self.model.tokenizer.create_batched_cache(current_newicks)

        for nw in newick_starting_trees:
            t = Tree(nw)
            enc = BHVEncoder()
            masks, lens = enc.return_BHV_encoding(t)
            # Initial trees have no polytomies and all lengths should be greater than 0, so any 0 edges need to be removed
            trees.append({m: float(l) for m, l in zip(masks, lens) if l is not None})
            num_leaves.append(t.n_leaves)
            mapping.append(t.id_to_name)

        t = 0.0
        n_events = 0
        n_steps = 0
        n_topology_changes = 0
        polytomy_sizes = []  # Track sizes of polytomies encountered during sampling

        while t < T and n_steps < max_steps and n_events < max_events:
            n_steps += 1

            # --- encode/tokenize current trees for the model ---

            # Use CACHED tokenizer
            tokenized = self.model.tokenizer.forward_batched(token_cache, trees)

            with torch.no_grad():
                velocity, edge_splits, edge_split_mask = self.forward(
                    tokenized, t, phyla_embeddings
                )

            # ---- FIRST PASS: compute per-tree dt_hit, cache per-tree arrays ----

            dt_hit_list = []
            cache = []
            for b_idx, (td, v, n_leaves, mapp) in enumerate(zip(trees, velocity, num_leaves, mapping)):
                model_masks = edge_splits[b_idx]
                mask_idx = {mask: i for i, mask in enumerate(model_masks)}
                V = v.squeeze(1).detach().cpu().numpy()

                L = []
                V_val = []
                masks = []

                for m in td:
                    if m not in model_masks:
                        # WE GOTTA FIX THIS LOL
                        print(f"Whoa there is a split missing in velocity masks! {m}")
                    else:
                        L.append(td[m])
                        V_val.append(V[mask_idx[m]])
                        masks.append(m)

                V = np.array(V_val, dtype=np.float64)
                L = np.array(L, dtype=np.float64)
                
                if len(V) != len(L):
                    raise Exception("I assume these two things are equal length!")

                if (L < 0).any():
                    raise Exception("There are negative lengths that is not possible!")

                # --- compute dt_hit ---
                neg = (V < 0) & (L > eps_len)
                if np.any(neg):
                    dt_candidates = L[neg] / -V[neg]
                    dt_hit = float(np.min(dt_candidates))
                else:
                    dt_hit = float("inf")

                cache.append((td, L, V, n_leaves, mapp, dt_hit, masks))
                dt_hit_list.append(dt_hit)

            # ---- GLOBAL dt across the batch ----
            dt_hit_global = min(dt_hit_list) if len(dt_hit_list) else float("inf")
            # Experimenting here, dt_hit_global is not a good metric we just jump, jump, jump, so why not use dt_base
            # dt = min(dt_base, dt_hit_global, T - t)
            dt = min(dt_base, T-t)

            # defensive: prevent hard stall
            if dt <= 0:
                dt = min(dt_base, T - t)


            # ---- SECOND PASS: advance everyone with the SAME dt ----
            new_trees = []

            # Since update of token_cache happens per tree potentially, we need to defer it or track which ones changed.
            # However, batch indices align with zip(trees...), so we can update token_cache[i] if needed.

            for b_idx, (td, L, V, n_leaves, mapp, dt_hit, masks) in enumerate(
                cache
            ):
                model_masks = edge_splits[b_idx]
                # --- advance ---
                L_new = L + dt * V

                # Did we hit boundary this step?
                hit_boundary = (abs(dt - dt_hit) <= hit_tol) or (L_new <= eps_len).any()

                if hit_boundary:
                    hit = L_new <= eps_len
                    L_new[hit] = 0.0
                
                # update dict
                td2 = {m: float(l) for m, l in zip(masks, L_new) if l > eps_len}

                # We only need to rebuild Newick/Graph if we hit a boundary (topology changed)
                if hit_boundary:
                    num_merges = 0
                    topology_changed = True
                    while topology_changed:
                        graph, td2_newick = build_tree_from_splits(
                            list(td2.keys()),
                            td2,
                            n_leaves,
                            root_leaf=n_leaves - 1,
                            mapping=mapp,
                        )

                        polytomy_nodes = has_polytomy_fast(td2_newick, unrooted_ok=False)
                        # td2 = {m: float(l) for m, l in zip(active_masks, L_new)}

                        if polytomy_nodes:
                            # For autoregressive step, we just use standard tokenizer for now as it's rare event
                            tokenized_trees = self.model.tokenizer([td2_newick])
                            # import pdb; pdb.set_trace()

                            with torch.no_grad():
                                logit_outputs = self.forward(
                                    tokenized_trees,
                                    torch.tensor([num_merges/162], device=self.device),
                                    phyla_embeddings,
                                    autoregressive=True,
                                )
                            top_change = False
                            for output in logit_outputs:
                                # Track polytomy size
                                polytomy_sizes.append(len(output["splits_represented"]))
                                
                                if torch.sigmoid(output["polytomy_pred"]).item() < 0.5:
                                    x = output["logits"]
                                    W = 0.5 * (x + x.T)  # [G,G]
                                    W.fill_diagonal_(-float("inf"))
                                    P = torch.sigmoid(
                                        W
                                    )  # mergeability prob, pick_group already does the sigmoid but I'm doing it here for logging later
                                    # Can do something here to look at the prob of merging to see if the model is really learning anything or just learning junk for logging purposes
                                    # import pdb; pdb.set_trace()
                                    max_logits.append(P.max().item())
                                    res = pick_group(W, tau=0.55)
                                    if res is None:
                                        logger.debug("No merges found!")
                                    else:
                                        logger.debug(f"Merges found: {res}")
                                        # import pdb; pdb.set_trace()
                                        split_masks = [
                                            output["splits_represented"][idx] for idx in res
                                        ]
                                        new_split = 0
                                        for sm in split_masks:
                                            new_split |= sm

                                        if new_split in td2:
                                            if len(res) > 2:
                                                logger.debug("Whoa already in there! Trying partial merge")
                                                res = res[:-1]
                                                split_masks = [
                                                    output["splits_represented"][idx] for idx in res
                                                ]
                                                new_split = 0
                                                for sm in split_masks:
                                                    new_split |= sm
                                            
                                        if new_split in td2:
                                            logger.debug("Whoa already in there!")
                                        else:
                                            # New length is average of merged splits
                                            td2[new_split] = 1e-3
                                            top_change = True
                            
                            if not top_change:
                                logger.info("No more merges possible, pick a random polytomy and do a KNN merge")
                                output = random.choice(logit_outputs)
                                split_embeddings = output['group_embeddings']
                                group_represented = output['splits_represented']

                                if len(group_represented) != split_embeddings.size(0):
                                    raise Exception("Whoa size mismatch between groups and split embeddings")
                                
                                i, j = _pick_knn_pair(split_embeddings, topM=KNN_TOPM, tau=KNN_TAU, stochastic=KNN_STOCHASTIC)

                                sm_i, sm_j = group_represented[i], group_represented[j]
                                new_split = int(sm_i) | int(sm_j)

                                if new_split not in td2:
                                    # td2[new_split] = 1e-3  # tiny length
                                    curr_lens = list(td2.values())
                                    if len(curr_lens) > 0:
                                        td2[new_split] = float(np.median(curr_lens))
                                    else:
                                        td2[new_split] = 1e-3
                                else:
                                    # import pdb; pdb.set_trace()
                                    raise Exception("Not possible to merge into a split that already exists...")

                                top_change = True
                                num_merges += 1
                                n_events += 1
                                n_topology_changes += 1
                            
                            if not top_change:
                                topology_changed = False
                            else:
                                num_merges += 1

                            n_events += 1
                            logger.debug("Finished processing merges")
                            if topology_changed:
                                n_topology_changes += 1
                                
                        else:
                            topology_changed = False
                            


                    _, td2_newick_final = build_tree_from_splits(
                        list(td2.keys()),
                        td2,
                        n_leaves,
                        root_leaf=n_leaves - 1,
                        mapping=mapp,
                    )
                    # Update the cache for this batch index
                    new_item = self.model.tokenizer.compute_structural_cache(
                        [td2_newick_final]
                    )[0]

                    token_cache.update(b_idx, new_item)

                new_trees.append(td2)

            trees = new_trees
            t += dt

            if n_steps % 100 == 0:
                print(f"Step {n_steps}: dt={dt:.2e}, t={t:.2f}/{T}")

        # print(f"Sampling finished in {n_steps} steps. Total events: {n_events}")
        avg_polytomy_size = np.mean(polytomy_sizes) if polytomy_sizes else 0.0
        # if num_topology_changes > 0:
        #     import pdb; pdb.set_trace()
    
        logger.info(f"Sampling finished in {n_steps} steps. Total events: {n_events}, topology changes: {n_topology_changes}, average polytomy size: {avg_polytomy_size:.2f}")

        return [
            build_tree_from_splits(
                list(td.keys()),
                td,
                n_leaves=n_leaves,
                root_leaf=n_leaves - 1,
                mapping=mapp,
            )[1]
            for td, n_leaves, mapp in zip(trees, num_leaves, mapping)
        ], n_topology_changes, sum(max_logits) / len(max_logits) if len(max_logits) > 0 else 0.0, avg_polytomy_size, len(polytomy_sizes)

    def sample_compare(self, batch, train=True, num_samples=None, dt=0.02, save = True):
        if num_samples is None:
            num_samples = self.num_samples
        nexus_filepaths = batch["nexus_filepaths"]
        tree_paths = batch["tree_paths"]
        ids = batch["ids"]

        if len(set(nexus_filepaths)) != 1 or len(set(ids)) != 1:
            raise Exception(
                "Each batch should correspond to one ID, not multiple different IDs, logic is inconsitent somewhere"
            )

        nexus_filepath = batch["nexus_filepaths"][0]
        id = batch["ids"][0]
        mapping = batch["mappings"][0]

        if train:
            real_trees = self.dataset.dataset_train.return_posterior_trees(id)
            num_leaves = self.dataset.dataset_train.return_number_leaves(id)
        else:
            real_trees = self.dataset.dataset_val.return_posterior_trees(id)
            num_leaves = self.dataset.dataset_val.return_number_leaves(id)

        if len(real_trees) > num_samples:
            real_trees = random.sample(real_trees, num_samples)

        for i in real_trees:
            if has_polytomy_fast(i):
                raise Exception(
                    "Whoa there is a polytomy in the real trees, need to resolve first!"
                )

        sampled_trees = []
        num_topology_changes = []
        avg_max_logits = []
        num_polytomies = 0
        avg_polytomy_sizes = []
        num_polytomies_resolved = []

        for _ in tqdm(range(num_samples)):
            rt = Tree(num_leaves=num_leaves, random=True)
            starting_tree = str(rt)
            start_time = time.time()
            sampled_tree, n_topo_changes, avg_max_logit, avg_polytomy_size, n_polyt_resolved = self.sample(
                [starting_tree], batch["phyla_embeddings"], num_samples=num_samples, dt_base=dt
            )
            print(f"Sampling a single tree took {time.time() - start_time} seconds")

            avg_polytomy_sizes.append(avg_polytomy_size)
            num_polytomies_resolved.append(n_polyt_resolved)

            sampled_tree = sampled_tree[0]
            num_topology_changes.append(n_topo_changes)
            avg_max_logits.append(avg_max_logit)
            if has_polytomy_fast(sampled_tree):
                sampled_tree = resolve_polytomies_random_deterministic(sampled_tree)
                if has_polytomy_fast(sampled_tree):
                    raise Exception(
                        "Whoa there is STILL a polytomy in the sampled tree, something is wrong!"
                    )
                num_polytomies += 1

            # Now do something with the sampled tree and the real trees
            sampled_trees.append(sampled_tree)

        sampled = [number_to_name_newick(i, {int(i):v for i, v in mapping.items()}, True) for i in sampled_trees]
        posterior_trees = [number_to_name_newick(i, {int(i):v for i, v in mapping.items()}, False) for i in real_trees]

        rf_dists = []
        n_pairs = min(len(sampled), len(posterior_trees))
        for i in range(n_pairs):
            rf_dists.append(calculate_norm_rf(sampled[i], posterior_trees[i]))
        
        rf_norm_val = np.mean(rf_dists) if rf_dists else 0.0

        if save:
            import pickle
            with open(f"samples/sample_trees_{self.global_step}.pkl", "wb") as f:
                pickle.dump((sampled, posterior_trees), f)

        try:
            metrics = compare_likelihood_distributions(
                nexus_filepath, true_trees=posterior_trees, sampled_trees=sampled, threads=1
            )
        except Exception as e:
            print(f"An error occurred during likelihood comparison: {e}")
            metrics = {}
        
        metrics["rf_norm"] = float(rf_norm_val)

        metrics.update(
            kl_divergence_topological_distributions(
                posterior_trees, sampled, num_leaves=num_leaves
            )
        )
        metrics.update(
            split_bipartition_frequency_correlation(
                posterior_trees, sampled, num_leaves=num_leaves
            )
        )
        metrics.update(compare_branch_length_distributions(posterior_trees, sampled))
        print(f"Num polytomies resolved in sampling: {num_polytomies} out of {num_samples}")
        print("Average topology changes during sampling: ", np.mean(num_topology_changes))
        print("Average max logits during sampling: ", np.mean(avg_max_logits))
        overall_avg_polytomy_size = np.mean([s for s in avg_polytomy_sizes if s > 0]) if any(s > 0 for s in avg_polytomy_sizes) else 0.0
        print(f"Average polytomy size during sampling: {overall_avg_polytomy_size:.2f}")
        
        avg_num_polytomies_resolved = np.mean(num_polytomies_resolved)
        print(f"Average number of polytomies resolved during sampling: {avg_num_polytomies_resolved}")
        
        if self.record:
            wandb.log(
                {
                    "number_of_polytomies_resolved": num_polytomies,
                    "average_topology_changes": np.mean(num_topology_changes),
                    "average_max_logits": np.mean(avg_max_logits),
                    "samples/average_num_polytomies_resolved": avg_num_polytomies_resolved,
                },
                step=self.stepper,
            )

        return metrics
        
    def on_train_end(self):
        if self.record:
            wandb.finish()

    def training_step(self, batch, _):
        # Skip if batch is None (all items failed tokenization in collate_fn)
        if batch is None:
            logging.warning("Skipping training step: batch is None (tokenization failed for all items)")
            print("Skipping training step: batch is None (tokenization failed for all items)")
            return None
        
        # Increment stepper at the START to ensure all logs in this step use the same step number
        self.stepper += 1
        
        opt = self.optimizers()
        opt.zero_grad()

        if self.deepspeed:

            success = False
            num = 0
            failed = False

            # Logic, if we have an out of memmory error we just resample with a smaller subtree and rerun
            while not success:
                self.logger_.log(f"Entering step {num}", level=logging.INFO)
                error_tensor = torch.zeros(1).cuda()
                if num > 1:
                    self.logger_.log(
                        f"Batch is too large decreasing max tree size by a factor of 2 and num sequences",
                        level=logging.INFO,
                    )
                    if "loss" in locals():
                        loss = loss.detach()
                        del loss
                        gc.collect()

                    if num > 10:
                        return torch.tensor(0)

                    torch.cuda.empty_cache()
                    torch.distributed.barrier()
                    index, sub_tree_size, num_subtrees = self.dataset.chosen_tree

                    if sub_tree_size <= 5:
                        self.logger_.log(
                            f"We have reached the minimum tree size", level=logging.INFO
                        )
                        num_subtrees = 5
                        sub_tree_size = 5
                    elif num_subtrees > 100:
                        self.logger_.log(
                            f"Number of subtrees way too big {torch.distributed.get_rank()}",
                            level=logging.INFO,
                        )
                        num_subtrees = 50
                    else:
                        num_subtrees = int(num_subtrees // 2)
                        sub_tree_size = int(sub_tree_size // 2)

                        if sub_tree_size < 5:
                            sub_tree_size = 5
                        if num_subtrees < 1:
                            num_subtrees = 1

                    sub_batch = self.dataset.__getitem__(
                        index, preset_subtree_size=sub_tree_size
                    )
                    batch = self.dataset.collate_fn(
                        [sub_batch], preset_subtree_num=num_subtrees
                    )

                    # TODO: Run without adaptive batch size speedup
                    # TODO: Run with adaptive batch size speedup
                    if num <= 2:
                        new_max_aa = (
                            num_subtrees
                            * sub_tree_size
                            * self.dataset.return_max_length(self.dataset.name_to_seq)
                        )
                        self.logger_.log(
                            f"Updating the adaptive batch size sampler with this new information of the max aa of {new_max_aa}",
                            level=logging.INFO,
                        )
                        self.dataset.size_detector.update_max_aa(new_max_aa)

                    torch.distributed.barrier()
                    self.logger_.log(
                        f"We have all recreated our batches now moving on",
                        level=logging.INFO,
                    )
                try:
                    loss_status_tensor = torch.zeros(
                        torch.distributed.get_world_size()
                    ).cuda()
                    logs = self.step(batch)
                    if logs is not None:
                        loss = logs["loss"]
                        memory_error_tensor = torch.zeros(1).cuda()

                        # Go through every GPU get memmory used, if it is above 70% we will abort the manual backward and fail
                        stop_manual_backward = False
                        for i in range(torch.cuda.device_count()):
                            # Get the current memory usage
                            current_memory = torch.cuda.memory_allocated(i)
                            # Get the total memory
                            total_memory = torch.cuda.get_device_properties(
                                i
                            ).total_memory
                            fraction = current_memory / total_memory
                            if fraction > 0.75:
                                self.logger_.log(
                                    f"We detected that {i} device is above 75% memory usage!, will avoid manual backward!",
                                    level=logging.INFO,
                                )
                                stop_manual_backward = True
                                memory_error_tensor[0] = 1
                            self.logger_.log(
                                f"Device {i} is using {fraction} of its memory",
                                level=logging.INFO,
                            )
                            torch.distributed.barrier()

                        # If one at least fails the memory check then we will scuttle the backward
                        torch.distributed.all_reduce(memory_error_tensor)
                        if memory_error_tensor[0] > 0:
                            self.logger_.log(
                                f"Wow some is about to OOM we are scuttling the backward",
                                level=logging.INFO,
                            )
                            stop_manual_backward = True

                        # Okay what if one passes the memory check and still fails?
                        # loss_status_tensor = torch.zeros(1).cuda()

                        if not stop_manual_backward:
                            self.manual_backward(loss)
                            success = True
                            failed = False
                            self.logger_.log(f"Succeded!", level=logging.INFO)
                            loss_status_tensor[torch.distributed.get_rank()] = 1
                        else:
                            self.logger_.log(f"Skipping backward!", level=logging.INFO)
                            failed = True
                            success = False
                            num += 1
                            logs = None
                            loss_status_tensor[torch.distributed.get_rank()] = 1

                    else:
                        self.logger_.log(f"Failed!", level=logging.INFO)
                        num += 1
                        loss_status_tensor[torch.distributed.get_rank()] = 1
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.logger_.log(f"WARNING: out of memory", level=logging.INFO)
                        error_tensor[0] = 1
                        failed = True
                        logs = {"loss": torch.tensor(0)}
                        num += 1
                        loss_status_tensor[torch.distributed.get_rank()] = 1
                        self.logger_.log(f"Set up my status", level=logging.INFO)
                    else:
                        self.logger_.log(f"RAISING NEW ERROR {e}", level=logging.INFO)
                        raise e
                finally:
                    self.logger_.log(f"Entering check for the loss", level=logging.INFO)

                    while (
                        loss_status_tensor.sum() != torch.distributed.get_world_size()
                    ):
                        torch.distributed.all_reduce(loss_status_tensor)
                        self.logger_.log(
                            f"Waiting for everyone to finish\t{loss_status_tensor.sum()}\t{loss_status_tensor}",
                            level=logging.INFO,
                        )

                    torch.distributed.barrier()
                    torch.distributed.all_reduce(error_tensor)
                    if error_tensor[0] > 0:
                        self.logger_.log(
                            "Ooops someone had a OOM we should scuttle",
                            level=logging.INFO,
                        )
                        failed = True
                        success = False
                        num += 1

                    # print("Waiting")
                    torch.distributed.barrier()

                num += 1
                torch.distributed.barrier()
        else:
            success = False
            num = 0

            # Logic, if we have an out of memmory error we just resample with a smaller subtree and rerun
            while not success:

                # If fail will call zero grad again, may need this for deepspeed?
                opt.zero_grad()
                if num > 0:
                    logging.info(
                        "Batch is too large decreasing max tree and number of subtrees by a factor of 1.2"
                    )
                    index, sub_tree_size, num_subtrees = self.dataset.chosen_tree
                    new_sub_tree_size = sub_tree_size
                    new_num_subtrees = int(num_subtrees // 1.2)

                    if new_num_subtrees == 0:
                        new_num_subtrees = 1
                        new_sub_tree_size = int(sub_tree_size // 1.2)

                    if new_sub_tree_size < 5:
                        new_sub_tree_size = 5
                        new_num_subtrees = 1

                    if num <= 2:
                        new_max_aa = (
                            new_num_subtrees
                            * new_sub_tree_size
                            * self.dataset.return_max_length(self.dataset.name_to_seq)
                        )
                        logging.info(
                            f"Updating the adaptive batch size sampler with this new information of the max aa of {new_max_aa}"
                        )
                        self.dataset.size_detector.update_max_aa(new_max_aa)

                    if num > 10:
                        logging.info("We are spiraling, moving on")
                        return torch.tensor(0)

                    sub_batch = self.dataset.__getitem__(
                        index, preset_subtree_size=new_sub_tree_size
                    )
                    batch = self.dataset.collate_fn(
                        [sub_batch], preset_subtree_num=new_num_subtrees
                    )
                    logging.info(
                        f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                    )
                    logging.info(
                        f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                    )

                    gc.collect()
                try:
                    logging.info(
                        f"Memory allocated before step: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                    )
                    logging.info(
                        f"Memory reserved before step: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                    )

                    # --- HEAD 1: VELOCITY ---
                    logging.info("DEBUG: Starting Velocity Head Training")
                    logs_vel = self.step(batch, autoregressive=False)
                    loss_vel = logs_vel["loss"]
                    self.manual_backward(loss_vel)
                    self.clip_gradients(
                        opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
                    )
                    opt.step()
                    opt.zero_grad()
                    logging.info("DEBUG: Finished Velocity Head Training")

                    del logs_vel
                    del loss_vel
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()

                    # --- HEAD 2: AUTOREGRESSIVE ---
                    logging.info("DEBUG: Starting Autoregressive Head Training")
                    logs = self.step(batch, autoregressive=True)
                    if "loss" not in logs:
                        import pickle

                        with open("debug_batch.pkl", "wb") as f:
                            pickle.dump(batch, f)
                        raise Exception(
                            "Loss not found in logs for autoregressive head!"
                        )
                    loss = logs["loss"]

                    logging.info(
                        f"Memory allocated before backward: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                    )
                    logging.info(
                        f"Memory reserved before backward: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                    )

                    self.manual_backward(loss)
                    self.clip_gradients(
                        opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
                    )
                    opt.step()
                    opt.zero_grad()

                    success = True
                    failed = False

                    logging.info(
                        f"Memory allocated after backward: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                    )
                    logging.info(
                        f"Memory reserved after backward: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                    )

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.warning("WARNING: out of memory")
                        if hasattr(torch.cuda, "empty_cache"):
                            # Not sure about this
                            torch.cuda.empty_cache()

                        logging.info(
                            f"Memory allocated after OOM: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                        )
                        logging.info(
                            f"Memory reserved after OOM: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                        )

                        num += 1
                    else:
                        raise e

        # print(f"Entering a new world with status {failed}")
        if not failed and logs is not None:
            for k, v in logs.items():
                self.log(
                    k,
                    v.to("cuda"),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

            index, sub_tree_size, num_subtrees = self.dataset.chosen_tree
            lr = opt.optimizer.param_groups[0]["lr"]
            self.log("num_seq_per_subtree", sub_tree_size)
            logs["num_seq_per_subtree"] = sub_tree_size
            self.log("num_subtrees", num_subtrees)
            logs["num_subtrees"] = num_subtrees
            self.log("lr", lr)
            logs["lr"] = lr
            if self.logger_ is not None:
                self.logger_.log(logs, level=logging.INFO)
        else:
            print(logs)

        if logs is not None:
            if self.record:
                # wandb.log(logs)
                wandb.log(logs, step=self.stepper)
            if not self.dataset.msa_distance:
                self.dataset.update_normrf(logs["norm_rf_distance"])

            if self.deepspeed:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=1.0,  # tighten / loosen here
                    gradient_clip_algorithm="norm",
                )

            self.current_step_value += 1
            if self.deepspeed:
                opt.step()
            # print("Hi Im here waiting!")
            if self.deepspeed:
                torch.distributed.barrier()

            # Perform learning rate schedling
            if self.lr_scheduler == "cosine":
                sch1 = self.lr_schedulers()
                sch1.step()
            elif self.lr_scheduler == "cosine_warmup":
                sch1, sch2 = self.lr_schedulers()
                # Perform warmup
                if self.num_warmup_steps > 0:
                    sch1.step()
                    self.num_warmup_steps -= 1
                # Perform cosine annealing
                else:
                    sch2.step()
            elif self.lr_scheduler == "warmup":
                sch1 = self.lr_schedulers()
                # Perform warmup
                if self.num_warmup_steps > 0:
                    sch1.step()
                    self.num_warmup_steps -= 1

            # ADD CODE HERE TO UPDATE ADAPTIVE BATCH SIZE SAMPLER

            if self.global_step >= self.training_sampling_start and (self.global_step - self.training_sampling_start) % self.training_sampling_frequency == 0:
                metrics = self.sample_compare(batch, train=True, dt=self.dt)
                for k, v in metrics.items():
                    self.log(f"sample_metrics/{k}", v, on_step=True, logger=True)
                if self.record:
                    wandb.log({f"sample_metrics/{k}": v for k, v in metrics.items()}, step=self.stepper)
                print(metrics)

            return logs["loss"]
        else:
            return torch.tensor(0)

    def validation_step(self, batch, batch_idx):
        pass

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        norms = grad_norm(self, norm_type=2)
        if "grad_2.0_norm_total" in norms:
            total = norms["grad_2.0_norm_total"]
        else:
            total = norms.get("total_grad_norm", 0.0)  # hypothetical fallback
            if total == 0.0:
                # Just take the first key that looks like total if exists
                keys = [k for k in norms.keys() if "total" in k]
                if keys:
                    total = norms[keys[0]]

        # total = norms.get("grad_2.0_norm_total", 0.0)

        layer_norms = {k: v for k, v in norms.items() if "total" not in k}
        if layer_norms:
            max_grad = max(layer_norms.values())
            mean_grad = torch.mean(torch.stack(list(layer_norms.values())))
        else:
            max_grad = 0.0
            mean_grad = 0.0

        self.log("grad_norm_max", max_grad, prog_bar=True, on_step=True)
        self.log("grad_norm_mean", mean_grad, prog_bar=False, on_step=True)

        # Print a warning if exploding
        if max_grad > 1:
            print(
                f"[Warning] Gradient norm unusually high: max={max_grad:.2e}, mean={mean_grad:.2e}"
            )

        self.log("grad_norm_total", total)
        print(
            f"step {self.global_step:4d}  total_grad_norm = {total:.2f} mean is {mean_grad:.2f} max is {max_grad:.2f}"
        )
        if self.record:
            wandb.log({
                "grad/grad_norm_total": total,
                "grad/grad_norm_max": max_grad,
                "grad/grad_norm_mean": mean_grad,
            }, step=self.stepper)

    def configure_optimizers(self):
        if self.deepspeed:
            optimizer = FusedAdam(self.parameters(), lr=self.lr)
        else:
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        if self.lr_scheduler == "cosine":
            sch1 = CosineAnnealingLR(
                optimizer, T_max=self.num_annealing_steps
            )  # Set to current number of steps for training 7 days
            return [optimizer], [sch1]
        elif self.lr_scheduler == "cosine_warmup":
            sch1 = LinearLR(
                optimizer, start_factor=self.lr, total_iters=self.num_warmup_steps
            )
            sch2 = CosineAnnealingLR(optimizer, T_max=self.num_annealing_steps)
            return [optimizer], [sch1, sch2]
        elif self.lr_scheduler == "warmup":
            sch1 = LinearLR(
                optimizer, start_factor=self.lr, total_iters=self.num_warmup_steps
            )
            return [optimizer], [sch1]
        else:
            scheduler = []
            return optimizer
