import random
import time
import inspect
import math
import json
import functools
import torch, torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import wandb
import logging
import gc
import torch.distributed
import gc
import torch
import sys
import os
import torch.nn.functional as F
from ete3 import Tree as EteTree

# Ensure the current directory is in sys.path to import 'phyla'
sys.path.append(os.getcwd())
# Import utilities from the provided codebase
from utils.utils import remove_bit, has_polytomy_fast

from utils.random_tree import Tree
from utils.bhv_utils import (
    BHVEncoder,
    get_structural_polytomy_groups_from_newick,
    return_sampled_tree_boundary_decisions,
    return_sampled_tree_orthant_velocity,
    return_tree_boundary_merge_paths,
)
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
from utils.utils import _velocity_diagnostics

logger = logging.getLogger(__name__)

try:
    from deepspeed.ops.adam import FusedAdam
except Exception:
    FusedAdam = optim.Adam


def _load_phyla_runtime():
    from phyla.utils.utils import load_config
    from phyla.eval.evo_reasoning_eval import (
        Config,
        load_model,
        _encode_sequences_openfold_style,
    )

    return load_config, Config, load_model, _encode_sequences_openfold_style


def _decode_positive_merge_subsets(group_output, threshold_logit=0.0):
    splits = [int(split) for split in group_output["splits_represented"]]
    logits = group_output["logits"].detach()
    num_splits = len(splits)
    adjacency = {idx: set() for idx in range(num_splits)}

    for i in range(num_splits):
        for j in range(i + 1, num_splits):
            score = float(logits[i, j].item())
            if not math.isfinite(score) or score <= float(threshold_logit):
                continue
            adjacency[i].add(j)
            adjacency[j].add(i)

    visited = set()
    decoded_subsets = []

    for idx in range(num_splits):
        if idx in visited:
            continue

        stack = [idx]
        component = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(adjacency[node] - visited)

        if len(component) >= 2:
            decoded_subsets.append(
                tuple(sorted(int(splits[node_idx]) for node_idx in component))
            )

    return decoded_subsets


def _score_merge_subset(group_output, subset):
    splits = [int(split) for split in group_output["splits_represented"]]
    split_to_index = {split: idx for idx, split in enumerate(splits)}
    logits = group_output["logits"].detach()
    subset_indices = [split_to_index[int(split)] for split in subset if int(split) in split_to_index]

    if len(subset_indices) < 2:
        return float("-inf")

    scores = []
    for i, left_idx in enumerate(subset_indices):
        for right_idx in subset_indices[i + 1 :]:
            score = float(logits[left_idx, right_idx].item())
            if math.isfinite(score):
                scores.append(score)

    return float(sum(scores) / len(scores)) if scores else float("-inf")


def _plan_autoregressive_boundary_merges(logit_outputs, existing_splits, threshold_logit=0.0):
    existing_splits = {int(split) for split in existing_splits}
    outputs_sorted = sorted(
        logit_outputs,
        key=lambda output: float(output["polytomy_pred"].detach().cpu().item()),
        reverse=True,
    )

    planned = []
    planned_new_splits = set()
    for output in outputs_sorted:
        polytomy_score = float(output["polytomy_pred"].detach().cpu().item())
        if (len(logit_outputs) != 1) and polytomy_score <= float(threshold_logit):
            continue

        valid_subsets = []
        for subset in _decode_positive_merge_subsets(output, threshold_logit=threshold_logit):
            new_split = 0
            for component in subset:
                new_split |= int(component)

            if new_split in existing_splits or new_split in planned_new_splits:
                continue

            valid_subsets.append((subset, int(new_split)))

        if valid_subsets:
            best_subset = max(
                valid_subsets,
                key=lambda item: _score_merge_subset(output, item[0]),
            )
            planned_new_splits.add(int(best_subset[1]))
            planned.append(
                {
                    "polytomy_score": polytomy_score,
                    "splits_represented": [int(split) for split in output["splits_represented"]],
                    "subsets": [best_subset],
                    "logits": output["logits"],
                }
            )

    return planned


def _is_strict_subset_mask(mask, region):
    mask = int(mask)
    region = int(region)
    return mask != region and (mask & ~region) == 0


def _move_tokenized_batch_to_device(tokenized, device):
    moved = []
    for item in tokenized:
        if torch.is_tensor(item):
            moved.append(item.to(device))
        else:
            moved.append(item)
    return tuple(moved)


def _extract_edge_splits_from_tokenized(tokenized, batch_index=0):
    edge_masks = tokenized[-1][batch_index]
    return [int(split) for split in edge_masks if int(split) != 0]


def _subset_target_matrix(group_splits, subset, device):
    subset = {int(split) for split in subset}
    size = len(group_splits)
    target = torch.zeros(size, size, dtype=torch.float32, device=device)
    subset_indices = [
        idx for idx, split in enumerate(group_splits) if int(split) in subset
    ]
    for i in subset_indices:
        for j in subset_indices:
            if i != j:
                target[i, j] = 1.0
    return target


def _ready_target_merge_subsets_for_group(group_splits, target_newick, n_leaves):
    group_splits = tuple(sorted({int(split) for split in group_splits}))
    if len(group_splits) < 2:
        return []

    biological_bits = max(int(n_leaves) - 1, 0)
    if biological_bits <= 0:
        return []
    full_mask = (1 << biological_bits) - 1

    parent_mask = 0
    for split in group_splits:
        parent_mask |= int(split)

    target_tree = Tree(target_newick)
    enc = BHVEncoder()
    target_masks, target_lengths = enc.return_BHV_encoding(target_tree)
    dummy_bit_idx = target_tree.n_leaves - 1

    target_clusters = set()
    for raw_mask, raw_length in zip(target_masks, target_lengths):
        if raw_length is None or float(raw_length) <= 0.0:
            continue

        mask = int(raw_mask)
        if biological_bits > 0 and ((mask >> dummy_bit_idx) & 1):
            mask = remove_bit(mask, dummy_bit_idx)
        elif biological_bits > 0 and mask.bit_length() > biological_bits:
            continue

        oriented = None
        candidates = [mask]
        if full_mask:
            candidates.append(full_mask ^ mask)
        for candidate in candidates:
            candidate = int(candidate)
            if candidate in (0, full_mask, parent_mask):
                continue
            if (candidate & ~parent_mask) == 0:
                oriented = candidate
                break
        if oriented is not None:
            target_clusters.add(oriented)

    atoms = tuple(sorted(group_splits))
    atom_set = set(atoms)
    relevant_clusters = sorted(
        [cluster for cluster in target_clusters if cluster not in atom_set],
        key=lambda cluster: (int(cluster).bit_count(), int(cluster)),
    )

    child_map = {}
    for cluster in relevant_clusters:
        candidates = [atom for atom in atoms if (int(atom) & ~int(cluster)) == 0]
        candidates.extend(
            other
            for other in relevant_clusters
            if _is_strict_subset_mask(other, cluster)
        )

        maximal_children = []
        for candidate in candidates:
            dominated = False
            for other in candidates:
                if _is_strict_subset_mask(candidate, other) and _is_strict_subset_mask(
                    other, cluster
                ):
                    dominated = True
                    break
            if not dominated:
                maximal_children.append(int(candidate))

        maximal_children = tuple(sorted(set(maximal_children)))
        union = 0
        for child in maximal_children:
            union |= int(child)

        if union == int(cluster) and len(maximal_children) >= 2:
            child_map[int(cluster)] = maximal_children

    ready_subsets = sorted(
        {
            children
            for children in child_map.values()
            if all(int(child) in atom_set for child in children)
        },
        key=lambda subset: (len(subset), tuple(int(split) for split in subset)),
    )
    return ready_subsets


def _apply_merge_subset_to_newick(tokenizer, current_newick, subset, new_split=None):
    if tokenizer is None:
        tree = Tree(current_newick)
        existing_splits, _ = BHVEncoder().return_BHV_encoding(tree)
        existing_splits = [int(split) for split in existing_splits if int(split) != 0]
    else:
        tokenized = tokenizer([current_newick])
        existing_splits = _extract_edge_splits_from_tokenized(tokenized, batch_index=0)

    if new_split is None:
        new_split = 0
        for component in subset:
            new_split |= int(component)

    new_split = int(new_split)
    if new_split in existing_splits:
        return None

    split_lengths = {int(split): 0.1 for split in existing_splits}
    split_lengths[new_split] = 0.1

    tree = Tree(current_newick)
    _, newick = build_tree_from_splits(
        list(split_lengths.keys()),
        split_lengths,
        tree.n_leaves,
        root_leaf=tree.n_leaves - 1,
        mapping=tree.id_to_name,
    )
    return newick


def _jitter_internal_lengths_newick(current_newick, jitter_scale, min_length=1e-4):
    tree = EteTree(current_newick, format=1)
    internal_nodes = [
        node
        for node in tree.traverse("postorder")
        if not node.is_leaf() and not node.is_root()
    ]
    if not internal_nodes:
        return None

    changed = False
    lower = max(0.0, 1.0 - float(jitter_scale))
    upper = 1.0 + float(jitter_scale)
    for node in internal_nodes:
        dist = float(node.dist)
        if not math.isfinite(dist) or dist <= 0.0:
            continue
        factor = random.uniform(lower, upper)
        new_dist = max(float(min_length), dist * factor)
        if abs(new_dist - dist) > 1e-12:
            node.dist = new_dist
            changed = True

    if not changed:
        return None
    return tree.write(format=1)


def _normalize_tree_like_dataset(tree_newick):
    t_obj = EteTree(tree_newick, format=1)
    leaves = t_obj.get_leaves()
    leaves.sort(key=lambda leaf: leaf.name)

    seq_ordering_map = {}
    for i, leaf in enumerate(leaves):
        original_name = leaf.name
        mapped_name = str(i)
        leaf.name = mapped_name
        seq_ordering_map[original_name] = mapped_name

    return t_obj.write(format=1), seq_ordering_map


def _topology_key(newick_tree, eps_len=1e-8):
    masks, lengths = BHVEncoder().return_BHV_encoding(Tree(newick_tree))
    active_masks = [
        int(mask)
        for mask, length in zip(masks, lengths)
        if length is not None and float(length) > eps_len
    ]
    return tuple(sorted(active_masks))


@functools.lru_cache(maxsize=None)
def _oracle_training_topology_keys(current_newick, target_newick):
    keys = {_topology_key(current_newick)}
    boundary_paths = return_tree_boundary_merge_paths(current_newick, target_newick)
    for boundary_path in boundary_paths:
        keys.add(_topology_key(boundary_path["start_newick"]))
        keys.add(_topology_key(boundary_path["end_newick"]))
        for event in boundary_path["events"]:
            keys.add(_topology_key(event["newick"]))
    return tuple(sorted(keys))


def _remap_tree_with_sequence_ordering(
    tree_newick, seq_ordering_map, offset=0, tree_kind="tree"
):
    t_obj = EteTree(tree_newick, format=1)
    for leaf in t_obj.get_leaves():
        lookup_name = leaf.name
        if offset:
            try:
                lookup_name = str(int(lookup_name) + offset)
            except ValueError as exc:
                raise ValueError(
                    f"Non-integer leaf name '{leaf.name}' encountered in "
                    f"{tree_kind} while applying offset {offset}."
                ) from exc

        mapped_name = seq_ordering_map.get(lookup_name)
        if mapped_name is None:
            raise ValueError(
                f"Leaf name '{lookup_name}' in {tree_kind} not found in "
                "sequence ordering map."
            )
        leaf.name = mapped_name

    return t_obj.write(format=1)


def _choose_wrong_pair_merge_subset(current_newick, target_newick, tokenizer):
    current_tree = Tree(current_newick)
    current_groups = get_structural_polytomy_groups_from_newick(current_newick)
    if not current_groups:
        return None

    n_leaves = current_tree.n_leaves
    shuffled_groups = [tuple(int(component) for component in group) for group in current_groups]
    random.shuffle(shuffled_groups)

    for group_splits in shuffled_groups:
        if len(group_splits) < 3:
            continue

        ready_subsets = {
            tuple(sorted(int(split) for split in subset))
            for subset in _ready_target_merge_subsets_for_group(
                group_splits,
                target_newick,
                n_leaves,
            )
        }

        candidates = []
        for left_idx in range(len(group_splits)):
            for right_idx in range(left_idx + 1, len(group_splits)):
                subset = tuple(
                    sorted(
                        (
                            int(group_splits[left_idx]),
                            int(group_splits[right_idx]),
                        )
                    )
                )
                if subset in ready_subsets:
                    continue
                candidates.append(subset)

        random.shuffle(candidates)
        for subset in candidates:
            perturbed_newick = _apply_merge_subset_to_newick(
                tokenizer,
                current_newick,
                subset,
            )
            if perturbed_newick is None or perturbed_newick == current_newick:
                continue
            return subset

    return None


def _choose_model_wrong_pair_merge_subset(
    module,
    current_newick,
    target_newick,
    current_time,
    phyla_embedding=None,
):
    current_tree = Tree(current_newick)
    current_groups = get_structural_polytomy_groups_from_newick(current_newick)
    if not current_groups:
        return None

    tokenized = _move_tokenized_batch_to_device(
        module.model.tokenizer([current_newick]),
        module.device,
    )
    existing_splits = set(_extract_edge_splits_from_tokenized(tokenized, batch_index=0))
    time_tensor = torch.tensor(
        [float(current_time)],
        dtype=torch.float32,
        device=module.device,
    )

    with torch.no_grad():
        logit_outputs = module.forward(
            tokenized,
            time_tensor,
            phyla_embedding,
            autoregressive=True,
            autoregressive_component_groups=[current_groups],
        )

    if not logit_outputs:
        return None

    n_leaves = current_tree.n_leaves
    ready_subsets_by_group = {}
    for group_splits in current_groups:
        normalized_group = tuple(sorted(int(component) for component in group_splits))
        ready_subsets_by_group[normalized_group] = {
            tuple(sorted(int(split) for split in subset))
            for subset in _ready_target_merge_subsets_for_group(
                normalized_group,
                target_newick,
                n_leaves,
            )
        }

    planned_merges = _plan_autoregressive_boundary_merges(
        logit_outputs,
        existing_splits,
    )
    for planned in planned_merges:
        group_splits = tuple(sorted(int(split) for split in planned["splits_represented"]))
        ready_subsets = ready_subsets_by_group.get(group_splits, set())
        subset, new_split = planned["subsets"][0]
        normalized_subset = tuple(sorted(int(split) for split in subset))
        if normalized_subset in ready_subsets or int(new_split) in existing_splits:
            continue
        return {
            "subset": normalized_subset,
            "new_split": int(new_split),
            "source": "planned_wrong",
        }

    best_candidate = None
    best_rank = None
    for output in logit_outputs:
        group_splits = tuple(sorted(int(split) for split in output["splits_represented"]))
        if len(group_splits) < 3:
            continue

        ready_subsets = ready_subsets_by_group.get(group_splits, set())
        polytomy_score = float(output["polytomy_pred"].detach().cpu().item())
        logits = output["logits"].detach()
        splits = [int(split) for split in output["splits_represented"]]

        for left_idx in range(len(splits)):
            for right_idx in range(left_idx + 1, len(splits)):
                subset = tuple(sorted((int(splits[left_idx]), int(splits[right_idx]))))
                if subset in ready_subsets:
                    continue

                new_split = int(subset[0]) | int(subset[1])
                if new_split in existing_splits:
                    continue

                score = float(logits[left_idx, right_idx].item())
                if not math.isfinite(score):
                    continue

                rank = (score, polytomy_score)
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_candidate = {
                        "subset": subset,
                        "new_split": new_split,
                        "source": "top_wrong_pair",
                    }

    return best_candidate


def _plan_first_autoregressive_model_merge(
    module,
    current_newick,
    current_time,
    phyla_embedding=None,
):
    tokenized = _move_tokenized_batch_to_device(
        module.model.tokenizer([current_newick]),
        module.device,
    )
    groups = [get_structural_polytomy_groups_from_newick(current_newick)]
    if not groups[0]:
        return None

    time_tensor = module._effective_autoregressive_time_tensor(current_time)

    with torch.no_grad():
        logit_outputs = module.forward(
            tokenized,
            time_tensor,
            phyla_embedding,
            autoregressive=True,
            autoregressive_component_groups=groups,
        )

    planned_merges = _plan_autoregressive_boundary_merges(
        logit_outputs,
        _extract_edge_splits_from_tokenized(tokenized, batch_index=0),
    )
    if not planned_merges:
        return None

    subset, new_split = planned_merges[0]["subsets"][0]
    return {
        "subset": tuple(int(split) for split in subset),
        "new_split": int(new_split),
    }


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
        training_sampling_mode: str = "batch_compare",
        training_sampling_dt_base: float = 0.02,
        num_samples: int = 10,
        dt: float = 0.1,
        # Figure out how to do typing here
        global_splits=None,
        random_trees=None,
        verbose: bool = False,
        phyla_checkpoint_path=None,
        velocity_loss_mode: str = "weighted",
        velocity_loss_plain_weight: float = 0.5,
        velocity_sign_eps: float = 1e-3,
        training_step_velocity_weight: float = 1.0,
        training_step_autoregressive_weight: float = 1.0,
        training_step_autoregressive_grad_ratio = None,
        autoregressive_use_time: bool = False,
        autoregressive_target_mode: str = "scheduled",
        autoregressive_rollin_prob: float = 0.0,
        autoregressive_dagger_prob: float = 0.0,
        autoregressive_dagger_max_steps: int = 4,
        autoregressive_structure_perturb_prob: float = 0.0,
        autoregressive_structure_perturb_mode: str = "random_wrong_pair",
        velocity_length_jitter_prob: float = 0.0,
        velocity_length_jitter_scale: float = 0.0,
        velocity_dt_candidate_weight: float = 0.0,
        velocity_dt_hit_weight: float = 0.0,
        velocity_dt_eps: float = 1e-6,
        velocity_event_weight: float = 0.5,
        velocity_event_temp: float = 0.5,
        velocity_event_rate_beta: float = 5.0,
        velocity_event_normalize_by_log_candidates: bool = True,
        sample_metrics_trace_path: str | None = None,
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
        self.training_sampling_mode = str(training_sampling_mode)
        self.training_sampling_dt_base = float(training_sampling_dt_base)
        self.num_samples = num_samples
        self.dt = dt
        self.train_tokenized_trees = None
        self.train_batched_time = None
        self.train_tree = None
        self._cached_harness_sampling_pairs = {}

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

        valid_velocity_loss_modes = {"plain", "weighted", "blended"}
        if velocity_loss_mode not in valid_velocity_loss_modes:
            raise ValueError(
                f"Invalid velocity_loss_mode={velocity_loss_mode!r}. "
                f"Expected one of {sorted(valid_velocity_loss_modes)}."
            )
        if not (0.0 <= float(velocity_loss_plain_weight) <= 1.0):
            raise ValueError(
                "velocity_loss_plain_weight must be in [0, 1], "
                f"got {velocity_loss_plain_weight}."
            )
        if float(velocity_sign_eps) < 0.0:
            raise ValueError(
                f"velocity_sign_eps must be non-negative, got {velocity_sign_eps}."
            )
        if float(training_step_velocity_weight) < 0.0:
            raise ValueError(
                "training_step_velocity_weight must be non-negative, "
                f"got {training_step_velocity_weight}."
            )
        if float(training_step_autoregressive_weight) < 0.0:
            raise ValueError(
                "training_step_autoregressive_weight must be non-negative, "
                f"got {training_step_autoregressive_weight}."
            )
        if (
            training_step_autoregressive_grad_ratio is not None
            and float(training_step_autoregressive_grad_ratio) < 0.0
        ):
            raise ValueError(
                "training_step_autoregressive_grad_ratio must be non-negative "
                f"or None, got {training_step_autoregressive_grad_ratio}."
            )
        valid_autoregressive_target_modes = {"scheduled", "ready_alternatives"}
        if autoregressive_target_mode not in valid_autoregressive_target_modes:
            raise ValueError(
                f"Invalid autoregressive_target_mode={autoregressive_target_mode!r}. "
                f"Expected one of {sorted(valid_autoregressive_target_modes)}."
            )
        valid_training_sampling_modes = {"batch_compare", "harness_sanity"}
        if self.training_sampling_mode not in valid_training_sampling_modes:
            raise ValueError(
                f"Invalid training_sampling_mode={self.training_sampling_mode!r}. "
                f"Expected one of {sorted(valid_training_sampling_modes)}."
            )
        if self.training_sampling_dt_base <= 0.0:
            raise ValueError(
                "training_sampling_dt_base must be > 0, "
                f"got {training_sampling_dt_base}."
            )
        valid_structure_perturb_modes = {"random_wrong_pair", "model_wrong_pair"}
        if (
            autoregressive_structure_perturb_mode
            not in valid_structure_perturb_modes
        ):
            raise ValueError(
                "Invalid autoregressive_structure_perturb_mode="
                f"{autoregressive_structure_perturb_mode!r}. Expected one of "
                f"{sorted(valid_structure_perturb_modes)}."
            )
        if not (0.0 <= float(autoregressive_rollin_prob) <= 1.0):
            raise ValueError(
                "autoregressive_rollin_prob must be in [0, 1], "
                f"got {autoregressive_rollin_prob}."
            )
        if not (0.0 <= float(autoregressive_dagger_prob) <= 1.0):
            raise ValueError(
                "autoregressive_dagger_prob must be in [0, 1], "
                f"got {autoregressive_dagger_prob}."
            )
        if int(autoregressive_dagger_max_steps) < 1:
            raise ValueError(
                "autoregressive_dagger_max_steps must be >= 1, "
                f"got {autoregressive_dagger_max_steps}."
            )
        if not (0.0 <= float(autoregressive_structure_perturb_prob) <= 1.0):
            raise ValueError(
                "autoregressive_structure_perturb_prob must be in [0, 1], "
                f"got {autoregressive_structure_perturb_prob}."
            )
        if not (0.0 <= float(velocity_length_jitter_prob) <= 1.0):
            raise ValueError(
                "velocity_length_jitter_prob must be in [0, 1], "
                f"got {velocity_length_jitter_prob}."
            )
        if float(velocity_length_jitter_scale) < 0.0:
            raise ValueError(
                "velocity_length_jitter_scale must be non-negative, "
                f"got {velocity_length_jitter_scale}."
            )
        if float(velocity_dt_candidate_weight) < 0.0:
            raise ValueError(
                "velocity_dt_candidate_weight must be non-negative, "
                f"got {velocity_dt_candidate_weight}."
            )
        if float(velocity_dt_hit_weight) < 0.0:
            raise ValueError(
                "velocity_dt_hit_weight must be non-negative, "
                f"got {velocity_dt_hit_weight}."
            )
        if float(velocity_dt_eps) <= 0.0:
            raise ValueError(
                f"velocity_dt_eps must be > 0, got {velocity_dt_eps}."
            )
        if float(velocity_event_weight) < 0.0:
            raise ValueError(
                "velocity_event_weight must be non-negative, "
                f"got {velocity_event_weight}."
            )
        if float(velocity_event_temp) <= 0.0:
            raise ValueError(
                f"velocity_event_temp must be > 0, got {velocity_event_temp}."
            )
        if float(velocity_event_rate_beta) <= 0.0:
            raise ValueError(
                f"velocity_event_rate_beta must be > 0, got {velocity_event_rate_beta}."
            )
        self.velocity_loss_mode = velocity_loss_mode
        self.velocity_loss_plain_weight = float(velocity_loss_plain_weight)
        self.velocity_sign_eps = float(velocity_sign_eps)
        self.training_step_velocity_weight = float(training_step_velocity_weight)
        self.training_step_autoregressive_weight = float(
            training_step_autoregressive_weight
        )
        if training_step_autoregressive_grad_ratio is None:
            self.training_step_autoregressive_grad_ratio = None
        else:
            self.training_step_autoregressive_grad_ratio = float(
                training_step_autoregressive_grad_ratio
            )
        self.autoregressive_use_time = bool(autoregressive_use_time)
        self.autoregressive_target_mode = str(autoregressive_target_mode)
        self.autoregressive_rollin_prob = float(autoregressive_rollin_prob)
        self.autoregressive_dagger_prob = float(autoregressive_dagger_prob)
        self.autoregressive_dagger_max_steps = int(autoregressive_dagger_max_steps)
        self.autoregressive_structure_perturb_prob = float(
            autoregressive_structure_perturb_prob
        )
        self.autoregressive_structure_perturb_mode = str(
            autoregressive_structure_perturb_mode
        )
        self.velocity_length_jitter_prob = float(velocity_length_jitter_prob)
        self.velocity_length_jitter_scale = float(velocity_length_jitter_scale)
        self.velocity_dt_candidate_weight = float(velocity_dt_candidate_weight)
        self.velocity_dt_hit_weight = float(velocity_dt_hit_weight)
        self.velocity_dt_eps = float(velocity_dt_eps)
        self.velocity_event_weight = float(velocity_event_weight)
        self.velocity_event_temp = float(velocity_event_temp)
        self.velocity_event_rate_beta = float(velocity_event_rate_beta)
        self.velocity_event_normalize_by_log_candidates = bool(
            velocity_event_normalize_by_log_candidates
        )
        self.sample_metrics_trace_path = (
            str(sample_metrics_trace_path).strip()
            if sample_metrics_trace_path
            else None
        )

        phyla_config_path = "configs/sample_eval_config.yaml"

        if self.phyla_checkpoint_path is not None:
            original_argv = sys.argv
            sys.argv = ["script", phyla_config_path]
            try:
                if not os.path.exists(phyla_config_path):
                    logging.warning(
                        f"Phyla configuration file not found at {phyla_config_path}"
                    )

                load_config, Config, load_model, _ = _load_phyla_runtime()
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

    def _effective_autoregressive_time_value(self, time_value):
        if not self.autoregressive_use_time:
            return 0.0
        return float(time_value)

    def _effective_autoregressive_time_tensor(self, time_value):
        if torch.is_tensor(time_value):
            if self.autoregressive_use_time:
                return time_value
            return torch.zeros_like(time_value, dtype=torch.float32, device=time_value.device)
        return torch.tensor(
            [self._effective_autoregressive_time_value(time_value)],
            dtype=torch.float32,
            device=self.device,
        )

    def _rollin_single_autoregressive_state(
        self,
        current_newick,
        target_newick,
        current_time,
        phyla_embedding=None,
    ):
        planned_merge = _plan_first_autoregressive_model_merge(
            self,
            current_newick=current_newick,
            current_time=current_time,
            phyla_embedding=phyla_embedding,
        )
        if planned_merge is None:
            return None

        rolled_newick = _apply_merge_subset_to_newick(
            self.model.tokenizer,
            current_newick,
            planned_merge["subset"],
            new_split=planned_merge["new_split"],
        )
        if rolled_newick is None:
            return None

        corrective_events = return_sampled_tree_boundary_decisions(
            rolled_newick,
            target_newick,
        )
        if not corrective_events:
            return None

        next_event = corrective_events[0]
        return {
            "newick": rolled_newick,
            "labels": next_event["labels"],
            "time": self._effective_autoregressive_time_value(current_time),
        }

    def _dagger_rollin_single_autoregressive_state(
        self,
        current_newick,
        target_newick,
        current_time,
        phyla_embedding=None,
    ):
        oracle_training_topologies = set(
            _oracle_training_topology_keys(current_newick, target_newick)
        )
        state_newick = current_newick
        state_time = float(current_time)

        for rollout_step in range(self.autoregressive_dagger_max_steps):
            planned_merge = _plan_first_autoregressive_model_merge(
                self,
                current_newick=state_newick,
                current_time=state_time,
                phyla_embedding=phyla_embedding,
            )
            if planned_merge is None:
                return None

            next_newick = _apply_merge_subset_to_newick(
                self.model.tokenizer,
                state_newick,
                planned_merge["subset"],
                new_split=planned_merge["new_split"],
            )
            if next_newick is None or next_newick == state_newick:
                return None

            if _topology_key(next_newick) not in oracle_training_topologies:
                corrective_events = return_sampled_tree_boundary_decisions(
                    next_newick,
                    target_newick,
                )
                if not corrective_events:
                    return None

                next_event = corrective_events[0]
                return {
                    "newick": next_newick,
                    "labels": next_event["labels"],
                    "time": self._effective_autoregressive_time_value(state_time),
                    "rollout_steps": rollout_step + 1,
                }

            state_newick = next_newick

        return None

    def _perturb_autoregressive_single_state(
        self,
        current_newick,
        target_newick,
        current_time,
        phyla_embedding=None,
    ):
        if self.autoregressive_structure_perturb_mode == "model_wrong_pair":
            chosen_merge = _choose_model_wrong_pair_merge_subset(
                self,
                current_newick=current_newick,
                target_newick=target_newick,
                current_time=current_time,
                phyla_embedding=phyla_embedding,
            )
        else:
            subset = _choose_wrong_pair_merge_subset(
                current_newick,
                target_newick,
                self.model.tokenizer,
            )
            chosen_merge = None if subset is None else {"subset": subset, "new_split": None}

        if chosen_merge is None:
            return None

        perturbed_newick = _apply_merge_subset_to_newick(
            self.model.tokenizer,
            current_newick,
            chosen_merge["subset"],
            new_split=chosen_merge.get("new_split"),
        )
        if perturbed_newick is None:
            return None

        corrective_events = return_sampled_tree_boundary_decisions(
            perturbed_newick,
            target_newick,
        )
        if not corrective_events:
            return None

        next_event = corrective_events[0]
        return {
            "newick": perturbed_newick,
            "labels": next_event["labels"],
            "time": self._effective_autoregressive_time_value(current_time),
        }

    def _prepare_velocity_training_batch(self, batch):
        if (
            self.velocity_length_jitter_prob <= 0.0
            or self.velocity_length_jitter_scale <= 0.0
            or "original_trees" not in batch
            or "target_trees" not in batch
        ):
            return batch, {"attempted": 0.0, "applied": 0.0}

        newicks = list(batch["original_trees"])
        velocity_labels = list(batch["batched_velocity"])
        batched_time = batch.get("batched_time")
        attempted = 0
        applied = 0

        for batch_index, (current_newick, target_newick) in enumerate(
            zip(newicks, batch["target_trees"])
        ):
            if random.random() > self.velocity_length_jitter_prob:
                continue
            attempted += 1

            perturbed_newick = _jitter_internal_lengths_newick(
                current_newick,
                self.velocity_length_jitter_scale,
            )
            if perturbed_newick is None:
                continue

            try:
                time_point = (
                    float(batched_time[batch_index].item())
                    if batched_time is not None
                    else 0.0
                )
                _, perturbed_velocity = return_sampled_tree_orthant_velocity(
                    perturbed_newick,
                    target_newick,
                    time_point,
                )
            except Exception:
                continue

            newicks[batch_index] = perturbed_newick
            velocity_labels[batch_index] = perturbed_velocity
            applied += 1

        if applied == 0:
            return batch, {"attempted": float(attempted), "applied": 0.0}

        updated_batch = dict(batch)
        updated_batch["original_trees"] = newicks
        updated_batch["batched_velocity"] = velocity_labels
        updated_batch["tokenized_trees"] = _move_tokenized_batch_to_device(
            self.model.tokenizer(newicks),
            self.device,
        )
        return updated_batch, {"attempted": float(attempted), "applied": float(applied)}

    def _prepare_autoregressive_training_batch(self, batch):
        if (
            self.autoregressive_rollin_prob <= 0.0
            and self.autoregressive_dagger_prob <= 0.0
            and self.autoregressive_structure_perturb_prob <= 0.0
            or "target_trees" not in batch
            or "newick_autoregressive_trees" not in batch
        ):
            return batch, {
                "rollin_attempted": 0.0,
                "rollin_applied": 0.0,
                "dagger_attempted": 0.0,
                "dagger_applied": 0.0,
                "dagger_rollout_steps": 0.0,
                "structure_perturb_attempted": 0.0,
                "structure_perturb_applied": 0.0,
            }

        newicks = list(batch["newick_autoregressive_trees"])
        labels = list(batch["batched_autoregressive_labels"])
        times = batch["batched_autoregressive_time"].detach().clone()

        rollin_attempted = 0
        rollin_applied = 0
        dagger_attempted = 0
        dagger_applied = 0
        dagger_rollout_steps = 0
        structure_attempted = 0
        structure_applied = 0
        for batch_index, (current_newick, target_newick) in enumerate(
            zip(newicks, batch["target_trees"])
        ):
            if self.autoregressive_rollin_prob > 0.0:
                if random.random() <= self.autoregressive_rollin_prob:
                    rollin_attempted += 1

                    phyla_embedding = None
                    if batch["phyla_embeddings"] is not None:
                        phyla_embedding = batch["phyla_embeddings"][batch_index : batch_index + 1]

                    rolled = self._rollin_single_autoregressive_state(
                        current_newick=current_newick,
                        target_newick=target_newick,
                        current_time=float(times[batch_index].item()),
                        phyla_embedding=phyla_embedding,
                    )
                    if rolled is not None:
                        current_newick = rolled["newick"]
                        newicks[batch_index] = rolled["newick"]
                        labels[batch_index] = rolled["labels"]
                        times[batch_index] = float(rolled["time"])
                        rollin_applied += 1

            if self.autoregressive_dagger_prob > 0.0:
                if random.random() <= self.autoregressive_dagger_prob:
                    dagger_attempted += 1

                    phyla_embedding = None
                    if batch["phyla_embeddings"] is not None:
                        phyla_embedding = batch["phyla_embeddings"][
                            batch_index : batch_index + 1
                        ]

                    dagger = self._dagger_rollin_single_autoregressive_state(
                        current_newick=current_newick,
                        target_newick=target_newick,
                        current_time=float(times[batch_index].item()),
                        phyla_embedding=phyla_embedding,
                    )
                    if dagger is not None:
                        current_newick = dagger["newick"]
                        newicks[batch_index] = dagger["newick"]
                        labels[batch_index] = dagger["labels"]
                        times[batch_index] = float(dagger["time"])
                        dagger_applied += 1
                        dagger_rollout_steps += int(dagger["rollout_steps"])

            if self.autoregressive_structure_perturb_prob > 0.0:
                if random.random() <= self.autoregressive_structure_perturb_prob:
                    structure_attempted += 1
                    phyla_embedding = None
                    if batch["phyla_embeddings"] is not None:
                        phyla_embedding = batch["phyla_embeddings"][batch_index : batch_index + 1]
                    perturbed = self._perturb_autoregressive_single_state(
                        current_newick=current_newick,
                        target_newick=target_newick,
                        current_time=float(times[batch_index].item()),
                        phyla_embedding=phyla_embedding,
                    )
                    if perturbed is not None:
                        newicks[batch_index] = perturbed["newick"]
                        labels[batch_index] = perturbed["labels"]
                        times[batch_index] = float(perturbed["time"])
                        structure_applied += 1

        if rollin_applied == 0 and dagger_applied == 0 and structure_applied == 0:
            return batch, {
                "rollin_attempted": float(rollin_attempted),
                "rollin_applied": 0.0,
                "dagger_attempted": float(dagger_attempted),
                "dagger_applied": 0.0,
                "dagger_rollout_steps": float(dagger_rollout_steps),
                "structure_perturb_attempted": float(structure_attempted),
                "structure_perturb_applied": 0.0,
            }

        updated_batch = dict(batch)
        updated_batch["newick_autoregressive_trees"] = newicks
        updated_batch["batched_autoregressive_labels"] = labels
        updated_batch["batched_autoregressive_time"] = times
        updated_batch["tokenized_autoregressive_trees"] = _move_tokenized_batch_to_device(
            self.model.tokenizer(newicks),
            self.device,
        )
        return updated_batch, {
            "rollin_attempted": float(rollin_attempted),
            "rollin_applied": float(rollin_applied),
            "dagger_attempted": float(dagger_attempted),
            "dagger_applied": float(dagger_applied),
            "dagger_rollout_steps": float(dagger_rollout_steps),
            "structure_perturb_attempted": float(structure_attempted),
            "structure_perturb_applied": float(structure_applied),
        }

    def _append_sample_metrics_trace(self, metrics):
        if not self.sample_metrics_trace_path:
            return

        payload = {
            "global_step": int(self.global_step),
            "stepper": int(self.stepper),
            "timestamp": time.time(),
        }
        for key, value in metrics.items():
            if torch.is_tensor(value):
                payload[key] = float(value.detach().cpu().item())
            elif isinstance(value, np.generic):
                payload[key] = float(value)
            elif isinstance(value, (int, float, bool)):
                payload[key] = float(value) if isinstance(value, bool) else value
            else:
                payload[key] = value

        os.makedirs(os.path.dirname(self.sample_metrics_trace_path), exist_ok=True)
        with open(self.sample_metrics_trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _get_harness_sampling_pair(self, train=True):
        cache_key = "train" if train else "val"
        cached = self._cached_harness_sampling_pairs.get(cache_key)
        if cached is not None:
            return cached

        dataset_split = self.dataset.dataset_train if train else self.dataset.dataset_val
        random_state = random.getstate()
        try:
            random.seed(13)
            real_tree_raw = dataset_split.return_posterior_trees(0)[0]
            target_tree, seq_ordering_map = _normalize_tree_like_dataset(real_tree_raw)
            random_tree_raw = dataset_split.sample_random_tree(real_tree_raw)
        finally:
            random.setstate(random_state)

        start_tree = _remap_tree_with_sequence_ordering(
            random_tree_raw,
            seq_ordering_map,
            offset=0,
            tree_kind="start tree",
        )
        pair = {
            "start_tree": start_tree,
            "target_tree": target_tree,
            "n_leaves": len(EteTree(target_tree, format=1).get_leaves()),
        }
        self._cached_harness_sampling_pairs[cache_key] = pair
        return pair

    def sample_compare_harness(self, train=True):
        pair = self._get_harness_sampling_pair(train=train)
        phyla_dim = int(self.model.phyla_proj.in_features)
        phyla_embeddings = torch.zeros(
            (1, pair["n_leaves"], phyla_dim),
            dtype=torch.float32,
            device=self.device,
        )

        sampled_trees, _, _, _, _ = self.sample(
            [pair["start_tree"]],
            phyla_embeddings=phyla_embeddings,
            num_samples=1,
            T=1.0,
            dt_base=self.training_sampling_dt_base,
            max_steps=256,
            max_events=1024,
        )
        sampled_tree = sampled_trees[0]
        return {
            "rf_norm": float(calculate_norm_rf(sampled_tree, pair["target_tree"])),
            "start_rf_norm": float(
                calculate_norm_rf(pair["start_tree"], pair["target_tree"])
            ),
        }

    def compute_phyla_embeddings(self, sequences, names, device="cuda"):
        """
        Generates Phyla embeddings for a batch of sequences.
        """
        if self.phyla_model is None:
            raise ValueError("Phyla model not loaded.")

        # This utility handles tokenization, padding, and CLS token placement
        _, _, _, _encode_sequences_openfold_style = _load_phyla_runtime()
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
        self,
        batched_tokenized_trees,
        t,
        phyla_embeddings,
        autoregressive=False,
        autoregressive_component_groups=None,
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
            model_kwargs = dict(
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=False,
                return_edges_only=True,
                autoregressive=True,
            )
            model_signature = inspect.signature(self.model.forward)
            supports_component_groups = (
                "autoregressive_component_groups" in model_signature.parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in model_signature.parameters.values()
                )
            )
            if (
                autoregressive_component_groups is not None
                and supports_component_groups
            ):
                model_kwargs["autoregressive_component_groups"] = (
                    autoregressive_component_groups
                )

            all_group_logits = self.model(
                batched_tokenized_trees,
                t,
                **model_kwargs,
            )
            return all_group_logits

    def step(self, batch, eval=False, autoregressive=False):
        logs = {}
        if not eval and not autoregressive:
            self.current_step_value += 1
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
            batch, velocity_perturb_stats = self._prepare_velocity_training_batch(batch)
            v_pred, edge_split_masks, edge_mask = self.forward(
                batch["tokenized_trees"],
                batch["batched_time"],
                batch["phyla_embeddings"],
            )

            enc = BHVEncoder()

            if self.train_tokenized_trees is None:
                self.train_tokenized_trees = batch["tokenized_trees"]
                self.train_batched_time = batch["batched_time"]
                self.train_tree = batch["original_trees"]
            # else:
            #     if calculate_norm_rf(batch['original_trees'][0], self.train_tree[0]) != 0:
            #         raise Exception("Training tree topology changed during training!")
            #     elif not torch.equal(batch["tokenized_trees"][0], self.train_tokenized_trees[0]):
            #         import pdb; pdb.set_trace()
            #         raise Exception("Training tokenized trees changed during training!")

            velocity_labels = batch["batched_velocity"]
            num_leaves = batch["num_leaves"]
            gathered_velocity_labels = []
            v_pred_indices = []
            gathered_velocity_lengths = []

            for num in range(len(velocity_labels)):
                sub_gathered_velocity_labels = []
                sub_v_pred_indices = []
                sub_gathered_velocity_lengths = []

                num_leave = int(num_leaves[num])
                split_masks_num = [int(m) for m in edge_split_masks[num]]
                split_masks_nonzero = [m for m in split_masks_num if m != 0]
                if len(split_masks_nonzero) == 0:
                    gathered_velocity_labels.append(
                        torch.tensor(sub_gathered_velocity_labels)
                    )
                    v_pred_indices.append(torch.tensor(sub_v_pred_indices))
                    gathered_velocity_lengths.append(
                        torch.tensor(sub_gathered_velocity_lengths)
                    )
                    continue

                real_max_bit = max(m.bit_length() for m in split_masks_nonzero)
                full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0
                mask_to_idx = {m: i for i, m in enumerate(split_masks_num)}

                tree_obj = Tree(batch["original_trees"][num])
                tree_masks, tree_lengths = enc.return_BHV_encoding(tree_obj)
                length_map = {
                    int(m): float(l)
                    for m, l in zip(tree_masks, tree_lengths)
                    if l is not None
                }
                for vel in velocity_labels[num]:
                    original_vel = vel
                    if vel.bit_length() == real_max_bit + 1:
                        vel = remove_bit(vel, num_leave - 1)
                    elif vel.bit_length() > real_max_bit + 1:
                        raise Exception(
                            f"Whoa there is a big problem with this split mask {vel} vs real max {real_max_bit}!"
                        )

                    matched_vel = vel
                    if matched_vel not in mask_to_idx:
                        # Split orientation can flip after dummy-root removal; allow complement match.
                        complement_vel = full_mask ^ vel
                        if complement_vel in mask_to_idx:
                            matched_vel = complement_vel
                        else:
                            print(
                                f"This split {vel} from velocity labels is not in edge splits {split_masks_num}!"
                            )
                            print([i for i in range(vel.bit_length()) if (vel >> i) & 1])
                            raise Exception("Split not found in edge splits")
                    
                    #Ignore leaf edges
                    n_bits = real_max_bit
                    k = int(matched_vel).bit_count()
                    is_pendant = min(k, n_bits - k) == 1
                    if is_pendant:
                        continue

                    edge_len = length_map.get(int(matched_vel))
                    if edge_len is None and full_mask:
                        edge_len = length_map.get(full_mask ^ int(matched_vel))
                    if edge_len is None:
                        print(
                            f"Edge length not found for split {matched_vel} (original {original_vel}) in tree {batch['original_trees'][num]}"
                        )
                        print(f"Available splits: {split_masks_num}")
                        print(f"Length map keys: {list(length_map.keys())}")
                        raise Exception("Edge length not found for matched split")
                    if edge_len is None or float(edge_len) <= 1e-8:
                        continue

                    sub_gathered_velocity_labels.append(velocity_labels[num][original_vel])
                    sub_v_pred_indices.append(mask_to_idx[int(matched_vel)])
                    sub_gathered_velocity_lengths.append(float(edge_len))


                gathered_velocity_labels.append(
                    torch.tensor(sub_gathered_velocity_labels)
                )
                v_pred_indices.append(torch.tensor(sub_v_pred_indices))
                gathered_velocity_lengths.append(torch.tensor(sub_gathered_velocity_lengths))

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
                gathered_velocity_lengths_flat = torch.cat(gathered_velocity_lengths).to(
                    v_pred_gathered.device
                )

                y = gathered_velocity_labels_flat
                p = v_pred_gathered
                lengths = gathered_velocity_lengths_flat

                # --- Velocity diagnostics ---
                with torch.no_grad():
                    vel_metrics = _velocity_diagnostics(
                        p,
                        y,
                        topk=3,
                        sign_eps=self.velocity_sign_eps,
                        lengths=gathered_velocity_lengths_flat,
                    )
                if self.verbose:
                    logger.info(
                        f"Velocity metrics: MSE={vel_metrics['mse']:.6f}  "
                        f"Cosine={vel_metrics['cosine']:.4f}  "
                        f"Pearson={vel_metrics['pearson']:.4f}  "
                        f"Spearman={vel_metrics['spearman']:.4f}  "
                        f"SignAcc={vel_metrics['sign_acc']:.4f}  "
                        f"TopK={vel_metrics['topk_overlap']:.4f}  "
                        f"dtTopK={vel_metrics['dt_topk_overlap']:.4f}  "
                        f"dtFirstHitRecall={vel_metrics['dt_first_hit_recall']:.4f}  "
                        f"dtHitRelErr={vel_metrics['dt_hit_rel_err']:.4f}  "
                        f"N={vel_metrics['n_edges']}"
                    )
                logs.update(
                    {
                        "velocity/cosine": torch.tensor(
                            vel_metrics["cosine"], device=v_pred.device
                        ),
                        "velocity/pearson": torch.tensor(
                            vel_metrics["pearson"], device=v_pred.device
                        ),
                        "velocity/spearman": torch.tensor(
                            vel_metrics["spearman"], device=v_pred.device
                        ),
                        "velocity/sign_acc": torch.tensor(
                            vel_metrics["sign_acc"], device=v_pred.device
                        ),
                        "velocity/topk_overlap": torch.tensor(
                            vel_metrics["topk_overlap"], device=v_pred.device
                        ),
                        "velocity/dt_topk_overlap": torch.tensor(
                            vel_metrics["dt_topk_overlap"], device=v_pred.device
                        ),
                        "velocity/dt_first_hit_recall": torch.tensor(
                            vel_metrics["dt_first_hit_recall"], device=v_pred.device
                        ),
                        "velocity/dt_first_hit_precision": torch.tensor(
                            vel_metrics["dt_first_hit_precision"], device=v_pred.device
                        ),
                        "velocity/dt_neg_jaccard": torch.tensor(
                            vel_metrics["dt_neg_jaccard"], device=v_pred.device
                        ),
                        "velocity/length_jitter_attempted": torch.tensor(
                            velocity_perturb_stats["attempted"],
                            device=v_pred.device,
                        ),
                        "velocity/length_jitter_applied": torch.tensor(
                            velocity_perturb_stats["applied"],
                            device=v_pred.device,
                        ),
                    }
                )

                # eps = 1e-6
                # first_hit_tol = 0.01
                # contract = (y < -self.velocity_sign_eps) & (lengths > 1e-8)

                # Lc = lengths[contract].clamp_min(eps)
                # yc = y[contract]
                # pc = p[contract]

                # tau_true = Lc / (-yc).clamp_min(eps)
                # w = (tau_true.median().clamp_min(eps) / tau_true).clamp(max=20.0)

                # tau_min = tau_true.min()
                # first = (tau_true - tau_min).abs() <= 0.01  # true tol (could be 0)

                # boost = 5.0
                # w = w * (1.0 + boost * first.float())

                # loss = (w * (pc - yc)**2).mean()
                # import pdb; pdb.set_trace()

                # tau_true = lengths[contract] / (-y[contract]).clamp_min(eps)
                # tau_min = tau_true.min()
                # first = torch.abs(tau_true - tau_min) <= first_hit_tol
                # w = torch.ones_like(y[contract])
                # alpha = 10
                # w[first] = 1.0 + alpha
                # loss = (w * (p[contract] - y[contract])**2).mean()

                # ####OG LOSS HERE
                residual_sq = (p - y).pow(2)
                plain_mse = residual_sq.mean()

                abs_y = y.abs()
                eps = 1e-6
                scale = abs_y.median().clamp_min(eps)  # robust scale
                w = (abs_y / scale).clamp(min=0.0, max=20.0)
                weighted_mse = (w * residual_sq).sum() / w.sum().clamp_min(eps)
                if self.velocity_loss_mode == "plain":
                    loss = plain_mse
                elif self.velocity_loss_mode == "weighted":
                    loss = weighted_mse
                else:
                    loss = (
                        self.velocity_loss_plain_weight * plain_mse
                        + (1.0 - self.velocity_loss_plain_weight) * weighted_mse
                    )

                # ------------------------------------------------------------------
                # First-hit structured loss on true contracting edges:
                #   1) keep the fastest true contracting edges accurate in rate space
                #   2) ensure true first-hit edges remain earlier than later edges
                #   3) keep the tied first-hit set collapsed together
                # ------------------------------------------------------------------
                # contract_mask = (y < -self.velocity_sign_eps) & (lengths > 1e-8)
                # fast_rate_loss = p.new_tensor(0.0)
                # first_hit_dt_loss = p.new_tensor(0.0)

                # if int(contract_mask.sum()) > 0:
                #     Lc = lengths[contract_mask].clamp_min(eps)
                #     yc = y[contract_mask]
                #     pc = p[contract_mask]

                #     # First-hit ordering is governed by contraction rate (-v / length).
                #     rate_true = (-yc).clamp_min(eps) / Lc
                #     rate_pred = F.softplus(
                #         -pc, beta=self.velocity_event_rate_beta
                #     ) / Lc

                #     # True collapse times for truly contracting edges
                #     tau_true = 1.0 / rate_true.clamp_min(eps)
                #     tau_pred = 1.0 / rate_pred.clamp_min(eps)

                #     # Identify the true first-hit set with tolerance
                #     tau_true_min = tau_true.min()
                #     first_mask = torch.abs(tau_true - tau_true_min) <= 0.01
                #     later_mask = ~first_mask
                #     fast_k = min(8, int(tau_true.numel()))
                #     fast_idx = torch.argsort(tau_true)[:fast_k]
                #     fast_mask = torch.zeros_like(first_mask)
                #     fast_mask[fast_idx] = True
                #     rate_scale = rate_true[fast_mask].median().clamp_min(1.0)
                #     fast_rate_loss = F.smooth_l1_loss(
                #         rate_pred[fast_mask] / rate_scale,
                #         rate_true[fast_mask] / rate_scale,
                #     )

                #     z_pred = torch.log(tau_pred.clamp_min(eps))

                #     # 1) Tie loss: first-hit edges should have similar predicted dt
                #     if int(first_mask.sum()) > 1:
                #         z_first = z_pred[first_mask]
                #         first_hit_tie_loss = ((z_first - z_first.mean()) ** 2).mean()
                #     else:
                #         first_hit_tie_loss = p.new_tensor(0.0)
                #     if int(first_mask.sum()) > 0:
                #         first_hit_dt_loss = F.smooth_l1_loss(
                #             z_pred[first_mask],
                #             torch.log(tau_true[first_mask].clamp_min(eps)),
                #         )

                #     # 2) Rank loss: first-hit edges should be earlier than later contracting edges
                #     if int(first_mask.sum()) > 0 and int(later_mask.sum()) > 0:
                #         z_first = z_pred[first_mask][:, None]   # shape [F, 1]
                #         z_later = z_pred[later_mask][None, :]   # shape [1, L]
                #         first_hit_rank_loss = F.relu(
                #             z_later - z_first + 0.02
                #         ).mean()
                #     else:
                #         first_hit_rank_loss = p.new_tensor(0.0)

                #     first_hit_loss = (
                #         first_hit_tie_loss
                #         + first_hit_rank_loss
                #     )

                #     n_contract = int(contract_mask.sum())
                #     n_first = int(first_mask.sum())
                #     n_later = int(later_mask.sum())
                # else:
                #     first_hit_tie_loss = p.new_tensor(0.0)
                #     first_hit_rank_loss = p.new_tensor(0.0)
                #     first_hit_loss = p.new_tensor(0.0)
                #     n_contract = 0
                #     n_first = 0
                #     n_later = 0

                # first_hit_aux_weight = float(
                #     min(max((self.current_step_value - 100) / 200.0, 0.0), 1.0)
                # )

                first_hit_velocity_loss = p.new_tensor(0.0)
                contract_mask = (y < -self.velocity_sign_eps) & (lengths > 1e-8)
                if self.velocity_dt_hit_weight > 0.0 and int(contract_mask.sum()) > 0:
                    Lc = lengths[contract_mask].clamp_min(eps)
                    yc = y[contract_mask]
                    pc = p[contract_mask]
                    tau_true = Lc / (-yc).clamp_min(eps)
                    tau_true_min = tau_true.min()
                    first_mask = torch.abs(tau_true - tau_true_min) <= 0.01
                    if int(first_mask.sum()) > 0:
                        first_hit_velocity_loss = F.smooth_l1_loss(
                            pc[first_mask], yc[first_mask]
                        )

                # first_hit_aux_weight = float(
                #     min(max((self.current_step_value - 100) / 200.0, 0.0), 1.0)
                # )

                # loss = (
                #     loss
                #     + 1
                #     * (0.5 * self.velocity_dt_hit_weight * first_hit_velocity_loss)
                # )

                # loss = loss

                # loss = (
                #     loss
                #     + self.velocity_dt_hit_weight * first_hit_velocity_loss
                # )

                loss = loss + (
                    self.velocity_dt_hit_weight * first_hit_velocity_loss
                )

                # loss = (
                #     loss
                #     + first_hit_aux_weight
                #     * (
                #         0.02 * fast_rate_loss
                #         + 0.05 * first_hit_dt_loss
                #         + 0.02 * first_hit_loss
                #     )
                # )
    
            else:
                loss = torch.tensor(0.0, device=v_pred.device, requires_grad=True)
                event_loss_raw = loss.detach() * 0.0
                event_loss = loss.detach() * 0.0
                n_contract = 0
            # print("Wow congrats")
            logs["loss"] = loss
            # if len(preds_list) > 0:
            #     logger.info(
            #         f"Velocity loss ({self.velocity_loss_mode}): total={loss.item():.6f} "
            #         f"plain={plain_mse.item():.6f} weighted={weighted_mse.item():.6f} "
            #         # f"dt_gate={dt_gate.item():.4f} dt_candidates={dt_candidates_loss.item():.6f} "
            #         # f"dt_hit={dt_hit_loss.item():.6f}"
            #     )
            # else:

            if self.record:
                dt_hit_pred_log = (
                    vel_metrics["dt_hit_pred"]
                    if np.isfinite(vel_metrics["dt_hit_pred"])
                    else -1.0
                )
                dt_hit_true_log = (
                    vel_metrics["dt_hit_true"]
                    if np.isfinite(vel_metrics["dt_hit_true"])
                    else -1.0
                )
                dt_hit_abs_err_log = (
                    vel_metrics["dt_hit_abs_err"]
                    if np.isfinite(vel_metrics["dt_hit_abs_err"])
                    else 1e6
                )
                dt_hit_rel_err_log = (
                    vel_metrics["dt_hit_rel_err"]
                    if np.isfinite(vel_metrics["dt_hit_rel_err"])
                    else 1e6
                )
                vel_wandb = {"train/velocity_loss": loss.item()}
                if len(preds_list) > 0:
                    vel_wandb.update({
                        "velocity/loss_plain_mse": plain_mse.item(),
                        "velocity/loss_weighted_mse": weighted_mse.item(),
                        "velocity/mse": vel_metrics["mse"],
                        "velocity/mse_vs_zero": vel_metrics["mse_vs_zero"],
                        "velocity/mse_vs_mean": vel_metrics["mse_vs_mean"],
                        "velocity/zero_baseline_mse": vel_metrics["zero_baseline_mse"],
                        "velocity/mean_baseline_mse": vel_metrics["mean_baseline_mse"],
                        "velocity/cosine": vel_metrics["cosine"],
                        "velocity/pearson": vel_metrics["pearson"],
                        "velocity/spearman": vel_metrics["spearman"],
                        "velocity/sign_acc": vel_metrics["sign_acc"],
                        "velocity/topk_overlap": vel_metrics["topk_overlap"],
                        "velocity/dt_hit_pred": dt_hit_pred_log,
                        "velocity/dt_hit_true": dt_hit_true_log,
                        "velocity/dt_hit_abs_err": dt_hit_abs_err_log,
                        "velocity/dt_hit_rel_err": dt_hit_rel_err_log,
                        "velocity/dt_first_hit_match": vel_metrics["dt_first_hit_match"],
                        "velocity/dt_first_hit_recall": vel_metrics["dt_first_hit_recall"],
                        "velocity/dt_first_hit_precision": vel_metrics["dt_first_hit_precision"],
                        "velocity/dt_topk_overlap": vel_metrics["dt_topk_overlap"],
                        "velocity/length_jitter_attempted": velocity_perturb_stats["attempted"],
                        "velocity/length_jitter_applied": velocity_perturb_stats["applied"],
                    })
                wandb.log(vel_wandb, step=self.stepper)
            # import pdb

            # pdb.set_trace()
        else:
            batch, ar_prep_stats = self._prepare_autoregressive_training_batch(batch)
            if "newick_autoregressive_trees" in batch:
                autoregressive_component_groups = [
                    get_structural_polytomy_groups_from_newick(newick_tree)
                    for newick_tree in batch["newick_autoregressive_trees"]
                ]
            else:
                autoregressive_component_groups = []
                for labeled_merge_cluster in batch["batched_autoregressive_labels"]:
                    seen_groups = set()
                    groups = []
                    for label in labeled_merge_cluster:
                        components = tuple(int(component) for component in label["components"])
                        if components in seen_groups:
                            continue
                        seen_groups.add(components)
                        groups.append(list(components))
                    autoregressive_component_groups.append(groups)

            autoregressive_times = self._effective_autoregressive_time_tensor(
                batch["batched_autoregressive_time"]
            )
            all_group_logits = self.forward(
                batch["tokenized_autoregressive_trees"],
                autoregressive_times,
                batch["phyla_embeddings"],
                autoregressive=True,
                autoregressive_component_groups=autoregressive_component_groups,
            )

            found = {}
            label_targets_by_batch = []
            for batch_index, labeled_merge_cluster in enumerate(batch["batched_autoregressive_labels"]):
                group_targets = {}
                for label in labeled_merge_cluster:
                    result_split = int(label["result_split"])
                    components = tuple(int(component) for component in label["components"])
                    merge_indices = [int(idx) for idx in label["merge_indices"]]
                    found[(batch_index, result_split)] = False
                    group_targets.setdefault(components, []).append(
                        (result_split, merge_indices)
                    )
                label_targets_by_batch.append(group_targets)

            losses = []

            total_metrics = []
            alternative_target_counts = []

            chosen_polytomies = []
            polytomy_logits = []
            polytomy_sizes = []  # Track size of each polytomy encountered

            for group in all_group_logits:
                logits = group["logits"]
                splits_in_polytomy = tuple(int(split) for split in group["splits_represented"])
                batch_index = int(group["batch_index"])
                
                # Track polytomy size (number of splits in the polytomy)
                polytomy_sizes.append(len(splits_in_polytomy))

                explicit_subsets = []
                for resulting_split, idxs in label_targets_by_batch[batch_index].get(
                    splits_in_polytomy,
                    [],
                ):
                    found[(batch_index, resulting_split)] = True
                    explicit_subsets.append(
                        tuple(sorted(int(splits_in_polytomy[i]) for i in idxs))
                    )

                candidate_subsets = list(dict.fromkeys(explicit_subsets))
                if (
                    self.autoregressive_target_mode == "ready_alternatives"
                    and "target_trees" in batch
                ):
                    ready_subsets = _ready_target_merge_subsets_for_group(
                        splits_in_polytomy,
                        batch["target_trees"][batch_index],
                        Tree(batch["newick_autoregressive_trees"][batch_index]).n_leaves,
                    )
                    for subset in ready_subsets:
                        subset = tuple(sorted(int(split) for split in subset))
                        if subset not in candidate_subsets:
                            candidate_subsets.append(subset)

                alternative_target_counts.append(float(len(candidate_subsets)))

                if not candidate_subsets:
                    chosen_polytomies.append(torch.tensor(0.0))
                else:
                    chosen_polytomies.append(torch.tensor(1.0))

                polytomy_logits.append(group["polytomy_pred"])

                if candidate_subsets:
                    G = logits.size(0)
                    mask = ~torch.eye(
                        G, dtype=torch.bool, device=logits.device
                    )  # off-diagonal only

                    # optionally only use one triangle (avoid double-counting symmetric pairs)
                    tri = torch.triu(mask, diagonal=1)

                    logits_vec = logits[tri]
                    finite = torch.isfinite(logits_vec)
                    candidate_losses = []
                    candidate_targets = []
                    for subset in candidate_subsets:
                        y = _subset_target_matrix(
                            splits_in_polytomy,
                            subset,
                            logits.device,
                        )
                        y_vec = y[tri]
                        candidate_targets.append(y)

                        logits_vec_f = logits_vec[finite]
                        y_vec_f = y_vec[finite]

                        pos = y_vec_f.sum().clamp(min=1.0)
                        neg = (y_vec_f.numel() - y_vec_f.sum()).clamp(min=1.0)
                        pos_weight = (neg / pos).detach()

                        candidate_losses.append(
                            F.binary_cross_entropy_with_logits(
                                logits_vec_f,
                                y_vec_f,
                                pos_weight=pos_weight,
                                reduction="mean",
                            )
                        )

                    loss_stack = torch.stack(candidate_losses)
                    best_candidate_index = int(torch.argmin(loss_stack).item())
                    loss = loss_stack[best_candidate_index]
                    best_target = candidate_targets[best_candidate_index]

                    metrics = compute_merge_metrics(
                        logits,
                        best_target,
                        threshold_logit=0.0,
                    )
                    total_metrics.append(metrics)

                    losses.append(loss)

            for (batch_index, split_mask), was_found in found.items():
                if not was_found:
                    print(
                        "Missing split: ",
                        [j for j in range(int(split_mask).bit_length()) if (int(split_mask) >> j) & 1],
                    )
                    raise Exception(
                        f"Did not find merge for split {split_mask} in batch element {batch_index}!"
                    )

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
            avg_alternative_targets = (
                float(np.mean(alternative_target_counts))
                if alternative_target_counts
                else 0.0
            )
            logger.info(f"Average polytomy size: {avg_polytomy_size}")
            logger.info(
                f"Average alternative autoregressive targets: {avg_alternative_targets}"
            )
            logs["autoregressive_stats/avg_candidate_targets"] = torch.tensor(
                avg_alternative_targets,
                device=all_group_logits[0]["logits"].device if all_group_logits else self.device,
            )
            logs["autoregressive_stats/rollin_attempted"] = torch.tensor(
                ar_prep_stats["rollin_attempted"],
                device=all_group_logits[0]["logits"].device if all_group_logits else self.device,
            )
            logs["autoregressive_stats/rollin_applied"] = torch.tensor(
                ar_prep_stats["rollin_applied"],
                device=all_group_logits[0]["logits"].device if all_group_logits else self.device,
            )
            logs["autoregressive_stats/dagger_attempted"] = torch.tensor(
                ar_prep_stats["dagger_attempted"],
                device=all_group_logits[0]["logits"].device if all_group_logits else self.device,
            )
            logs["autoregressive_stats/dagger_applied"] = torch.tensor(
                ar_prep_stats["dagger_applied"],
                device=all_group_logits[0]["logits"].device if all_group_logits else self.device,
            )
            dagger_avg_steps = (
                ar_prep_stats["dagger_rollout_steps"] / ar_prep_stats["dagger_applied"]
                if ar_prep_stats["dagger_applied"] > 0.0
                else 0.0
            )
            logs["autoregressive_stats/dagger_avg_rollout_steps"] = torch.tensor(
                dagger_avg_steps,
                device=all_group_logits[0]["logits"].device if all_group_logits else self.device,
            )
            logs["autoregressive_stats/structure_perturb_attempted"] = torch.tensor(
                ar_prep_stats["structure_perturb_attempted"],
                device=all_group_logits[0]["logits"].device if all_group_logits else self.device,
            )
            logs["autoregressive_stats/structure_perturb_applied"] = torch.tensor(
                ar_prep_stats["structure_perturb_applied"],
                device=all_group_logits[0]["logits"].device if all_group_logits else self.device,
            )

            if self.record:
                # Batch all metrics into a single wandb.log call to avoid step conflicts
                wandb_metrics = {
                    "train/autoregressive_loss": L_merging.item(),
                    "autoregressive_stats/avg_polytomy_size": avg_polytomy_size,
                    "autoregressive_stats/num_polytomies": num_polytomies,
                    "autoregressive_stats/avg_candidate_targets": avg_alternative_targets,
                    "autoregressive_stats/rollin_attempted": ar_prep_stats["rollin_attempted"],
                    "autoregressive_stats/rollin_applied": ar_prep_stats["rollin_applied"],
                    "autoregressive_stats/dagger_attempted": ar_prep_stats["dagger_attempted"],
                    "autoregressive_stats/dagger_applied": ar_prep_stats["dagger_applied"],
                    "autoregressive_stats/dagger_avg_rollout_steps": dagger_avg_steps,
                    "autoregressive_stats/structure_perturb_attempted": ar_prep_stats["structure_perturb_attempted"],
                    "autoregressive_stats/structure_perturb_applied": ar_prep_stats["structure_perturb_applied"],
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
        first_hit_tol=1e-4,
        max_events=1000,
        max_steps=20000,
        KNN_TOPM = 32,
        KNN_TAU = 0.05,
        KNN_STOCHASTIC = False,
        debug_real_tree=None,
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
        #tokenized = self.dataset.tree_tokenizer(current_newicks[0])
        # new_tokenized = ()
        # for i in tokenized:
        #     if torch.is_tensor(i):
        #         new_tokenized += (i.to(self.device),)
        #     else:
        #         new_tokenized += (i,)


        for b_idx, nw in enumerate(newick_starting_trees):
            t = Tree(nw)
            enc = BHVEncoder()
            masks, lens = enc.return_BHV_encoding(t)
            # BHV encoder uses canonical split orientation (with dummy-root influence),
            # while model/tokenizer uses directed edge masks on dummy-free Newick.
            # Convert initial lengths into tokenizer split-mask space once, so the
            # sampler state remains in one consistent representation.
            bhv_lengths = {int(m): float(l) for m, l in zip(masks, lens) if l is not None}
            model_masks_init = [int(m) for m in token_cache.edge_split_masks_list[b_idx] if int(m) != 0]

            biological_bits = max(t.n_leaves - 1, 0)
            full_model_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0

            td_init = {}
            for m_model in model_masks_init:
                length = bhv_lengths.get(m_model)
                if length is None and full_model_mask:
                    length = bhv_lengths.get(full_model_mask ^ m_model)

                if length is None:
                    raise Exception(
                        f"Could not map initial split {m_model} from BHV encoding to tokenizer mask space."
                    )
                # Keep sampler state aligned with active edges only; zero-length edges
                # are represented as absent and do not participate in dynamics.
                if float(length) > eps_len:
                    td_init[m_model] = float(length)

            trees.append(td_init)
            num_leaves.append(t.n_leaves)
            mapping.append(t.id_to_name)

        t = 0.0
        n_events = 0
        n_steps = 0
        n_topology_changes = 0
        num_topology_changes = 0
        polytomy_sizes = []  # Track sizes of polytomies encountered during sampling

        while t < T and n_steps < max_steps and n_events < max_events:
            n_steps += 1

            # --- encode/tokenize current trees for the model ---

            # Use CACHED tokenizer
            tokenized = self.model.tokenizer.forward_batched(token_cache, trees)
            #import pdb; pdb.set_trace()

            # if calculate_norm_rf(current_newicks[0], self.train_tree[0]) != 0:
            #     raise Exception("Current tree does not match training tree topology!")
            # #import pdb; pdb.set_trace()
            # if tokenized[0].shape[1] != self.train_tokenized_trees[0].shape[1]:
            #     raise Exception("Tokenized tree length mismatch!")
            # elif (new_tokenized[0] == self.train_tokenized_trees[0]).all().item() is False:
            #     raise Exception("Tokenized trees do not match!")
            
 
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
                # Use biological leaf universe (exclude dummy root leaf) for canonicalization.
                # Deriving bit-width from observed model masks can undercount when a high-index
                # leaf is absent in the current split set, which breaks complement matching.
                biological_bits = max(n_leaves - 1, 0)
                full_model_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
                # In Tree(...) representation we carry a dummy leaf at index n_leaves-1.
                # The edge incident to the dummy can appear as "all biological leaves" split;
                # tokenizer masks (built on dummy-free Newick) do not include this split.
                dummy_artifact_mask = full_model_mask
                V_model = v.squeeze(1).detach().cpu().numpy()

                L = []
                V_val = []
                masks = []
                supervised_edge_flags = []
                aligned_model_masks = []
                for m in td:
                    if m == dummy_artifact_mask:
                        continue

                    matched_m = m
                    dummy_bit_idx = n_leaves - 1
                    if biological_bits > 0 and ((m >> dummy_bit_idx) & 1):
                        matched_m = remove_bit(m, dummy_bit_idx)

                    if biological_bits > 0 and matched_m.bit_length() > biological_bits:
                        print(
                            f"Skipping split with unexpected bit_length: {m} "
                            f"(bit_length={m.bit_length()}, expected <= {biological_bits + 1})"
                        )
                        raise Exception("Unexpected split in tree while sampling that cannot be matched to model masks!")

                    idx = mask_idx.get(matched_m)
                    if idx is None and full_model_mask:
                        complement_m = full_model_mask ^ matched_m
                        idx = mask_idx.get(complement_m)
                        if idx is not None:
                            matched_m = complement_m

                    if idx is None:
                        print(
                            f"Whoa there is a split missing in velocity masks! {m} or "
                            f"{[i for i in range(m.bit_length()) if (m >> i) & 1]}"
                        )
                        raise Exception("Missing split in velocity masks!")

                    curr_len = float(td[m])
                    if curr_len <= eps_len:
                        continue
                    L.append(curr_len)

                    #We should not be making moves based on leafs! If leaf, velocity is 0
                    k_bits = int(matched_m).bit_count()
                    is_pendant = biological_bits > 0 and min(
                        k_bits, biological_bits - k_bits
                    ) == 1
                    if is_pendant:
                        V_val.append(0.0)
                        supervised_edge_flags.append(False)
                    else:
                        V_val.append(V_model[idx])
                        supervised_edge_flags.append(True)

                    masks.append(m)
                    aligned_model_masks.append(int(matched_m))


                V = np.array(V_val, dtype=np.float64)
                L = np.array(L, dtype=np.float64)
                supervised_mask = np.array(supervised_edge_flags, dtype=bool)
                
                if len(V) != len(L):
                    raise Exception("I assume these two things are equal length!")
                if len(supervised_mask) != len(L):
                    raise Exception("Supervised-mask and edge-length arrays must align!")

                if (L < 0).any():
                    raise Exception("There are negative lengths that is not possible!")

                # --- DEBUG: compare predicted vs true velocity at t=0 ---
                if debug_real_tree is not None and n_steps == 1:
                    try:
                        _, true_velocity = return_sampled_tree_orthant_velocity(
                            newick_starting_trees[b_idx], debug_real_tree, 0.0
                        )
                        # Match exactly the supervised subset used during training:
                        # remove dummy bit if present, allow complement orientation,
                        # and drop pendant edges.
                        v_pred_arr = []
                        v_true_arr = []
                        matched_masks_dbg = []
                        true_vel_by_model_mask = {}
                        skipped_pendant = 0
                        unmatched = 0
                        for vel_mask, tv in true_velocity.items():
                            vel = int(vel_mask)
                            if biological_bits > 0 and vel.bit_length() == biological_bits + 1:
                                vel = remove_bit(vel, n_leaves - 1)
                            elif biological_bits > 0 and vel.bit_length() > biological_bits + 1:
                                unmatched += 1
                                continue

                            matched_vel = vel
                            idx = mask_idx.get(matched_vel)
                            if idx is None and full_model_mask:
                                complement_vel = full_model_mask ^ matched_vel
                                idx = mask_idx.get(complement_vel)
                                if idx is not None:
                                    matched_vel = complement_vel

                            if idx is None:
                                unmatched += 1
                                continue

                            k_bits = int(matched_vel).bit_count()
                            is_pendant = biological_bits > 0 and min(
                                k_bits, biological_bits - k_bits
                            ) == 1
                            if is_pendant:
                                skipped_pendant += 1
                                continue

                            v_pred_arr.append(float(V_model[idx]))
                            v_true_arr.append(float(tv))
                            matched_masks_dbg.append(int(matched_vel))
                            true_vel_by_model_mask[int(matched_vel)] = float(tv)

                        if len(v_pred_arr) > 0:
                            v_pred_np = np.array(v_pred_arr)
                            v_true_np = np.array(v_true_arr)
                            mse = float(np.mean((v_pred_np - v_true_np) ** 2))
                            mae = float(np.mean(np.abs(v_pred_np - v_true_np)))
                            cos_num = np.dot(v_pred_np, v_true_np)
                            cos_den = (np.linalg.norm(v_pred_np) * np.linalg.norm(v_true_np))
                            cosine_sim = float(cos_num / cos_den) if cos_den > 0 else 0.0
                            print(f"\n===== DEBUG: Predicted vs True velocity at t=0 (tree {b_idx}) =====")
                            print(
                                f"  Matched supervised internal edges: {len(v_pred_arr)} "
                                f"(of {len(true_velocity)} true; skipped pendant={skipped_pendant}, unmatched={unmatched})"
                            )
                            print(f"  MSE:  {mse:.6e}")
                            print(f"  MAE:  {mae:.6e}")
                            print(f"  Cosine similarity: {cosine_sim:.6f}")
                            print(f"  Pred  range: [{v_pred_np.min():.6e}, {v_pred_np.max():.6e}]")
                            print(f"  True  range: [{v_true_np.min():.6e}, {v_true_np.max():.6e}]")

                            # Compare dt_hit on matched supervised edges using the same lengths.
                            matched_idx = [
                                i
                                for i, mm in enumerate(aligned_model_masks)
                                if supervised_mask[i] and (mm in true_vel_by_model_mask)
                            ]
                            if len(matched_idx) > 0:
                                L_match = L[matched_idx]
                                v_pred_match = V[matched_idx]
                                v_true_match = np.array(
                                    [true_vel_by_model_mask[aligned_model_masks[i]] for i in matched_idx],
                                    dtype=np.float64,
                                )

                                pred_neg_match = (v_pred_match < 0.0) & (L_match > eps_len)
                                true_neg_match = (v_true_match < 0.0) & (L_match > eps_len)

                                pred_dt_candidates = (
                                    L_match[pred_neg_match] / -v_pred_match[pred_neg_match]
                                    if np.any(pred_neg_match)
                                    else np.array([], dtype=np.float64)
                                )
                                true_dt_candidates = (
                                    L_match[true_neg_match] / -v_true_match[true_neg_match]
                                    if np.any(true_neg_match)
                                    else np.array([], dtype=np.float64)
                                )

                                dt_hit_pred_dbg = (
                                    float(np.min(pred_dt_candidates))
                                    if pred_dt_candidates.size > 0
                                    else float("inf")
                                )
                                dt_hit_true_dbg = (
                                    float(np.min(true_dt_candidates))
                                    if true_dt_candidates.size > 0
                                    else float("inf")
                                )

                                print(
                                    f"  dt_hit(pred, matched supervised): {dt_hit_pred_dbg:.6e} "
                                    f"(neg={int(pred_neg_match.sum())}, candidates={pred_dt_candidates.size})"
                                )
                                print(
                                    f"  dt_hit(true, matched supervised): {dt_hit_true_dbg:.6e} "
                                    f"(neg={int(true_neg_match.sum())}, candidates={true_dt_candidates.size})"
                                )
                                if pred_dt_candidates.size > 0:
                                    print(
                                        f"  Pred dt candidates (min-5): {np.sort(pred_dt_candidates)[:5]}"
                                    )
                                if true_dt_candidates.size > 0:
                                    print(
                                        f"  True dt candidates (min-5): {np.sort(true_dt_candidates)[:5]}"
                                    )
                            else:
                                print(
                                    "  dt_hit debug: no matched supervised edges with both length and true velocity."
                                )
                            # Show top-5 worst mismatches
                            abs_err = np.abs(v_pred_np - v_true_np)
                            worst_idx = np.argsort(abs_err)[::-1][:5]
                            print(f"  Top-5 worst mismatches:")
                            for wi in worst_idx:
                                print(f"    split={matched_masks_dbg[wi]:>12}  pred={v_pred_np[wi]:+.6e}  true={v_true_np[wi]:+.6e}  err={abs_err[wi]:.6e}")
                            print(f"============================================================\n")
                            # import pdb; pdb.set_trace()
                        else:
                            print(f"DEBUG: Could not match any velocity splits for tree {b_idx}")
                    except Exception as e:
                        print(f"DEBUG: Failed to compute true velocity for comparison: {e}")

                # --- compute dt_hit ---
                moving_neg = supervised_mask & (V < 0.0) & (L > eps_len)
                if np.any(moving_neg):
                    dt_candidates = L[moving_neg] / -V[moving_neg]
                    dt_hit = float(np.min(dt_candidates))
                else:
                    dt_hit = float("inf")

                cache.append((td, L, V, n_leaves, mapp, dt_hit, masks, moving_neg))
                dt_hit_list.append(dt_hit)

            # ---- GLOBAL dt across the batch ----
            dt_hit_global = min(dt_hit_list) if len(dt_hit_list) else float("inf")
            # Experimenting here, dt_hit_global is not a good metric we just jump, jump, jump, so why not use dt_base
            dt = min(dt_base, dt_hit_global, T - t)
            #dt = min(dt_base, T-t)

            # defensive: prevent hard stall
            if dt <= 0:
                dt = min(dt_base, T - t)


            # ---- SECOND PASS: advance everyone with the SAME dt ----
            new_trees = []

            # Since update of token_cache happens per tree potentially, we need to defer it or track which ones changed.
            # However, batch indices align with zip(trees...), so we can update token_cache[i] if needed.

            for b_idx, (td, L, V, n_leaves, mapp, dt_hit, masks, moving_neg) in enumerate(
                cache
            ):
                model_masks = edge_splits[b_idx]
                # --- advance ---
                L_new = L + dt * V
                # import pdb; pdb.set_trace()
                
                # treat as boundary if we stepped past the first hit time for THIS tree
                # (float equality with hit_tol=1e-10 is too strict)
                hit_boundary = np.isfinite(dt_hit) and dt >= dt_hit
                if not hit_boundary and np.any(moving_neg):
                    # Numerical fallback: only moving supervised edges can trigger hits.
                    hit_boundary = bool((L_new[moving_neg] <= eps_len).any())

                if hit_boundary and np.isfinite(dt_hit) and np.any(moving_neg):
                    # Collapse near-simultaneous first-hit edges into the same boundary event.
                    neg_idx = np.where(moving_neg)[0]
                    dt_candidates = L[neg_idx] / np.maximum(-V[neg_idx], eps_len)
                    near_first_hit = np.abs(dt_candidates - dt_hit) <= float(first_hit_tol)
                    if np.any(near_first_hit):
                        L_new[neg_idx[near_first_hit]] = 0.0

                
                # update dict
                td2 = {m: float(l) for m, l in zip(masks, L_new) if l > eps_len}

                # We only need to rebuild Newick/Graph if we hit a boundary (topology changed)
                if hit_boundary:
                    num_merges = 0
                    topology_changed = True
                    while topology_changed and n_events < max_events and num_merges < max_events:
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
                        else:
                            break

                        with torch.no_grad():
                            autoregressive_component_groups = [
                                get_structural_polytomy_groups_from_newick(td2_newick)
                            ]
                            logit_outputs = self.forward(
                                tokenized_trees,
                                self._effective_autoregressive_time_tensor(
                                    t
                                ),
                                phyla_embeddings,
                                autoregressive=True,
                                autoregressive_component_groups=autoregressive_component_groups,
                            )
                        
                        planned_merges = _plan_autoregressive_boundary_merges(
                            logit_outputs,
                            td2.keys(),
                        )
                        if planned_merges:
                            planned_merges = planned_merges[:1]

                        top_change = False
                        if not planned_merges:
                            logger.info("No valid merges found!")
                        else:
                            for planned in planned_merges:
                                polytomy_sizes.append(len(planned["splits_represented"]))

                                logits = planned["logits"]
                                G = logits.size(0)
                                tri = torch.triu(
                                    ~torch.eye(G, dtype=torch.bool, device=logits.device),
                                    diagonal=1,
                                )
                                logits_vec = logits[tri]
                                finite_logits = logits_vec[torch.isfinite(logits_vec)]
                                if finite_logits.numel() > 0:
                                    max_logits.append(
                                        float(torch.sigmoid(finite_logits).max().item())
                                    )

                                for subset, new_split in planned["subsets"]:
                                    to_print = [
                                        i
                                        for i in range(new_split.bit_length())
                                        if (new_split >> i) & 1
                                    ]
                                    logger.info(
                                        f"Merging subset {list(subset)} to create split {new_split}: {to_print}"
                                    )

                                    # New splits are born at the boundary; seed them
                                    # with a small positive length to avoid an
                                    # immediate re-collapse while keeping geometry local.
                                    td2[new_split] = 1e-3
                                    n_events += 1
                                    num_topology_changes += 1
                                    top_change = True

                            if top_change:
                                num_merges += 1
                                logger.info("Merge step performed from decoded subsets")
                        topology_changed = top_change

                    if topology_changed and (n_events >= max_events or num_merges >= max_events):
                        logger.info(
                            "Stopping boundary-resolution loop after hitting the merge-event cap."
                        )

                        # if not top_change:
                        #     logger.info("No more merges possible, pick a random polytomy and do a KNN merge")
                        #     output = random.choice(logit_outputs)
                        #     split_embeddings = output['group_embeddings']
                        #     group_represented = output['splits_represented']

                        #     if len(group_represented) != split_embeddings.size(0):
                        #         raise Exception("Whoa size mismatch between groups and split embeddings")
                            
                        #     i, j = _pick_knn_pair(split_embeddings, topM=KNN_TOPM, tau=KNN_TAU, stochastic=KNN_STOCHASTIC)

                        #     sm_i, sm_j = group_represented[i], group_represented[j]
                        #     new_split = int(sm_i) | int(sm_j)

                        #     if new_split not in td2:
                        #         # td2[new_split] = 1e-3  # tiny length
                        #         curr_lens = list(td2.values())
                        #         if len(curr_lens) > 0:
                        #             td2[new_split] = float(np.median(curr_lens))
                        #         else:
                        #             td2[new_split] = 1e-3
                        #     else:
                        #         # import pdb; pdb.set_trace()
                        #         raise Exception("Not possible to merge into a split that already exists...")

                        #     top_change = True
                        #     num_merges += 1
                        #     n_events += 1
                        #     num_topology_changes += 1

                        
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
    
        logger.info(f"Sampling finished in {n_steps} steps. Total events: {n_events}, topology changes: {num_topology_changes}, average polytomy size: {avg_polytomy_size:.2f}")

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
        seq_ordering_map = batch["sequence_ordering_maps"][0]

        if train:
            real_trees = self.dataset.dataset_train.return_posterior_trees(id)
            num_leaves = self.dataset.dataset_train.return_number_leaves(id)
        else:
            real_trees = self.dataset.dataset_val.return_posterior_trees(id)
            num_leaves = self.dataset.dataset_val.return_number_leaves(id)

        if len(real_trees) > num_samples:
            pot_real_trees = random.sample(real_trees, num_samples)
        else:
            pot_real_trees = real_trees

        sanity_check = self.dataset.dataset_train.sanity_check if train else self.dataset.dataset_val.sanity_check
        random_sanity_check = self.dataset.dataset_train.random_sanity_check if train else self.dataset.dataset_val.random_sanity_check

        def _remap_tree_to_batch_indexing(tree_newick, offset=0, tree_kind="tree"):
            t_obj = EteTree(tree_newick, format=1)
            for leaf in t_obj.get_leaves():
                lookup_name = leaf.name
                if offset:
                    try:
                        lookup_name = str(int(lookup_name) + offset)
                    except ValueError:
                        raise Exception(
                            f"Non-integer leaf name '{leaf.name}' encountered in {tree_kind} while applying offset {offset}."
                        )

                mapped_name = seq_ordering_map.get(lookup_name)
                if mapped_name is None:
                    raise Exception(
                        f"Leaf name '{lookup_name}' in {tree_kind} not found in sequence ordering map."
                    )
                leaf.name = mapped_name

            return t_obj.write(format=1)

        real_trees = []
        for i in pot_real_trees:
            real_trees.append(_remap_tree_to_batch_indexing(i, offset=0, tree_kind="real tree"))

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
            # rt = Tree(num_leaves=num_leaves, random=True)
            # starting_tree = str(rt)
            if train:
                starting_tree = self.dataset.dataset_train.sample_random_tree(
                    real_trees[0]
                )
            else:
                starting_tree = self.dataset.dataset_val.sample_random_tree(
                    real_trees[0]
                )

            # Random trees are offset only in the default non-sanity path.
            random_tree_offset = 1 if (not sanity_check and not random_sanity_check) else 0
            starting_tree = _remap_tree_to_batch_indexing(
                starting_tree, offset=random_tree_offset, tree_kind="random tree"
            )


            #### DEBUG CHANGE LATER MADE ONE TIMEPOINT ####
            timepoint = random.uniform(0, 1)

            start_time = time.time()
            sampled_tree, n_topology_changes_one, avg_max_logit, avg_polytomy_size, n_polytomies_resolved_one = self.sample(
                [starting_tree], batch["phyla_embeddings"], num_samples=1, dt_base=dt,
                debug_real_tree=real_trees[0],
            )
            print(f"Sampling a single tree took {time.time() - start_time} seconds")

            avg_polytomy_sizes.append(avg_polytomy_size)
            num_polytomies_resolved.append(n_polytomies_resolved_one)

            sampled_tree = sampled_tree[0]
            num_topology_changes.append(n_topology_changes_one)
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
        posterior_trees = [number_to_name_newick(i, {int(i):v for i, v in mapping.items()}, True) for i in real_trees]

        rf_dists = []
        n_pairs = min(len(sampled), len(posterior_trees))
        for i in range(n_pairs):
            rf_dists.append(calculate_norm_rf(sampled[i], posterior_trees[i]))
        
        rf_norm_val = np.mean(rf_dists) if rf_dists else 0.0

        if save:
            import pickle
            os.makedirs("samples", exist_ok=True)
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
                    "samples/number_of_polytomies_resolved": num_polytomies,
                    "samples/average_topology_changes": np.mean(num_topology_changes),
                    "samples/average_max_logits": np.mean(avg_max_logits),
                    "samples/average_num_polytomies_resolved": avg_num_polytomies_resolved,
                    "samples/average_polytomy_size": overall_avg_polytomy_size,
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
                        loss_unscaled = logs["loss"]
                        loss = (
                            loss_unscaled * self.training_step_velocity_weight
                        )
                        logs["train/velocity_loss_unscaled"] = loss_unscaled.detach()
                        logs["train/velocity_loss_scaled"] = loss.detach()
                        logs["loss"] = loss
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
                    velocity_metric_logs = {
                        k: v for k, v in logs_vel.items() if k.startswith("velocity/")
                    }
                    loss_vel_unscaled = logs_vel["loss"]
                    loss_vel = (
                        loss_vel_unscaled * self.training_step_velocity_weight
                    )
                    loss_vel_unscaled_detached = loss_vel_unscaled.detach()
                    loss_vel_scaled_detached = loss_vel.detach()
                    logging.info(
                        "Velocity head loss: raw=%.6f scaled=%.6f weight=%.4f",
                        float(loss_vel_unscaled_detached.item()),
                        float(loss_vel_scaled_detached.item()),
                        float(self.training_step_velocity_weight),
                    )
                    self.manual_backward(loss_vel)
                    pre_ar_grads = None
                    velocity_grad_norm = None
                    if self.training_step_autoregressive_grad_ratio is not None:
                        pre_ar_grads = {}
                        vel_sq = 0.0
                        for p in self.model.parameters():
                            if p.grad is None:
                                continue
                            g_prev = p.grad.detach().clone()
                            pre_ar_grads[p] = g_prev
                            vel_sq += float(torch.sum(g_prev * g_prev))
                        velocity_grad_norm = vel_sq ** 0.5
                    # self.clip_gradients(
                    #     opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
                    # )
                    # opt.step()
                    # opt.zero_grad()
                    logging.info("DEBUG: Finished Velocity Head Training")

                    del logs_vel
                    del loss_vel_unscaled
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
                    loss_unscaled = loss
                    loss = (
                        loss_unscaled * self.training_step_autoregressive_weight
                    )
                    logs["train/autoregressive_loss_unscaled"] = (
                        loss_unscaled.detach()
                    )
                    logs["train/autoregressive_loss_scaled"] = loss.detach()
                    logs["train/velocity_loss_unscaled"] = (
                        loss_vel_unscaled_detached
                    )
                    logs["train/velocity_loss_scaled"] = (
                        loss_vel_scaled_detached
                    )
                    logs.update(velocity_metric_logs)
                    logs["loss"] = loss
                    logging.info(
                        "Autoregressive head loss: raw=%.6f scaled=%.6f weight=%.4f",
                        float(loss_unscaled.detach().item()),
                        float(loss.detach().item()),
                        float(self.training_step_autoregressive_weight),
                    )

                    logging.info(
                        f"Memory allocated before backward: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                    )
                    logging.info(
                        f"Memory reserved before backward: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                    )

                    self.manual_backward(loss)
                    if (
                        self.training_step_autoregressive_grad_ratio is not None
                        and pre_ar_grads is not None
                        and velocity_grad_norm is not None
                    ):
                        ar_sq = 0.0
                        for p in self.model.parameters():
                            if p.grad is None:
                                continue
                            g_prev = pre_ar_grads.get(p)
                            if g_prev is None:
                                ar_delta = p.grad.detach()
                            else:
                                ar_delta = p.grad.detach() - g_prev
                            ar_sq += float(torch.sum(ar_delta * ar_delta))
                        autoregressive_grad_norm = ar_sq ** 0.5
                        grad_scale = 1.0
                        if (
                            autoregressive_grad_norm > 1e-12
                            and velocity_grad_norm > 1e-12
                        ):
                            target_norm = (
                                velocity_grad_norm
                                * self.training_step_autoregressive_grad_ratio
                            )
                            if autoregressive_grad_norm > target_norm:
                                grad_scale = target_norm / (
                                    autoregressive_grad_norm + 1e-12
                                )
                                for p in self.model.parameters():
                                    g_prev = pre_ar_grads.get(p)
                                    if p.grad is None:
                                        if g_prev is not None:
                                            p.grad = g_prev.clone()
                                        continue
                                    if g_prev is None:
                                        p.grad.mul_(grad_scale)
                                    else:
                                        p.grad.copy_(
                                            g_prev + (p.grad - g_prev) * grad_scale
                                        )
                        device_for_logs = loss.device
                        logs["train/velocity_grad_norm"] = torch.tensor(
                            velocity_grad_norm, device=device_for_logs
                        )
                        logs["train/autoregressive_grad_norm"] = torch.tensor(
                            autoregressive_grad_norm, device=device_for_logs
                        )
                        logs["train/autoregressive_grad_scale"] = torch.tensor(
                            grad_scale, device=device_for_logs
                        )
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
                if self.training_sampling_mode == "harness_sanity":
                    metrics = self.sample_compare_harness(train=True)
                else:
                    metrics = self.sample_compare(batch, train=True, dt=self.dt)
                
                for k, v in metrics.items():
                    self.log(f"sample_metrics/{k}", v, on_step=True, logger=True)
                self._append_sample_metrics_trace(metrics)
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
