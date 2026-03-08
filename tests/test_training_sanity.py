import random
import sys
import types
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
import copy
import math

import torch
from ete3 import Tree as EteTree

_DT_FIRST_HIT_TOL = 0.01


def _install_phyla_stub():
    phyla_mod = types.ModuleType("phyla")
    phyla_utils_mod = types.ModuleType("phyla.utils")
    phyla_utils_utils_mod = types.ModuleType("phyla.utils.utils")
    phyla_eval_mod = types.ModuleType("phyla.eval")
    phyla_eval_evo_mod = types.ModuleType("phyla.eval.evo_reasoning_eval")

    class Config:
        def __init__(self):
            self.trainer = types.SimpleNamespace(checkpoint_path="")
            self.eval = types.SimpleNamespace(device="cpu")

    def load_config(config_cls):
        return config_cls()

    def load_model(config=None, random_model=False):
        return {"model": MagicMock()}

    def _encode_sequences_openfold_style(sequences, names):
        batch = {
            "encoded_sequences": torch.zeros((len(sequences), 1), dtype=torch.long),
            "sequence_mask": torch.ones((len(sequences), 1), dtype=torch.long),
            "cls_positions": torch.ones((len(sequences), 1), dtype=torch.bool),
        }
        return batch, None

    phyla_utils_utils_mod.load_config = load_config
    phyla_eval_evo_mod.Config = Config
    phyla_eval_evo_mod.load_model = load_model
    phyla_eval_evo_mod._encode_sequences_openfold_style = (
        _encode_sequences_openfold_style
    )

    phyla_mod.utils = phyla_utils_mod
    phyla_mod.eval = phyla_eval_mod
    phyla_utils_mod.utils = phyla_utils_utils_mod
    phyla_eval_mod.evo_reasoning_eval = phyla_eval_evo_mod

    sys.modules["phyla"] = phyla_mod
    sys.modules["phyla.utils"] = phyla_utils_mod
    sys.modules["phyla.utils.utils"] = phyla_utils_utils_mod
    sys.modules["phyla.eval"] = phyla_eval_mod
    sys.modules["phyla.eval.evo_reasoning_eval"] = phyla_eval_evo_mod


def _install_deepspeed_stub():
    if "deepspeed" in sys.modules:
        return

    ds_mod = types.ModuleType("deepspeed")
    ds_ops_mod = types.ModuleType("deepspeed.ops")
    ds_adam_mod = types.ModuleType("deepspeed.ops.adam")

    class FusedAdam(torch.optim.Adam):
        pass

    ds_adam_mod.FusedAdam = FusedAdam
    ds_ops_mod.adam = ds_adam_mod
    ds_mod.ops = ds_ops_mod

    sys.modules["deepspeed"] = ds_mod
    sys.modules["deepspeed.ops"] = ds_ops_mod
    sys.modules["deepspeed.ops.adam"] = ds_adam_mod


try:
    from deepspeed.ops.adam import FusedAdam as _FusedAdamCheck  # noqa: F401
except Exception:
    _install_deepspeed_stub()

try:
    from phyla.utils.utils import load_config as _LoadConfigCheck  # noqa: F401
    from phyla.eval.evo_reasoning_eval import Config as _ConfigCheck  # noqa: F401
except Exception:
    _install_phyla_stub()

from model.model import TreeDenoiserTokenGT
from data.dataset import TreeDataset
from run.TrainingModule import TrainingModule
from utils.bhv_distance import bhv_geodesic_with_support
from utils.bhv_utils import (
    BHVEncoder,
    get_structural_polytomy_groups_from_newick,
    return_sampled_tree_boundary_decisions,
    return_sampled_tree_orthant_velocity,
    return_tree_boundary_merge_paths,
)
from utils.bhv_movie import build_tree_from_splits
from utils.random_tree import Tree
from utils.utils import remove_bit


def _detach_tokenized_batch(tokenized):
    out = []
    for item in tokenized:
        if torch.is_tensor(item):
            out.append(item.detach())
        else:
            out.append(item)
    return tuple(out)


def _make_single_velocity_batch(tokenizer, n_leaves, seed):
    random.seed(seed)
    torch.manual_seed(seed)

    start_tree = str(Tree(num_leaves=n_leaves, random=True))
    target_tree = str(Tree(num_leaves=n_leaves, random=True))
    sampled_newick, velocity = return_sampled_tree_orthant_velocity(
        start_tree, target_tree, 0.0
    )
    with torch.no_grad():
        tokenized = _detach_tokenized_batch(tokenizer([sampled_newick]))

    return {
        "tokenized_trees": tokenized,
        "batched_time": torch.tensor([0.0], dtype=torch.float32),
        "phyla_embeddings": None,
        "original_trees": [sampled_newick],
        "batched_velocity": [velocity],
        "num_leaves": [Tree(sampled_newick).n_leaves],
    }


def _leaf_sort_key(name):
    try:
        return (0, int(name))
    except ValueError:
        return (1, str(name))


def _prune_and_renumber_tree_pair(start_newick, target_newick, keep_leaves=12):
    t_start = EteTree(start_newick, format=1)
    t_target = EteTree(target_newick, format=1)

    start_names = {leaf.name for leaf in t_start.get_leaves()}
    target_names = {leaf.name for leaf in t_target.get_leaves()}
    common = sorted(start_names & target_names, key=_leaf_sort_key)
    if len(common) < keep_leaves:
        raise AssertionError(
            f"Not enough shared leaves to prune: have {len(common)}, need {keep_leaves}"
        )

    kept = common[:keep_leaves]
    t_start.prune(kept, preserve_branch_length=True)
    t_target.prune(kept, preserve_branch_length=True)

    renumber_map = {
        old_name: str(i)
        for i, old_name in enumerate(sorted(kept, key=_leaf_sort_key))
    }

    for tree in (t_start, t_target):
        for leaf in tree.get_leaves():
            leaf.name = renumber_map[leaf.name]

    return t_start.write(format=1), t_target.write(format=1)


def _make_batch_from_tree_pair(tokenizer, start_tree, target_tree, time_point=0.0):
    sampled_newick, velocity = return_sampled_tree_orthant_velocity(
        start_tree, target_tree, time_point
    )
    with torch.no_grad():
        tokenized = _detach_tokenized_batch(tokenizer([sampled_newick]))

    return {
        "tokenized_trees": tokenized,
        "batched_time": torch.tensor([float(time_point)], dtype=torch.float32),
        "phyla_embeddings": None,
        "original_trees": [sampled_newick],
        "batched_velocity": [velocity],
        "num_leaves": [Tree(sampled_newick).n_leaves],
    }


def _make_batch_from_tree_pair_with_autoregressive(
    tokenizer,
    start_tree,
    target_tree,
    time_point=0.0,
    max_boundary_attempts=20,
):
    sampled_newick, velocity = return_sampled_tree_orthant_velocity(
        start_tree, target_tree, time_point
    )

    boundary_labels = []
    for _ in range(max_boundary_attempts):
        boundary_labels = return_sampled_tree_boundary_decisions(start_tree, target_tree)
        if boundary_labels:
            break

    if not boundary_labels:
        raise AssertionError(
            "Could not sample autoregressive boundary decisions for the sanity tree pair."
        )

    chosen_boundary = boundary_labels[0]
    with torch.no_grad():
        tokenized = _detach_tokenized_batch(tokenizer([sampled_newick]))
        tokenized_ar = _detach_tokenized_batch(tokenizer([chosen_boundary["newick"]]))

    return {
        "tokenized_trees": tokenized,
        "batched_time": torch.tensor([float(time_point)], dtype=torch.float32),
        "phyla_embeddings": None,
        "original_trees": [sampled_newick],
        "batched_velocity": [velocity],
        "num_leaves": [Tree(sampled_newick).n_leaves],
        "tokenized_autoregressive_trees": tokenized_ar,
        "newick_autoregressive_trees": [chosen_boundary["newick"]],
        "batched_autoregressive_time": torch.tensor([0.0], dtype=torch.float32),
        "batched_autoregressive_labels": [chosen_boundary["labels"]],
    }


def _make_autoregressive_event_batch(tokenizer, newick, labels, event_time):
    with torch.no_grad():
        tokenized_ar = _detach_tokenized_batch(tokenizer([newick]))

    return {
        "tokenized_autoregressive_trees": tokenized_ar,
        "newick_autoregressive_trees": [newick],
        "batched_autoregressive_time": torch.tensor(
            [float(event_time)], dtype=torch.float32
        ),
        "batched_autoregressive_labels": [labels],
        "phyla_embeddings": None,
    }


def _select_random_nonbinary_boundary_path(start_tree, target_tree, seed=777):
    boundary_paths = return_tree_boundary_merge_paths(start_tree, target_tree)
    candidates = [
        path
        for path in boundary_paths
        if len(path["events"]) > 1
        and any(
            len(label["merge_indices"]) > 2
            for event in path["events"]
            for label in event["labels"]
        )
    ]
    if not candidates:
        raise AssertionError(
            "Did not find any multi-step non-binary boundary path on the sanity tree pair."
        )

    rng = random.Random(seed)
    return rng.choice(candidates)


def _target_merge_subsets_for_event(labels):
    target = {}
    for label in labels:
        components = tuple(int(component) for component in label["components"])
        merge_subset = frozenset(int(components[idx]) for idx in label["merge_indices"])
        target.setdefault(components, set()).add(merge_subset)
    return target


def _decode_positive_merge_subsets(group, threshold_logit=0.0):
    splits = [int(split) for split in group["splits_represented"]]
    logits = group["logits"].detach().cpu()
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
    decoded_subsets = set()
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
            decoded_subsets.add(
                frozenset(splits[node_idx] for node_idx in sorted(component))
            )

    return decoded_subsets


def _predict_autoregressive_event(module, tokenizer, newick, event_time):
    groups = get_structural_polytomy_groups_from_newick(newick)
    with torch.no_grad():
        tokenized_ar = _detach_tokenized_batch(tokenizer([newick]))
        outputs = module.forward(
            tokenized_ar,
            torch.tensor([float(event_time)], dtype=torch.float32),
            None,
            autoregressive=True,
            autoregressive_component_groups=[groups],
        )

    predicted = {}
    allow_all_groups = len(outputs) == 1
    for group in outputs:
        group_key = tuple(int(split) for split in group["splits_represented"])
        if (not allow_all_groups) and float(group["polytomy_pred"].detach().cpu().item()) <= 0.0:
            continue
        predicted[group_key] = _decode_positive_merge_subsets(group)

    return predicted, outputs


def _boundary_start_length_map(start_tree, target_tree, boundary_index):
    enc = BHVEncoder()
    start_obj = Tree(start_tree)
    target_obj = Tree(target_tree)
    start_masks, start_lengths = enc.return_BHV_encoding(start_obj)
    target_masks, target_lengths = enc.return_BHV_encoding(target_obj)
    geodesic = bhv_geodesic_with_support(
        {int(mask): float(length) for mask, length in zip(start_masks, start_lengths)},
        {int(mask): float(length) for mask, length in zip(target_masks, target_lengths)},
        n_leaves=start_obj.n_leaves,
    )
    boundary_lengths = {
        int(mask): float(length)
        for mask, length in geodesic["segments"][boundary_index]["end_lengths"].items()
        if float(length) > 1e-8
    }
    return boundary_lengths, start_obj.n_leaves, start_obj.id_to_name


def _apply_predicted_merge_subsets_to_length_map(length_map, predicted_group_subsets):
    next_lengths = dict(length_map)
    for merge_subsets in predicted_group_subsets.values():
        for subset in merge_subsets:
            new_split = 0
            for component in subset:
                new_split |= int(component)
            if new_split in next_lengths:
                raise AssertionError(
                    f"Predicted merge created an existing split {new_split}."
                )
            next_lengths[int(new_split)] = 0.1
    return next_lengths


def _assert_same_topology(testcase, left_newick, right_newick, message):
    left_tree = EteTree(left_newick)
    right_tree = EteTree(right_newick)
    rf_distance, max_rf, *_ = left_tree.robinson_foulds(
        right_tree,
        unrooted_trees=True,
    )
    testcase.assertEqual(
        0.0 if max_rf == 0 else rf_distance / max_rf,
        0.0,
        f"{message} RF distance={rf_distance}, max RF={max_rf}",
    )


def _normalized_rf(left_newick, right_newick):
    left_tree = EteTree(left_newick)
    right_tree = EteTree(right_newick)
    rf_distance, max_rf, *_ = left_tree.robinson_foulds(
        right_tree,
        unrooted_trees=True,
    )
    return 0.0 if max_rf == 0 else rf_distance / max_rf


def _rollout_boundary_with_autoregressive_head(
    testcase,
    module,
    tokenizer,
    boundary_path,
    start_tree,
    target_tree,
):
    current_lengths, n_leaves, mapping = _boundary_start_length_map(
        start_tree,
        target_tree,
        boundary_path["boundary_index"],
    )
    _, current_newick = build_tree_from_splits(
        list(current_lengths.keys()),
        current_lengths,
        n_leaves,
        root_leaf=n_leaves - 1,
        mapping=mapping,
    )

    for event_idx, event in enumerate(boundary_path["events"]):
        _assert_same_topology(
            testcase,
            current_newick,
            event["newick"],
            f"Rollout diverged before boundary event {event_idx}.",
        )
        predicted, _ = _predict_autoregressive_event(
            module,
            tokenizer,
            current_newick,
            event_idx / 63.0,
        )
        current_lengths = _apply_predicted_merge_subsets_to_length_map(
            current_lengths,
            predicted,
        )
        _, current_newick = build_tree_from_splits(
            list(current_lengths.keys()),
            current_lengths,
            n_leaves,
            root_leaf=n_leaves - 1,
            mapping=mapping,
        )
        expected_next_newick = (
            boundary_path["events"][event_idx + 1]["newick"]
            if event_idx + 1 < len(boundary_path["events"])
            else boundary_path["end_newick"]
        )
        _assert_same_topology(
            testcase,
            current_newick,
            expected_next_newick,
            f"Rollout diverged after boundary event {event_idx}.",
        )

    return current_newick


def _sample_to_first_boundary(module, start_tree, dt_hit_true, boundary_path):
    max_events = max(8, 4 * len(boundary_path["events"]))
    sampled_trees, *_ = module.sample(
        [start_tree],
        None,
        num_samples=1,
        T=float(dt_hit_true),
        dt_base=max(float(dt_hit_true), 1e-6),
        max_events=max_events,
        max_steps=2048,
    )
    sampled_tree = sampled_trees[0]
    return {
        "sampled_tree": sampled_tree,
        "target_tree": boundary_path["end_newick"],
        "rf_norm": _normalized_rf(sampled_tree, boundary_path["end_newick"]),
        "boundary_path": boundary_path,
    }


def _gather_supervised_velocity(module, batch, eps_len=1e-8):
    with torch.no_grad():
        v_pred, edge_split_masks, _ = module.forward(
            batch["tokenized_trees"],
            batch["batched_time"],
            batch["phyla_embeddings"],
        )

    velocity_labels = batch["batched_velocity"]
    num_leaves = batch["num_leaves"]
    original_trees = batch["original_trees"]

    preds = []
    labels = []
    matched_masks = []
    lengths = []
    encoder = BHVEncoder()

    for b_idx, vel_dict in enumerate(velocity_labels):
        model_masks = [int(m) for m in edge_split_masks[b_idx] if int(m) != 0]
        if not model_masks:
            continue
        real_max_bit = max(int(m).bit_length() for m in model_masks)
        full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0

        tree_obj = Tree(original_trees[b_idx])
        bhv_masks, bhv_lengths = encoder.return_BHV_encoding(tree_obj)
        bhv_len_map = {
            int(m): float(l)
            for m, l in zip(bhv_masks, bhv_lengths)
            if l is not None
        }
        model_length_map = {}
        for m_model in model_masks:
            length = bhv_len_map.get(m_model)
            if length is None and full_mask:
                length = bhv_len_map.get(full_mask ^ m_model)
            if length is None:
                continue
            model_length_map[m_model] = float(length)

        for original_vel, true_vel in vel_dict.items():
            vel = int(original_vel)
            if vel.bit_length() == real_max_bit + 1:
                vel = remove_bit(vel, int(num_leaves[b_idx]) - 1)
            elif vel.bit_length() > real_max_bit + 1:
                continue

            matched_vel = vel
            if matched_vel not in edge_split_masks[b_idx]:
                full_mask = (1 << real_max_bit) - 1
                complement_vel = full_mask ^ matched_vel
                if complement_vel in edge_split_masks[b_idx]:
                    matched_vel = complement_vel
                else:
                    continue

            n_bits = real_max_bit
            k_bits = int(matched_vel).bit_count()
            is_pendant = min(k_bits, n_bits - k_bits) == 1
            if is_pendant:
                continue

            split_list = [int(m) for m in edge_split_masks[b_idx]]
            edge_idx = split_list.index(matched_vel)
            edge_length = model_length_map.get(int(matched_vel), None)
            if edge_length is None or edge_length <= eps_len:
                continue
            preds.append(v_pred[b_idx, edge_idx, 0].detach().cpu())
            labels.append(torch.tensor(float(true_vel), dtype=torch.float32))
            matched_masks.append(int(matched_vel))
            lengths.append(torch.tensor(float(edge_length), dtype=torch.float32))

    if not preds:
        raise AssertionError("No supervised non-pendant velocity edges were matched.")

    return (
        torch.stack(preds).float(),
        torch.stack(labels).float(),
        matched_masks,
        torch.stack(lengths).float(),
    )


def _pearson_corr(x, y):
    xm = x - x.mean()
    ym = y - y.mean()
    denom = xm.norm() * ym.norm()
    if float(denom) <= 1e-12:
        return 1.0 if torch.allclose(x, y) else 0.0
    return float((xm * ym).sum() / denom)


def _spearman_corr(x, y):
    xr = torch.empty_like(x)
    yr = torch.empty_like(y)
    xr[torch.argsort(x)] = torch.arange(x.numel(), dtype=x.dtype)
    yr[torch.argsort(y)] = torch.arange(y.numel(), dtype=y.dtype)
    return _pearson_corr(xr, yr)


def _topk_mask_overlap(pred, true, masks, k):
    k = min(int(k), int(pred.numel()))
    if k <= 0:
        return 1.0
    pred_idx = torch.topk(pred.abs(), k=k).indices.tolist()
    true_idx = torch.topk(true.abs(), k=k).indices.tolist()
    pred_masks = {masks[i] for i in pred_idx}
    true_masks = {masks[i] for i in true_idx}
    return len(pred_masks & true_masks) / float(k)


def _dt_hit_and_candidates(lengths, velocity, eps_len=1e-8):
    valid = lengths > float(eps_len)
    neg = (velocity < 0.0) & valid
    if int(neg.sum()) == 0:
        return float("inf"), torch.empty(0, dtype=torch.float32), neg

    dt_candidates = lengths[neg] / (-velocity[neg])
    dt_hit = float(torch.min(dt_candidates))
    return dt_hit, dt_candidates, neg


def _first_hit_mask_set(dt_all, neg_idx, masks, tol=0.0): # Default to exact
    if int(neg_idx.numel()) == 0:
        return set()

    min_dt = float(torch.min(dt_all))
    # If tol is 0.0, this finds only the exact mathematical minimum
    mask_indices = torch.where(torch.abs(dt_all - min_dt) <= float(tol))[0]
    
    global_indices = neg_idx[mask_indices]
    return {int(masks[int(i)]) for i in global_indices.tolist()}


def _dt_by_mask(lengths, velocity, masks, eps_len=1e-8):
    dt_map = {}
    valid = lengths > float(eps_len)
    for idx, mask in enumerate(masks):
        if not bool(valid[idx]):
            dt_map[int(mask)] = float("inf")
            continue

        vel = float(velocity[idx])
        if vel < 0.0:
            dt_map[int(mask)] = float(lengths[idx] / (-velocity[idx]).clamp_min(1e-8))
        else:
            dt_map[int(mask)] = float("inf")
    return dt_map


def _format_dt(value):
    if math.isfinite(value):
        return f"{value:.6e}"
    return "inf"


def _format_first_hit_miss_details(metrics):
    missed_masks = sorted(metrics["true_first_masks"] - metrics["pred_first_masks"])
    if not missed_masks:
        return "none"

    details = []
    pred_dt_by_mask = metrics["pred_dt_by_mask"]
    true_dt_by_mask = metrics["true_dt_by_mask"]
    for mask in missed_masks:
        pred_dt = pred_dt_by_mask.get(mask, float("inf"))
        true_dt = true_dt_by_mask.get(mask, float("inf"))
        details.append(
            f"{mask}: pred_dt={_format_dt(pred_dt)}, true_dt={_format_dt(true_dt)}"
        )
    return "; ".join(details)


def _velocity_metrics(module, batch, topk=3):
    pred, true, masks, lengths = _gather_supervised_velocity(module, batch)
    mse = float(torch.mean((pred - true) ** 2))
    cosine = float(torch.nn.functional.cosine_similarity(pred, true, dim=0))
    pearson = _pearson_corr(pred, true)
    spearman = _spearman_corr(pred, true)
    pred_dt_by_mask = _dt_by_mask(lengths, pred, masks)
    true_dt_by_mask = _dt_by_mask(lengths, true, masks)

    # Tiny near-zero velocities are numerically unstable for sign comparisons.
    moving = true.abs() > 1e-3
    if int(moving.sum()) > 0:
        sign_acc = float((torch.sign(pred[moving]) == torch.sign(true[moving])).float().mean())
    else:
        sign_acc = 1.0

    pred_dt_hit, pred_dt_candidates, pred_neg = _dt_hit_and_candidates(lengths, pred)
    true_dt_hit, true_dt_candidates, true_neg = _dt_hit_and_candidates(lengths, true)
    both_neg = pred_neg & true_neg
    any_neg = pred_neg | true_neg

    if int(any_neg.sum()) > 0:
        dt_neg_jaccard = float(int(both_neg.sum()) / int(any_neg.sum()))
    else:
        dt_neg_jaccard = 1.0

    if int(both_neg.sum()) > 0:
        pred_dt_overlap = lengths[both_neg] / (-pred[both_neg])
        true_dt_overlap = lengths[both_neg] / (-true[both_neg])
        dt_candidates_mae = float(torch.mean(torch.abs(pred_dt_overlap - true_dt_overlap)))
        dt_candidates_rel_mae = float(
            torch.mean(
                torch.abs(pred_dt_overlap - true_dt_overlap)
                / torch.clamp(torch.abs(true_dt_overlap), min=1e-8)
            )
        )
    else:
        dt_candidates_mae = 0.0 if int(any_neg.sum()) == 0 else float("inf")
        dt_candidates_rel_mae = 0.0 if int(any_neg.sum()) == 0 else float("inf")

    pred_top_masks = set()
    true_top_masks = set()
    pred_first_masks = set()
    true_first_masks = set()
    dt_first_hit_recall = 1.0
    dt_first_hit_precision = 1.0
    pred_neg_idx = torch.where(pred_neg)[0]
    true_neg_idx = torch.where(true_neg)[0]
    if int(pred_neg_idx.numel()) == 0 and int(true_neg_idx.numel()) == 0:
        dt_first_hit_match = 1.0
        dt_topk_overlap = 1.0
    elif int(pred_neg_idx.numel()) == 0 or int(true_neg_idx.numel()) == 0:
        dt_first_hit_match = 0.0
        dt_topk_overlap = 0.0
        dt_first_hit_recall = 0.0
        dt_first_hit_precision = 0.0
    else:
        pred_dt_all = lengths[pred_neg_idx] / (-pred[pred_neg_idx]).clamp_min(1e-8)
        true_dt_all = lengths[true_neg_idx] / (-true[true_neg_idx]).clamp_min(1e-8)

        pred_order = pred_neg_idx[torch.argsort(pred_dt_all)]
        true_order = true_neg_idx[torch.argsort(true_dt_all)]
        pred_first_masks = _first_hit_mask_set(pred_dt_all, pred_neg_idx, masks, tol=_DT_FIRST_HIT_TOL)
        true_first_masks = _first_hit_mask_set(true_dt_all, true_neg_idx, masks, tol = _DT_FIRST_HIT_TOL)
        first_hit_overlap = pred_first_masks & true_first_masks
        dt_first_hit_recall = len(first_hit_overlap) / float(len(true_first_masks))
        dt_first_hit_precision = len(first_hit_overlap) / float(len(pred_first_masks))
        dt_first_hit_match = 1.0 if true_first_masks.issubset(pred_first_masks) else 0.0

        k = min(3, int(pred_order.numel()), int(true_order.numel()))
        pred_top_masks = {int(masks[int(i)]) for i in pred_order[:k].tolist()}
        true_top_masks = {int(masks[int(i)]) for i in true_order[:k].tolist()}
        dt_topk_overlap = len(pred_top_masks & true_top_masks) / float(k)

    if math.isfinite(pred_dt_hit) and math.isfinite(true_dt_hit):
        dt_hit_abs_err = abs(pred_dt_hit - true_dt_hit)
        dt_hit_rel_err = dt_hit_abs_err / max(abs(true_dt_hit), 1e-8)
    elif (not math.isfinite(pred_dt_hit)) and (not math.isfinite(true_dt_hit)):
        dt_hit_abs_err = 0.0
        dt_hit_rel_err = 0.0
    else:
        dt_hit_abs_err = float("inf")
        dt_hit_rel_err = float("inf")

    topk_overlap = _topk_mask_overlap(pred, true, masks, k=topk)
    return {
        "mse": mse,
        "cosine": cosine,
        "pearson": pearson,
        "spearman": spearman,
        "sign_acc": sign_acc,
        "topk_overlap": topk_overlap,
        "dt_hit_pred": pred_dt_hit,
        "dt_hit_true": true_dt_hit,
        "dt_hit_abs_err": dt_hit_abs_err,
        "dt_hit_rel_err": dt_hit_rel_err,
        "dt_neg_jaccard": dt_neg_jaccard,
        "dt_first_hit_match": dt_first_hit_match,
        "dt_first_hit_recall": dt_first_hit_recall,
        "dt_first_hit_precision": dt_first_hit_precision,
        "dt_first_hit_tol": _DT_FIRST_HIT_TOL,
        "dt_topk_overlap": dt_topk_overlap,
        "dt_candidates_mae": dt_candidates_mae,
        "dt_candidates_rel_mae": dt_candidates_rel_mae,
        "n_pred_dt_candidates": int(pred_dt_candidates.numel()),
        "n_true_dt_candidates": int(true_dt_candidates.numel()),
        "pred_first_masks": pred_first_masks,
        "true_first_masks": true_first_masks,
        "pred_dt_by_mask": pred_dt_by_mask,
        "true_dt_by_mask": true_dt_by_mask,
        'pred_top_masks': pred_top_masks if dt_topk_overlap > 0 else set(),
        'true_top_masks': true_top_masks if dt_topk_overlap > 0 else set(),
        "n_supervised_edges": int(pred.numel()),
    }


class _OptimizerProxy:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()


_ORIGINAL_TENSOR_TO = torch.Tensor.to


def _tensor_to_cpu_for_cuda(self, *args, **kwargs):
    if args:
        device = args[0]
        if isinstance(device, str) and device.startswith("cuda"):
            args = ("cpu",) + args[1:]
        elif isinstance(device, torch.device) and device.type == "cuda":
            args = (torch.device("cpu"),) + args[1:]

    if "device" in kwargs:
        device = kwargs["device"]
        if isinstance(device, str) and device.startswith("cuda"):
            kwargs["device"] = "cpu"
        elif isinstance(device, torch.device) and device.type == "cuda":
            kwargs["device"] = torch.device("cpu")

    return _ORIGINAL_TENSOR_TO(self, *args, **kwargs)


class TestTrainingSanity(unittest.TestCase):
    # def test_overfit_single_velocity_vector(self):
    #     random.seed(123)
    #     torch.manual_seed(123)
    #     device = torch.device("cpu")

    #     model = TreeDenoiserTokenGT(
    #         num_node_types=3,
    #         num_edge_types=2,
    #         embed_dim=64,
    #         n_layers=2,
    #         n_heads=4,
    #         output_dim=1,
    #         dropout=0.0,
    #         attention_dropout=0.0,
    #         activation_dropout=0.0,
    #         drop_path_rate=0.0,
    #         use_performer=False,
    #         performer_nb_features=None,
    #         performer_generalized_attention=False,
    #         layernorm_style="prenorm",
    #         tokenizer_lap_dim=8,
    #         tokenizer_lap_dropout=0.0,
    #         tokenizer_n_layers=2,
    #         phyla_dim=16,
    #     ).to(device)

    #     module = TrainingModule(
    #         model=model,
    #         dataset=MagicMock(),
    #         lr=1e-3,
    #         record=False,
    #         epochs=1,
    #         deepspeed=False,
    #         logger=None,
    #         velocity_loss_mode="plain",
    #         velocity_sign_eps=1e-3,
    #         velocity_event_weight = 0.0,
    #         verbose = True
    #     ).to(device)

    #     batch = _make_single_velocity_batch(
    #         tokenizer=model.tokenizer,
    #         n_leaves=10,
    #         seed=2024,
    #     )
    #     bootstrap_metrics = _velocity_metrics(module, batch, topk=3)
    #     self.assertGreaterEqual(
    #         bootstrap_metrics["n_supervised_edges"],
    #         6,
    #         "Not enough supervised internal edges to run a robust velocity-overfit sanity check.",
    #     )

    #     initial = _velocity_metrics(module, batch, topk=3)

    #     optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)
    #     best_metrics = dict(initial)
    #     best_state = copy.deepcopy(module.model.state_dict())
    #     max_steps = 1000

    #     for step in range(max_steps):
    #         module.train()
    #         optimizer.zero_grad(set_to_none=True)
    #         logs = module.step(batch, autoregressive=False)
    #         loss = logs["loss"]
    #         self.assertTrue(torch.isfinite(loss).item(), "Training loss became non-finite.")
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(module.model.parameters(), max_norm=1.0)
    #         optimizer.step()

    #         if (step + 1) % 10 == 0:
    #             probe = _velocity_metrics(module, batch, topk=3)
    #             has_strong_corr_probe = (
    #                 probe["cosine"] > 0.95
    #                 and probe["pearson"] > 0.95
    #                 and probe["spearman"] > 0.95
    #             )
    #             has_strong_corr_best = (
    #                 best_metrics["cosine"] > 0.95
    #                 and best_metrics["pearson"] > 0.95
    #                 and best_metrics["spearman"] > 0.95
    #             )
    #             if has_strong_corr_probe and has_strong_corr_best:
    #                 if (
    #                     probe["dt_first_hit_recall"] > best_metrics["dt_first_hit_recall"]
    #                     or (
    #                         probe["dt_first_hit_recall"] == best_metrics["dt_first_hit_recall"]
    #                         and (
    #                             probe["dt_hit_rel_err"] < best_metrics["dt_hit_rel_err"]
    #                             or (
    #                                 probe["dt_hit_rel_err"] == best_metrics["dt_hit_rel_err"]
    #                                 and (
    #                                     probe["dt_topk_overlap"] > best_metrics["dt_topk_overlap"]
    #                                     or (
    #                                         probe["dt_topk_overlap"] == best_metrics["dt_topk_overlap"]
    #                                         and probe["cosine"] >= best_metrics["cosine"]
    #                                     )
    #                                 )
    #                             )
    #                         )
    #                     )
    #                 ):
    #                     best_metrics = dict(probe)
    #                     best_state = copy.deepcopy(module.model.state_dict())
    #             elif (
    #                 probe["spearman"] > best_metrics["spearman"]
    #                 or (
    #                     probe["spearman"] == best_metrics["spearman"]
    #                     and (
    #                         probe["cosine"] > best_metrics["cosine"]
    #                         or (
    #                             probe["cosine"] == best_metrics["cosine"]
    #                             and probe["dt_first_hit_recall"] >= best_metrics["dt_first_hit_recall"]
    #                         )
    #                     )
    #                 )
    #             ):
    #                 best_metrics = dict(probe)
    #                 best_state = copy.deepcopy(module.model.state_dict())
    #             if (
    #                 probe["cosine"] > 0.99
    #                 and probe["pearson"] > 0.99
    #                 and probe["topk_overlap"] == 1.0
    #                 and probe["sign_acc"] > 0.90
    #                 and probe["dt_hit_rel_err"] < 0.15
    #                 and probe["dt_neg_jaccard"] >= 0.90
    #                 and probe["dt_first_hit_recall"] == 1.0
    #                 and probe["dt_topk_overlap"] >= 0.67
    #             ):
    #                 break

    #     module.model.load_state_dict(best_state)
    #     final = _velocity_metrics(module, batch, topk=3)
    #     print(final)

    #     self.assertLess(
    #         final["mse"],
    #         initial["mse"],
    #         "Overfit sanity check did not reduce velocity MSE.",
    #     )
    #     self.assertLess(
    #         final["mse"],
    #         max(3e-3, initial["mse"] * 0.1),
    #         f"MSE did not improve enough (initial={initial['mse']:.6f}, final={final['mse']:.6f})",
    #     )
    #     self.assertGreater(
    #         final["cosine"], 0.95, f"Cosine similarity too low: {final['cosine']:.6f}"
    #     )
    #     self.assertGreater(
    #         final["pearson"], 0.95, f"Pearson correlation too low: {final['pearson']:.6f}"
    #     )
    #     self.assertGreater(
    #         final["spearman"],
    #         0.95,
    #         f"Spearman correlation too low: {final['spearman']:.6f}",
    #     )
    #     self.assertGreater(
    #         final["sign_acc"], 0.95, f"Sign accuracy too low: {final['sign_acc']:.6f}"
    #     )
    #     self.assertEqual(
    #         final["topk_overlap"],
    #         1.0,
    #         f"Top-k velocity mask overlap not perfect: {final['topk_overlap']:.3f}",
    #     )
    #     self.assertLess(
    #         final["dt_hit_rel_err"],
    #         0.15,
    #         (
    #             f"dt_hit mismatch too large "
    #             f"(pred={final['dt_hit_pred']:.6e}, true={final['dt_hit_true']:.6e}, rel_err={final['dt_hit_rel_err']:.6f})"
    #         ),
    #     )
    #     self.assertGreaterEqual(
    #         final["dt_neg_jaccard"],
    #         0.90,
    #         f"Negative-velocity edge mismatch is too high (Jaccard={final['dt_neg_jaccard']:.3f})",
    #     )
    #     self.assertEqual(
    #         final["dt_first_hit_recall"],
    #         1.0,
    #         (
    #             "Did not recapitulate all true first-hit edge masks within the dt tolerance "
    #             f"{final['dt_first_hit_tol']:.2f} "
    #             f"(pred={sorted(final['pred_first_masks'])}, true={sorted(final['true_first_masks'])}, "
    #             f"missed={_format_first_hit_miss_details(final)})."
    #         ),
    #     )
    #     self.assertGreaterEqual(
    #         final["dt_topk_overlap"],
    #         0.66,
    #         f"dt candidate top-k overlap too low: {final['dt_topk_overlap']:.3f}",
    #     )

    # @patch.object(TreeDataset, "build_index", return_value=None)
    # def test_random_sanity_check_is_deterministic_for_tree_and_velocity(
    #     self, _mock_build_index
    # ):
    #     ds = TreeDataset(
    #         nexus_root="mock",
    #         mrbayes_root="mock",
    #         random_sanity_check=True,
    #         overfit_velocity_zero=True,
    #     )

    #     # random_sanity_check enforces fixed tree sources regardless tfiles input.
    #     real_one = ds.load_posterior_trees_from_tfiles([])[0]
    #     real_two = ds.load_posterior_trees_from_tfiles([])[0]
    #     self.assertEqual(real_one, real_two)

    #     rand_one = ds.sample_random_tree(real_one)
    #     rand_two = ds.sample_random_tree(real_one)
    #     self.assertEqual(rand_one, rand_two)

    #     # overfit_velocity_zero uses t=0.0, so sampled tree/velocity should be stable.
    #     sample_newick_1, velocity_1 = return_sampled_tree_orthant_velocity(
    #         rand_one, real_one, 0.0
    #     )
    #     sample_newick_2, velocity_2 = return_sampled_tree_orthant_velocity(
    #         rand_two, real_two, 0.0
    #     )

    #     self.assertEqual(sample_newick_1, sample_newick_2)
    #     self.assertEqual(set(velocity_1.keys()), set(velocity_2.keys()))
    #     for k in velocity_1:
    #         self.assertAlmostEqual(
    #             float(velocity_1[k]),
    #             float(velocity_2[k]),
    #             places=8,
    #             msg=f"Velocity mismatch for split {k}",
    #         )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_overfit_velocity_on_random_sanity_tree_pair(self, _mock_build_index):
        random.seed(321)
        torch.manual_seed(321)
        device = torch.device("cpu")

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )

        real_tree = ds.load_posterior_trees_from_tfiles([])[0]
        random_tree = ds.sample_random_tree(real_tree)
        # random_tree_small, real_tree_small = _prune_and_renumber_tree_pair(
        #     random_tree, real_tree, keep_leaves=8
        # )

        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=64,
            n_layers=2,
            n_heads=4,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=2,
            phyla_dim=16,
        ).to(device)

        module = TrainingModule(
            model=model,
            dataset=MagicMock(),
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            velocity_loss_mode="plain",
            velocity_sign_eps=1e-3,
            velocity_event_weight = 0.0,
            verbose=True,
        ).to(device)

        batch = _make_batch_from_tree_pair(
            tokenizer=model.tokenizer,
            start_tree=random_tree,
            target_tree=real_tree,
            time_point=0.0,
        )

        bootstrap_metrics = _velocity_metrics(module, batch, topk=3)
        self.assertGreaterEqual(
            bootstrap_metrics["n_supervised_edges"],
            3,
            "Not enough supervised internal edges in random_sanity_check pair.",
        )

        initial = _velocity_metrics(module, batch, topk=3)
        optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)
        best_metrics = dict(initial)
        best_state = copy.deepcopy(module.model.state_dict())

        for step in range(1000):
            module.train()
            optimizer.zero_grad(set_to_none=True)
            logs = module.step(batch, autoregressive=False)
            loss = logs["loss"]
            self.assertTrue(torch.isfinite(loss).item(), "Training loss became non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.model.parameters(), max_norm=1.0)
            optimizer.step()

            if (step + 1) % 10 == 0:
                probe = _velocity_metrics(module, batch, topk=3)
                has_strong_corr_probe = (
                    probe["cosine"] > 0.95
                    and probe["pearson"] > 0.95
                    and probe["spearman"] > 0.95
                )
                has_strong_corr_best = (
                    best_metrics["cosine"] > 0.95
                    and best_metrics["pearson"] > 0.95
                    and best_metrics["spearman"] > 0.95
                )
                if has_strong_corr_probe and has_strong_corr_best:
                    if (
                        probe["dt_first_hit_recall"] > best_metrics["dt_first_hit_recall"]
                        or (
                            probe["dt_first_hit_recall"] == best_metrics["dt_first_hit_recall"]
                            and (
                                probe["dt_hit_rel_err"] < best_metrics["dt_hit_rel_err"]
                                or (
                                    probe["dt_hit_rel_err"] == best_metrics["dt_hit_rel_err"]
                                    and (
                                        probe["dt_topk_overlap"] > best_metrics["dt_topk_overlap"]
                                        or (
                                            probe["dt_topk_overlap"] == best_metrics["dt_topk_overlap"]
                                            and probe["cosine"] >= best_metrics["cosine"]
                                        )
                                    )
                                )
                            )
                        )
                    ):
                        best_metrics = dict(probe)
                        best_state = copy.deepcopy(module.model.state_dict())
                elif (
                    probe["spearman"] > best_metrics["spearman"]
                    or (
                        probe["spearman"] == best_metrics["spearman"]
                        and (
                            probe["cosine"] > best_metrics["cosine"]
                            or (
                                probe["cosine"] == best_metrics["cosine"]
                                and probe["dt_first_hit_recall"] >= best_metrics["dt_first_hit_recall"]
                            )
                        )
                    )
                ):
                    best_metrics = dict(probe)
                    best_state = copy.deepcopy(module.model.state_dict())
                if (
                    probe["cosine"] > 0.99
                    and probe["pearson"] > 0.99
                    and probe["spearman"] > 0.95
                    and probe["topk_overlap"] == 1.0
                    and probe["sign_acc"] > 0.90
                    and probe["dt_hit_rel_err"] < 0.20
                    and probe["dt_neg_jaccard"] >= 0.85
                    and probe["dt_first_hit_recall"] == 1.0
                    and probe["dt_topk_overlap"] >= 0.67
                ):
                    break

        module.model.load_state_dict(best_state)
        final = _velocity_metrics(module, batch, topk=3)
        print(final)

        self.assertLess(final["mse"], initial["mse"])
        self.assertLess(
            final["mse"],
            max(3e-3, initial["mse"] * 0.1),
            f"MSE did not improve enough (initial={initial['mse']:.6f}, final={final['mse']:.6f})",
        )
        self.assertGreater(final["cosine"], 0.90)
        self.assertGreater(final["pearson"], 0.90)
        self.assertGreater(final["spearman"], 0.90)
        self.assertGreater(final["sign_acc"], 0.85)
        self.assertEqual(final["topk_overlap"], 1.0)
        self.assertLess(
            final["dt_hit_rel_err"],
            0.20,
            (
                f"dt_hit mismatch too large "
                f"(pred={final['dt_hit_pred']:.6e}, true={final['dt_hit_true']:.6e}, rel_err={final['dt_hit_rel_err']:.6f})"
            ),
        )
        self.assertGreaterEqual(
            final["dt_neg_jaccard"],
            0.85,
            f"Negative-velocity edge mismatch is too high (Jaccard={final['dt_neg_jaccard']:.3f})",
        )

        self.assertEqual(
            final["dt_first_hit_recall"],
            1.0,
            (
                "Did not recapitulate all true first-hit edge masks within the dt tolerance "
                f"{final['dt_first_hit_tol']:.2f} "
                f"(pred={sorted(final['pred_first_masks'])}, true={sorted(final['true_first_masks'])}, "
                f"missed={_format_first_hit_miss_details(final)})."
            ),
        )
        self.assertGreaterEqual(
            final["dt_topk_overlap"],
            0.67,
            f"dt candidate top-k overlap too low: {final['dt_topk_overlap']:.3f}",
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_training_step_with_autoregressive_still_converges_velocity(
        self, _mock_build_index
    ):
        random.seed(777)
        torch.manual_seed(777)
        device = torch.device("cpu")

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )
        real_tree = ds.load_posterior_trees_from_tfiles([])[0]
        random_tree = ds.sample_random_tree(real_tree)

        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=64,
            n_layers=2,
            n_heads=4,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=2,
            phyla_dim=16,
        ).to(device)

        dataset_stub = MagicMock()
        dataset_stub.msa_distance = True
        dataset_stub.chosen_tree = (0, 0, 1)

        module = TrainingModule(
            model=model,
            dataset=dataset_stub,
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            velocity_loss_mode="plain",
            velocity_sign_eps=1e-3,
            #I already weighed it by 100 in the loss, so this would make it 1000 which is a bit much
            training_step_velocity_weight=1,
            training_step_autoregressive_weight=0.1,
            velocity_dt_candidate_weight = 1,
            velocity_dt_hit_weight = 1,
            velocity_event_weight = 0.0,
        ).to(device)

        batch = _make_batch_from_tree_pair_with_autoregressive(
            tokenizer=model.tokenizer,
            start_tree=random_tree,
            target_tree=real_tree,
            time_point=0.0,
        )
        dataset_stub.chosen_tree = (0, int(batch["num_leaves"][0]), 1)

        bootstrap_metrics = _velocity_metrics(module, batch, topk=3)
        self.assertGreaterEqual(
            bootstrap_metrics["n_supervised_edges"],
            3,
            "Not enough supervised internal edges in random_sanity_check pair.",
        )

        initial = _velocity_metrics(module, batch, topk=3)
        best_metrics = dict(initial)
        best_state = copy.deepcopy(module.model.state_dict())

        optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)
        module.optimizers = MagicMock(return_value=_OptimizerProxy(optimizer))
        module.manual_backward = lambda loss: loss.backward()
        module.clip_gradients = lambda _opt, gradient_clip_val, gradient_clip_algorithm: torch.nn.utils.clip_grad_norm_(
            module.model.parameters(), max_norm=float(gradient_clip_val)
        )
        module.log = MagicMock()

        with patch.object(torch.Tensor, "to", _tensor_to_cpu_for_cuda):
            for step in range(1000):
                module.train()
                loss = module.training_step(batch, step)
                self.assertTrue(
                    torch.isfinite(loss).item(),
                    "training_step loss became non-finite.",
                )

                if (step + 1) % 5 == 0:
                    probe = _velocity_metrics(module, batch, topk=3)
                    has_strong_corr_probe = (
                        probe["cosine"] > 0.90
                        and probe["pearson"] > 0.90
                        and probe["spearman"] > 0.90
                    )
                    has_strong_corr_best = (
                        best_metrics["cosine"] > 0.90
                        and best_metrics["pearson"] > 0.90
                        and best_metrics["spearman"] > 0.90
                    )
                    if has_strong_corr_probe and has_strong_corr_best:
                        if (
                            probe["dt_first_hit_recall"] > best_metrics["dt_first_hit_recall"]
                            or (
                                probe["dt_first_hit_recall"] == best_metrics["dt_first_hit_recall"]
                                and (
                                    probe["dt_hit_rel_err"] < best_metrics["dt_hit_rel_err"]
                                    or (
                                        probe["dt_hit_rel_err"] == best_metrics["dt_hit_rel_err"]
                                        and (
                                            probe["dt_topk_overlap"] > best_metrics["dt_topk_overlap"]
                                            or (
                                                probe["dt_topk_overlap"] == best_metrics["dt_topk_overlap"]
                                                and probe["mse"] <= best_metrics["mse"]
                                            )
                                        )
                                    )
                                )
                            )
                        ):
                            best_metrics = dict(probe)
                            best_state = copy.deepcopy(module.model.state_dict())
                    elif (
                        probe["mse"] < best_metrics["mse"]
                        or (
                            probe["mse"] == best_metrics["mse"]
                            and (
                                probe["cosine"] > best_metrics["cosine"]
                                or (
                                    probe["cosine"] == best_metrics["cosine"]
                                    and probe["dt_first_hit_recall"] >= best_metrics["dt_first_hit_recall"]
                                )
                            )
                        )
                    ):
                        best_metrics = dict(probe)
                        best_state = copy.deepcopy(module.model.state_dict())

                    if (
                        probe["cosine"] > 0.95
                        and probe["pearson"] > 0.95
                        and probe["spearman"] > 0.90
                        and probe["topk_overlap"] == 1.0
                        and probe["sign_acc"] > 0.85
                        and probe["dt_hit_rel_err"] < 0.30
                        and probe["dt_neg_jaccard"] >= 0.75
                        and probe["dt_first_hit_recall"] == 1.0
                        and probe["dt_topk_overlap"] >= 0.50
                    ):
                        break

        module.model.load_state_dict(best_state)
        final = _velocity_metrics(module, batch, topk=3)
        print(final)

        self.assertLess(final["mse"], initial["mse"])
        self.assertLess(
            final["mse"],
            max(2e-2, initial["mse"] * 0.25),
            f"MSE did not improve enough (initial={initial['mse']:.6f}, final={final['mse']:.6f})",
        )
        self.assertGreater(final["cosine"], 0.90)
        self.assertGreater(final["pearson"], 0.90)
        self.assertGreater(final["spearman"], 0.90)
        self.assertGreater(final["sign_acc"], 0.80)
        self.assertEqual(final["topk_overlap"], 1.0)
        self.assertLess(
            final["dt_hit_rel_err"],
            0.30,
            (
                f"dt_hit mismatch too large "
                f"(pred={final['dt_hit_pred']:.6e}, true={final['dt_hit_true']:.6e}, rel_err={final['dt_hit_rel_err']:.6f})"
            ),
        )
        self.assertGreaterEqual(
            final["dt_neg_jaccard"],
            0.75,
            f"Negative-velocity edge mismatch is too high (Jaccard={final['dt_neg_jaccard']:.3f})",
        )
        self.assertGreaterEqual(
            final["dt_topk_overlap"],
            0.50,
            f"dt candidate top-k overlap too low: {final['dt_topk_overlap']:.3f}",
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_autoregressive_can_overfit_one_nonbinary_boundary_sequence(
        self, _mock_build_index
    ):
        random.seed(777)
        torch.manual_seed(777)
        device = torch.device("cpu")

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )
        real_tree = ds.load_posterior_trees_from_tfiles([])[0]
        random_tree = ds.sample_random_tree(real_tree)
        boundary_path = _select_random_nonbinary_boundary_path(
            random_tree,
            real_tree,
            seed=777,
        )

        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=64,
            n_layers=1,
            n_heads=4,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=4,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=1,
            phyla_dim=16,
        ).to(device)

        module = TrainingModule(
            model=model,
            dataset=MagicMock(),
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
        ).to(device)

        event_batches = [
            _make_autoregressive_event_batch(
                model.tokenizer,
                event["newick"],
                event["labels"],
                event_idx / 63.0,
            )
            for event_idx, event in enumerate(boundary_path["events"])
        ]

        optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)

        def evaluate_state():
            module.eval()
            event_details = []
            exact_events_ok = True
            for event_idx, event in enumerate(boundary_path["events"]):
                predicted, _ = _predict_autoregressive_event(
                    module,
                    model.tokenizer,
                    event["newick"],
                    event_idx / 63.0,
                )
                target = _target_merge_subsets_for_event(event["labels"])
                if predicted != target:
                    exact_events_ok = False
                    event_details.append(
                        f"event {event_idx}: predicted={predicted} target={target}"
                    )

            rollout_ok = False
            rollout_error = None
            try:
                _rollout_boundary_with_autoregressive_head(
                    self,
                    module,
                    model.tokenizer,
                    boundary_path,
                    random_tree,
                    real_tree,
                )
                rollout_ok = True
            except AssertionError as exc:
                rollout_error = str(exc)

            return {
                "exact_events_ok": exact_events_ok,
                "event_details": event_details,
                "rollout_ok": rollout_ok,
                "rollout_error": rollout_error,
            }

        best_state = copy.deepcopy(module.model.state_dict())
        best_eval = {
            "exact_events_ok": False,
            "event_details": ["evaluation not run"],
            "rollout_ok": False,
            "rollout_error": "evaluation not run",
        }
        best_score = (-1, -1)

        max_steps = 120
        for step in range(max_steps):
            module.train()
            optimizer.zero_grad()
            total_loss = None
            for batch in event_batches:
                logs = module.step(batch, autoregressive=True)
                total_loss = (
                    logs["loss"]
                    if total_loss is None
                    else total_loss + logs["loss"]
                )

            total_loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0 or step == 0:
                current_eval = evaluate_state()
                score = (
                    sum(
                        1
                        for event_idx, event in enumerate(boundary_path["events"])
                        if _target_merge_subsets_for_event(event["labels"])
                        == _predict_autoregressive_event(
                            module,
                            model.tokenizer,
                            event["newick"],
                            event_idx / 63.0,
                        )[0]
                    ),
                    int(current_eval["rollout_ok"]),
                )
                if score > best_score:
                    best_score = score
                    best_eval = current_eval
                    best_state = copy.deepcopy(module.model.state_dict())
                if current_eval["exact_events_ok"] and current_eval["rollout_ok"]:
                    break

        module.model.load_state_dict(best_state)
        final_eval = evaluate_state()

        self.assertTrue(
            final_eval["exact_events_ok"],
            "Autoregressive head did not overfit the selected non-binary boundary events. "
            + " | ".join(final_eval["event_details"]),
        )
        self.assertTrue(
            final_eval["rollout_ok"],
            "Autoregressive rollout did not reproduce the selected boundary sequence. "
            + str(final_eval["rollout_error"]),
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_velocity_and_autoregressive_can_overfit_first_boundary_transition(
        self, _mock_build_index
    ):
        random.seed(777)
        torch.manual_seed(777)
        device = torch.device("cpu")

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )
        real_tree = ds.load_posterior_trees_from_tfiles([])[0]
        random_tree = ds.sample_random_tree(real_tree)
        random_tree, real_tree = _prune_and_renumber_tree_pair(
            random_tree,
            real_tree,
            keep_leaves=12,
        )
        boundary_path = return_tree_boundary_merge_paths(random_tree, real_tree)[0]
        self.assertTrue(
            boundary_path["events"],
            "The first boundary on the sanity tree pair had no merge events.",
        )

        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=64,
            n_layers=2,
            n_heads=4,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=2,
            phyla_dim=16,
        ).to(device)

        dataset_stub = MagicMock()
        dataset_stub.msa_distance = True
        dataset_stub.chosen_tree = (0, 0, 1)

        module = TrainingModule(
            model=model,
            dataset=dataset_stub,
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            velocity_loss_mode="plain",
            velocity_sign_eps=1e-3,
            training_step_velocity_weight=1,
            training_step_autoregressive_weight=0.1,
            velocity_dt_candidate_weight=1,
            velocity_dt_hit_weight=1,
            velocity_event_weight=0.0,
        ).to(device)

        velocity_batch = _make_batch_from_tree_pair(
            tokenizer=model.tokenizer,
            start_tree=random_tree,
            target_tree=real_tree,
            time_point=0.0,
        )
        dataset_stub.chosen_tree = (0, int(velocity_batch["num_leaves"][0]), 1)

        initial_velocity = _velocity_metrics(module, velocity_batch, topk=3)
        self.assertTrue(
            math.isfinite(initial_velocity["dt_hit_true"]),
            "The sanity tree pair did not have a finite first boundary hit time.",
        )

        event_batches = [
            _make_autoregressive_event_batch(
                model.tokenizer,
                event["newick"],
                event["labels"],
                event_idx / 63.0,
            )
            for event_idx, event in enumerate(boundary_path["events"])
        ]

        optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)

        def evaluate_state():
            module.eval()
            velocity_eval = _velocity_metrics(module, velocity_batch, topk=3)

            exact_events_ok = True
            event_details = []
            for event_idx, event in enumerate(boundary_path["events"]):
                predicted, _ = _predict_autoregressive_event(
                    module,
                    model.tokenizer,
                    event["newick"],
                    event_idx / 63.0,
                )
                target = _target_merge_subsets_for_event(event["labels"])
                if predicted != target:
                    exact_events_ok = False
                    event_details.append(
                        f"event {event_idx}: predicted={predicted} target={target}"
                    )

            sample_eval = _sample_to_first_boundary(
                module,
                random_tree,
                initial_velocity["dt_hit_true"],
                boundary_path,
            )
            return {
                "velocity": velocity_eval,
                "exact_events_ok": exact_events_ok,
                "event_details": event_details,
                "sample": sample_eval,
            }

        def score_state(eval_state):
            velocity_eval = eval_state["velocity"]
            rf_zero = int(eval_state["sample"]["rf_norm"] == 0.0)
            exact_events = int(eval_state["exact_events_ok"])
            dt_recall = float(velocity_eval["dt_first_hit_recall"])
            dt_rel_err = velocity_eval["dt_hit_rel_err"]
            mse = velocity_eval["mse"]
            dt_rel_err_score = -dt_rel_err if math.isfinite(dt_rel_err) else float("-inf")
            return (rf_zero, exact_events, dt_recall, dt_rel_err_score, -mse)

        best_state = copy.deepcopy(module.model.state_dict())
        best_eval = None
        best_score = (-1, -1, -1.0, float("-inf"), float("-inf"))
        max_steps = 400

        for step in range(max_steps):
            module.train()
            optimizer.zero_grad()

            velocity_logs = module.step(velocity_batch, autoregressive=False)
            autoregressive_losses = [
                module.step(batch, autoregressive=True)["loss"]
                for batch in event_batches
            ]
            autoregressive_loss = torch.stack(autoregressive_losses).mean()
            total_loss = velocity_logs["loss"] + 0.1 * autoregressive_loss
            total_loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0 or step == 0:
                current_eval = evaluate_state()
                current_score = score_state(current_eval)
                if current_score > best_score:
                    best_score = current_score
                    best_eval = current_eval
                    best_state = copy.deepcopy(module.model.state_dict())

                if (
                    current_eval["sample"]["rf_norm"] == 0.0
                    and current_eval["exact_events_ok"]
                    and current_eval["velocity"]["dt_first_hit_recall"] == 1.0
                    and current_eval["velocity"]["dt_hit_rel_err"] < 0.30
                ):
                    break

        module.model.load_state_dict(best_state)
        if best_eval is None:
            best_eval = evaluate_state()
        final_eval = evaluate_state()

        self.assertLess(
            final_eval["velocity"]["mse"],
            initial_velocity["mse"],
            "Velocity supervision did not improve on the first-boundary sanity pair.",
        )
        self.assertEqual(
            final_eval["velocity"]["dt_first_hit_recall"],
            1.0,
            (
                "The trained velocity head did not recover the full first-hit set at t=0. "
                f"pred={sorted(final_eval['velocity']['pred_first_masks'])}, "
                f"true={sorted(final_eval['velocity']['true_first_masks'])}"
            ),
        )
        self.assertLess(
            final_eval["velocity"]["dt_hit_rel_err"],
            0.30,
            (
                "The trained velocity head did not localize the first boundary accurately enough. "
                f"pred={final_eval['velocity']['dt_hit_pred']:.6e}, "
                f"true={final_eval['velocity']['dt_hit_true']:.6e}, "
                f"rel_err={final_eval['velocity']['dt_hit_rel_err']:.6f}"
            ),
        )
        self.assertTrue(
            final_eval["exact_events_ok"],
            "Autoregressive supervision did not overfit the first boundary events. "
            + " | ".join(final_eval["event_details"]),
        )
        self.assertEqual(
            final_eval["sample"]["rf_norm"],
            0.0,
            (
                "Sampling to the true first-hit time did not recover the oracle post-boundary tree. "
                f"rf_norm={final_eval['sample']['rf_norm']:.6f}, "
                f"sampled={final_eval['sample']['sampled_tree']}, "
                f"target={final_eval['sample']['target_tree']}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
