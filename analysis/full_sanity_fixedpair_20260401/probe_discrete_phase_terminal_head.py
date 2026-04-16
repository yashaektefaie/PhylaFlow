import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path("/home/yektefai/PhylaFlow")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.full_sanity_fixedpair_20260401.probe_current_mainline_ckpt import (
    build_dataset,
    build_module,
    build_training_sampling_pair,
    set_seed,
)
from run.TrainingModule import (
    _align_model_outputs_to_tree_context,
    _build_autoregressive_replay_batch,
    _build_legacy_velocity_oracle_sample,
    _build_pair_oracle_orthant_velocity_label_map,
    _build_velocity_replay_batch,
    _plan_autoregressive_boundary_merges,
    _predict_first_hit_mask_with_fallback,
    _split_multi_label_training_events,
    _topology_key,
    _tree_to_model_split_lengths,
)
from utils.bhv_movie import build_tree_from_splits
from utils.bhv_utils import BHVEncoder, get_structural_polytomy_groups_from_newick, return_tree_boundary_merge_paths
from utils.metric_utils import calculate_norm_rf
from utils.random_tree import Tree
from utils.utils import has_polytomy_fast, remove_bit


CONFIG = ROOT / (
    "configs/"
    "randomstart20_singlepair_mainline_replay010_s50_f20_velonly_smallbank_"
    "basefh_nocap_controlreplay_fullchain_reuse_allbank_oracleorthantfh_20260416.yaml"
)
CKPT = ROOT / (
    "checkpoints/full_sanity_fixedpair_20260401/"
    "randomstart20_singlepair_mainline_replay010_s50_f20_velonly_smallbank_"
    "basefh_nocap_controlreplay_fullchain_reuse_allbank_oracleorthantfh_20260416/"
    "2026-04-16_03-04-57/epoch=199-step=000400.ckpt"
)
TRACE_COMPARE = ROOT / (
    "analysis/full_sanity_fixedpair_20260401/"
    "randomstart20_singlepair_velonly_oracleorthantfh_trace_compare_100_200_300_400_20260416.json"
)
OUT = ROOT / (
    "analysis/full_sanity_fixedpair_20260401/"
    "randomstart20_singlepair_discrete_phase_terminal_head_probe_20260416.json"
)


def target_splits_from_velocity_sample(sample):
    tree_obj = Tree(sample["newick_tree"])
    encoder = BHVEncoder()
    masks, lengths = encoder.return_BHV_encoding(tree_obj)
    len_map = {int(m): float(l) for m, l in zip(masks, lengths) if l is not None}
    model_masks = [
        int(m)
        for m, l in zip(masks, lengths)
        if l is not None and float(l) > 1e-8 and int(m) != 0
    ]
    if not model_masks:
        return []
    real_max_bit = max(int(m).bit_length() for m in model_masks)
    full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0
    taus = []
    matched = []
    for orig_mask, vel in sample.get("velocity", {}).items():
        vel = float(vel)
        mask = int(orig_mask)
        if mask.bit_length() == real_max_bit + 1:
            mask = remove_bit(mask, int(sample["num_leaves"]) - 1)
        elif mask.bit_length() > real_max_bit + 1:
            continue
        matched_mask = mask
        if matched_mask not in model_masks:
            comp = full_mask ^ matched_mask
            if comp in model_masks:
                matched_mask = comp
            else:
                continue
        k_bits = int(matched_mask).bit_count()
        if min(k_bits, real_max_bit - k_bits) == 1:
            continue
        length = len_map.get(int(matched_mask))
        if length is None and full_mask:
            length = len_map.get(full_mask ^ int(matched_mask))
        if length is None or float(length) <= 1e-8 or vel >= -1e-3:
            continue
        taus.append(float(length) / max(-vel, 1e-3))
        matched.append(int(matched_mask))
    if not taus:
        return []
    tau_min = min(taus)
    return sorted(m for m, t in zip(matched, taus) if abs(t - tau_min) <= 1e-3)


def build_direct_set_loss(module, sample, target_splits):
    batch = _build_velocity_replay_batch(module, [sample])
    _, edge_split_masks, _, first_hit_logits, _, _ = module.forward(
        batch["tokenized_trees"],
        batch["batched_time"],
        batch["phyla_embeddings"],
    )
    split_masks = [int(m) for m in edge_split_masks[0]]
    model_masks = [m for m in split_masks if m != 0]
    real_max_bit = max(int(m).bit_length() for m in model_masks)
    full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0

    tree_obj = Tree(sample["newick_tree"])
    encoder = BHVEncoder()
    bhv_masks, bhv_lengths = encoder.return_BHV_encoding(tree_obj)
    bhv_len_map = {
        int(m): float(l)
        for m, l in zip(bhv_masks, bhv_lengths)
        if l is not None
    }

    logits = []
    targets = []
    matched_masks = []
    target_set = {int(x) for x in target_splits}
    for edge_idx, mask in enumerate(split_masks):
        if mask == 0:
            continue
        k_bits = int(mask).bit_count()
        if min(k_bits, real_max_bit - k_bits) == 1:
            continue
        edge_length = bhv_len_map.get(int(mask))
        if edge_length is None and full_mask:
            edge_length = bhv_len_map.get(full_mask ^ int(mask))
        if edge_length is None or float(edge_length) <= 1e-8:
            continue
        logits.append(first_hit_logits[0, edge_idx, 0])
        matched_masks.append(int(mask))
        targets.append(1.0 if int(mask) in target_set else 0.0)

    logits = torch.stack(logits)
    targets = torch.tensor(targets, device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    pred_mask = probs > 0.5
    pred_splits = sorted(
        matched_masks[i] for i in range(len(matched_masks)) if bool(pred_mask[i].item())
    )
    pred_set = set(pred_splits)
    inter = len(pred_set & target_set)
    union = len(pred_set | target_set) if (pred_set or target_set) else 1
    return loss, {
        "pred_splits": pred_splits,
        "target_splits": sorted(target_set),
        "exact": pred_set == target_set,
        "precision": inter / len(pred_set) if pred_set else 0.0,
        "recall": inter / len(target_set) if target_set else 1.0,
        "jaccard": inter / union,
        "loss_raw": float(loss.detach().cpu().item()),
    }


def build_path_ar_samples(path, target_tree, phase_idx):
    events = []
    for ev in path.get("events", []):
        labels = list(ev.get("labels", []))
        if labels:
            events.append(
                {
                    "newick": str(ev["newick"]),
                    "labels": labels,
                }
            )
    split_events = _split_multi_label_training_events(events)
    out = []
    for ev in split_events:
        out.append(
            {
                "newick": str(ev["newick"]),
                "target_tree": target_tree,
                "labels": list(ev["labels"]),
                "stop_after_merge": False,
                "time": float(phase_idx),
            }
        )
    return out


def ar_eval(module, ar_batch):
    module.eval()
    with torch.no_grad():
        logs = module.step(ar_batch, eval=True, autoregressive=True)
    out = {"loss": float(logs["loss"].detach().cpu().item())}
    for key in [
        "autoregressive_stats/subset_size_accuracy",
        "autoregressive_stats/stop_after_merge_accuracy",
        "autoregressive_stats/stop_after_merge_target_rate",
        "autoregressive_stats/stop_after_merge_pred_rate",
        "autoregressive_stats/merge/precision",
        "autoregressive_stats/merge/recall",
        "autoregressive_stats/merge/f1",
    ]:
        if key in logs:
            out[key] = float(logs[key].detach().cpu().item())
    return out


def _match_planned_group(logit_outputs, planned):
    planned_splits = tuple(int(x) for x in planned["splits_represented"])
    for group in logit_outputs:
        if tuple(int(x) for x in group["splits_represented"]) == planned_splits:
            return group
    return None


class TerminalHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        hidden = max(32, min(256, in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def extract_tree_phase_feature(module, newick, phase, phyla_embeddings=None):
    _, n_leaves, _ = _tree_to_model_split_lengths(module, newick)
    tokenized = module.model.tokenizer([newick])
    velocity, edge_splits, _, first_hit_logits, boundary_vanish_logits, edge_features = module.forward(
        tokenized, float(phase), phyla_embeddings
    )
    aligned = _align_model_outputs_to_tree_context(
        module,
        newick,
        n_leaves,
        edge_splits[0],
        velocity[0, :, 0],
        first_hit_logits_tree=None if first_hit_logits is None else first_hit_logits[0, :, 0],
        boundary_vanish_logits_tree=None if boundary_vanish_logits is None else boundary_vanish_logits[0, :, 0],
        edge_features_tree=None if edge_features is None else edge_features[0],
        eps_len=1e-8,
    )
    aligned_first_hit_logits = module._compute_first_hit_logits(
        aligned["first_hit_logits"],
        lengths=aligned["lengths"],
        velocities=aligned["velocities"],
        edge_features=aligned["edge_features"],
        group_sizes=[int(aligned["lengths"].numel())],
    )

    def pooled_stats(x):
        if x is None or x.numel() == 0:
            dev = module.device
            return torch.zeros(4, device=dev, dtype=torch.float32)
        flat = x.reshape(-1).to(dtype=torch.float32)
        return torch.stack(
            [
                flat.mean(),
                flat.min(),
                flat.max(),
                torch.linalg.vector_norm(flat),
            ],
            dim=0,
        )

    supervised = aligned["supervised_mask"].to(dtype=torch.float32)
    count_feat = torch.tensor(
        [float(aligned["lengths"].numel()), float(supervised.sum().item())],
        device=module.device,
        dtype=torch.float32,
    )
    return torch.cat(
        [
            pooled_stats(aligned["lengths"]),
            pooled_stats(aligned["velocities"]),
            pooled_stats(aligned_first_hit_logits),
            pooled_stats(aligned["boundary_vanish_logits"]),
            count_feat,
        ],
        dim=0,
    )


def eval_terminal(module, terminal_head, terminal_samples):
    module.eval()
    terminal_head.eval()
    infos = {}
    exact = 0
    losses = []
    with torch.no_grad():
        for name, sample in terminal_samples.items():
            feature = extract_tree_phase_feature(
                module, sample["newick"], sample["phase"], phyla_embeddings=None
            ).unsqueeze(0)
            logit = terminal_head(feature)
            target = torch.tensor(
                [float(sample["label"])], device=logit.device, dtype=logit.dtype
            )
            loss = F.binary_cross_entropy_with_logits(logit, target)
            prob = float(torch.sigmoid(logit).item())
            pred = prob > 0.5
            losses.append(loss.detach())
            infos[name] = {
                "phase": int(sample["phase"]),
                "label": bool(sample["label"]),
                "pred": bool(pred),
                "prob": prob,
                "exact": bool(pred == bool(sample["label"])),
                "loss_raw": float(loss.item()),
            }
            exact += int(pred == bool(sample["label"]))
    mean_loss = (
        float(torch.stack(list(losses)).mean().item()) if losses else 0.0
    )
    return infos, int(exact), mean_loss


def discrete_phase_rollout(
    module,
    terminal_head,
    start_tree,
    target_tree,
    phyla_embeddings=None,
    max_phases=6,
    max_events=128,
):
    current_newick = str(start_tree)
    phase = 0
    n_events = 0
    trace = {"velocity": [], "autoregressive": [], "terminal": []}
    global_stop = False

    while phase < int(max_phases) and n_events < int(max_events):
        td, n_leaves, mapping = _tree_to_model_split_lengths(module, current_newick)
        tokenized = module.model.tokenizer([current_newick])
        with torch.inference_mode():
            velocity, edge_splits, _, first_hit_logits, boundary_vanish_logits, edge_features = module.forward(
                tokenized, float(phase), phyla_embeddings
            )

        aligned = _align_model_outputs_to_tree_context(
            module,
            current_newick,
            n_leaves,
            edge_splits[0],
            velocity[0, :, 0],
            first_hit_logits_tree=None if first_hit_logits is None else first_hit_logits[0, :, 0],
            boundary_vanish_logits_tree=None if boundary_vanish_logits is None else boundary_vanish_logits[0, :, 0],
            edge_features_tree=None if edge_features is None else edge_features[0],
            eps_len=1e-8,
        )
        aligned_first_hit_logits = module._compute_first_hit_logits(
            aligned["first_hit_logits"],
            lengths=aligned["lengths"],
            velocities=aligned["velocities"],
            edge_features=aligned["edge_features"],
            group_sizes=[int(aligned["lengths"].numel())],
        )

        lengths = aligned["lengths"].detach().cpu().numpy().astype("float64")
        velocities = aligned["velocities"].detach().cpu().numpy().astype("float64")
        supervised_mask = aligned["supervised_mask"].detach().cpu().numpy().astype(bool)
        masks = [int(x) for x in aligned["aligned_model_masks"]]
        first_logits = aligned_first_hit_logits.detach().cpu().numpy().astype("float64")
        candidate_mask = supervised_mask & (lengths > 1e-8)
        pred_mask, raw_count, used_fallback = _predict_first_hit_mask_with_fallback(
            first_logits,
            candidate_mask,
            max_edges=-1,
            fallback_threshold=-1,
            fallback_top_k=-1,
        )
        pred_neg = pred_mask & (velocities < 0.0) & (lengths > 1e-8)
        if not pred_neg.any():
            trace["velocity"].append(
                {
                    "phase_idx": int(phase),
                    "time_input": float(phase),
                    "newick_tree": current_newick,
                    "rf_to_target": float(calculate_norm_rf(current_newick, target_tree)),
                    "event": "no_predicted_negative_edges",
                    "predicted_masks": [],
                }
            )
            break

        dt_target = float((lengths[pred_neg] / (-velocities[pred_neg]).clip(min=1e-8)).max())
        collapse_mask = pred_mask.copy()
        L_new = lengths + dt_target * velocities
        L_new[collapse_mask] = 0.0
        blocked = supervised_mask & (~collapse_mask)
        if blocked.any():
            L_new[blocked] = L_new[blocked].clip(min=1e-7)

        td2 = {
            int(mask): float(length)
            for mask, length in zip(masks, L_new)
            if float(length) > 1e-8
        }
        current_newick = build_tree_from_splits(
            list(td2.keys()),
            td2,
            n_leaves,
            root_leaf=n_leaves - 1,
            mapping=mapping,
        )[1]
        trace["velocity"].append(
            {
                "phase_idx": int(phase),
                "time_input": float(phase),
                "dt_target": float(dt_target),
                "newick_tree": current_newick,
                "rf_to_target": float(calculate_norm_rf(current_newick, target_tree)),
                "predicted_masks": [masks[i] for i, on in enumerate(pred_mask.tolist()) if on],
            }
        )

        merges_this_phase = 0
        phase_exhausted = False
        while has_polytomy_fast(current_newick, unrooted_ok=False) and n_events < int(max_events):
            tokenized_trees = module.model.tokenizer([current_newick])
            component_groups = [get_structural_polytomy_groups_from_newick(current_newick)]
            with torch.inference_mode():
                logit_outputs = module.forward(
                    tokenized_trees,
                    torch.tensor([float(phase)], dtype=torch.float32, device=module.device),
                    phyla_embeddings,
                    autoregressive=True,
                    autoregressive_component_groups=component_groups,
                )
            td_ar, n_ar, m_ar = _tree_to_model_split_lengths(module, current_newick)
            planned_merges = _plan_autoregressive_boundary_merges(logit_outputs, td_ar.keys())
            if planned_merges:
                planned_merges = planned_merges[:1]
            if not planned_merges:
                trace["autoregressive"].append(
                    {
                        "phase_idx": int(phase),
                        "time_input": float(phase),
                        "newick": current_newick,
                        "rf_to_target": float(calculate_norm_rf(current_newick, target_tree)),
                        "planned_merge_count": 0,
                        "selected_result_split": None,
                        "pred_stop_prob": None,
                    }
                )
                phase_exhausted = True
                break

            planned = planned_merges[0]
            matched_group = _match_planned_group(logit_outputs, planned)
            stop_prob = None
            if matched_group is not None and matched_group.get("stop_after_merge_logit") is not None:
                stop_prob = float(torch.sigmoid(matched_group["stop_after_merge_logit"]).detach().cpu().item())

            subset, new_split = planned["subsets"][0]
            td_ar[int(new_split)] = 1e-3
            source_newick = current_newick
            current_newick = build_tree_from_splits(
                list(td_ar.keys()),
                td_ar,
                n_ar,
                root_leaf=n_ar - 1,
                mapping=m_ar,
            )[1]
            n_events += 1
            merges_this_phase += 1
            trace["autoregressive"].append(
                {
                    "phase_idx": int(phase),
                    "time_input": float(phase),
                    "source_newick": source_newick,
                    "newick": current_newick,
                    "rf_to_target": float(calculate_norm_rf(current_newick, target_tree)),
                    "planned_merge_count": int(len(planned_merges)),
                    "selected_result_split": int(new_split),
                    "pred_stop_prob": stop_prob,
                }
            )

        if global_stop:
            break

        if not has_polytomy_fast(current_newick, unrooted_ok=False) or phase_exhausted:
            with torch.inference_mode():
                feature = extract_tree_phase_feature(
                    module, current_newick, phase, phyla_embeddings=phyla_embeddings
                ).unsqueeze(0)
                term_logit = terminal_head(feature)
                term_prob = float(torch.sigmoid(term_logit).item())
            trace["terminal"].append(
                {
                    "phase_idx": int(phase),
                    "time_input": float(phase),
                    "newick": current_newick,
                    "rf_to_target": float(calculate_norm_rf(current_newick, target_tree)),
                    "pred_terminal_prob": term_prob,
                }
            )
            if term_prob > 0.5:
                global_stop = True
                break

        phase += 1

    return {
        "final_tree": current_newick,
        "final_rf": float(calculate_norm_rf(current_newick, target_tree)),
        "num_velocity_states": int(len(trace["velocity"])),
        "num_ar_states": int(len(trace["autoregressive"])),
        "num_terminal_queries": int(len(trace["terminal"])),
        "trace": trace,
    }


def main():
    config = yaml.safe_load(CONFIG.read_text())
    set_seed(int(config["trainer"].get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = build_dataset(config)
    module = build_module(config, dataset, CKPT, device)
    module.record = False
    module.autoregressive_use_time = True
    module.autoregressive_stop_after_merge_weight = 0.0
    pair = build_training_sampling_pair(dataset)
    num_leaves = pair.get("n_leaves", 20)
    trace_compare = json.loads(TRACE_COMPARE.read_text())

    canonical_map, _ = _build_pair_oracle_orthant_velocity_label_map(
        pair["start_tree"], pair["target_tree"]
    )
    start_key = _topology_key(pair["start_tree"])
    a1_key = tuple(trace_compare["checkpoints"]["400"]["A1_post"]["topology_key"])
    start_canonical = dict(canonical_map[start_key])
    a1_canonical = dict(canonical_map[a1_key])

    boundary_paths = return_tree_boundary_merge_paths(pair["start_tree"], pair["target_tree"])
    path0, path1, path2 = boundary_paths
    third_orthant_source = str(path1["end_newick"])
    o2_velocity = _build_legacy_velocity_oracle_sample(
        third_orthant_source,
        pair["target_tree"],
        timepoint=2.0,
        num_leaves=num_leaves,
    )

    velocity_samples = []
    velocity_samples.append(
        (
            "O0",
            "start",
            {
                "newick_tree": pair["start_tree"],
                "target_tree": pair["target_tree"],
                "timepoint": 0.0,
                "num_leaves": num_leaves,
                "velocity": dict(start_canonical["velocity"]),
                "velocity_next_boundary_tree": start_canonical.get("velocity_next_boundary_tree"),
            },
            target_splits_from_velocity_sample({**start_canonical, "num_leaves": num_leaves}),
        )
    )
    for ck in ["100", "200", "300", "400"]:
        st = trace_compare["checkpoints"][ck]["A1_post"]
        velocity_samples.append(
            (
                "A1",
                ck,
                {
                    "newick_tree": st["newick"],
                    "target_tree": pair["target_tree"],
                    "timepoint": 1.0,
                    "num_leaves": num_leaves,
                    "velocity": dict(a1_canonical["velocity"]),
                    "velocity_next_boundary_tree": a1_canonical.get("velocity_next_boundary_tree"),
                },
                target_splits_from_velocity_sample({**a1_canonical, "num_leaves": num_leaves}),
            )
        )
    velocity_samples.append(
        (
            "O2",
            "oracle",
            {
                "newick_tree": third_orthant_source,
                "target_tree": pair["target_tree"],
                "timepoint": 2.0,
                "num_leaves": num_leaves,
                "velocity": dict(o2_velocity["velocity"]),
                "velocity_next_boundary_tree": o2_velocity.get("velocity_next_boundary_tree"),
            },
            target_splits_from_velocity_sample({**o2_velocity, "num_leaves": num_leaves}),
        )
    )

    ar_samples = []
    ar_samples.extend(build_path_ar_samples(path0, pair["target_tree"], phase_idx=0))
    ar_samples.extend(build_path_ar_samples(path1, pair["target_tree"], phase_idx=1))
    ar_samples.extend(build_path_ar_samples(path2, pair["target_tree"], phase_idx=2))
    ar_batch = _build_autoregressive_replay_batch(module, ar_samples)

    terminal_samples = {
        "path0_end": {"newick": str(path0["end_newick"]), "phase": 0, "label": False},
        "path1_end": {"newick": str(path1["end_newick"]), "phase": 1, "label": False},
        "path2_end": {"newick": str(path2["end_newick"]), "phase": 2, "label": True},
    }
    with torch.no_grad():
        feature_dim = int(
            extract_tree_phase_feature(
                module, terminal_samples["path0_end"]["newick"], 0
            ).numel()
        )
    terminal_head = TerminalHead(feature_dim).to(module.device)

    def eval_velocity():
        module.eval()
        infos = {"O0": {}, "A1": {}, "O2": {}}
        exact_count = 0
        for family, name, sample, target in velocity_samples:
            _, info = build_direct_set_loss(module, sample, target)
            infos[family][name] = info
            exact_count += int(info["exact"])
        return infos, exact_count

    before_velocity_infos, before_velocity_exact = eval_velocity()
    before_ar = ar_eval(module, ar_batch)
    before_terminal_infos, before_terminal_exact, before_terminal_loss = eval_terminal(
        module, terminal_head, terminal_samples
    )
    before_rollout = discrete_phase_rollout(
        module, terminal_head, pair["start_tree"], pair["target_tree"]
    )

    best_state = copy.deepcopy(module.model.state_dict())
    best_terminal_state = copy.deepcopy(terminal_head.state_dict())
    best_payload = None
    best_score = (-before_velocity_exact, -before_terminal_exact, before_rollout["final_rf"])
    trajectory = []

    optimizer = torch.optim.Adam(
        list(module.model.parameters()) + list(terminal_head.parameters()), lr=5e-3
    )
    for step in range(1, 101):
        module.train()
        terminal_head.train()
        optimizer.zero_grad(set_to_none=True)
        vel_losses = []
        for _, _, sample, target in velocity_samples:
            loss, _ = build_direct_set_loss(module, sample, target)
            vel_losses.append(loss)
        vel_loss = torch.stack(vel_losses).mean()
        term_losses = []
        for sample in terminal_samples.values():
            feature = extract_tree_phase_feature(
                module, sample["newick"], sample["phase"]
            ).unsqueeze(0)
            logit = terminal_head(feature)
            target = torch.tensor(
                [float(sample["label"])], device=logit.device, dtype=logit.dtype
            )
            term_losses.append(F.binary_cross_entropy_with_logits(logit, target))
        terminal_loss = torch.stack(term_losses).mean()
        ar_logs = module.step(ar_batch, autoregressive=True)
        ar_loss = ar_logs["loss"]
        loss = vel_loss + terminal_loss + ar_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0 or step == 100:
            velocity_infos, velocity_exact = eval_velocity()
            terminal_infos, terminal_exact, terminal_loss_eval = eval_terminal(
                module, terminal_head, terminal_samples
            )
            ar_info = ar_eval(module, ar_batch)
            rollout = discrete_phase_rollout(
                module, terminal_head, pair["start_tree"], pair["target_tree"]
            )
            entry = {
                "step": int(step),
                "velocity_exact_count": int(velocity_exact),
                "terminal_exact_count": int(terminal_exact),
                "velocity_loss": float(vel_loss.detach().cpu().item()),
                "terminal_loss": float(terminal_loss_eval),
                "ar_loss": float(ar_loss.detach().cpu().item()),
                "velocity_infos": velocity_infos,
                "terminal_infos": terminal_infos,
                "ar_info": ar_info,
                "rollout": rollout,
            }
            trajectory.append(entry)
            score = (-velocity_exact, -terminal_exact, rollout["final_rf"])
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(module.model.state_dict())
                best_terminal_state = copy.deepcopy(terminal_head.state_dict())
                best_payload = copy.deepcopy(entry)

    module.model.load_state_dict(best_state)
    terminal_head.load_state_dict(best_terminal_state)
    after_velocity_infos, after_velocity_exact = eval_velocity()
    after_terminal_infos, after_terminal_exact, after_terminal_loss = eval_terminal(
        module, terminal_head, terminal_samples
    )
    after_ar = ar_eval(module, ar_batch)
    after_rollout = discrete_phase_rollout(
        module, terminal_head, pair["start_tree"], pair["target_tree"]
    )

    payload = {
        "config": str(CONFIG),
        "base_checkpoint": str(CKPT),
        "velocity_phase_targets": {
            "O0": velocity_samples[0][3],
            "A1": velocity_samples[1][3],
            "O2": velocity_samples[-1][3],
        },
        "ar_counts": {
            "path0": 1,
            "path1": 2,
            "path2": 15,
            "total": int(len(ar_samples)),
        },
        "before_velocity_exact_count": int(before_velocity_exact),
        "before_velocity_infos": before_velocity_infos,
        "before_terminal_exact_count": int(before_terminal_exact),
        "before_terminal_infos": before_terminal_infos,
        "before_terminal_loss": float(before_terminal_loss),
        "before_ar": before_ar,
        "before_rollout": before_rollout,
        "best_step_payload": best_payload,
        "after_velocity_exact_count": int(after_velocity_exact),
        "after_velocity_infos": after_velocity_infos,
        "after_terminal_exact_count": int(after_terminal_exact),
        "after_terminal_infos": after_terminal_infos,
        "after_terminal_loss": float(after_terminal_loss),
        "after_ar": after_ar,
        "after_rollout": after_rollout,
        "trajectory": trajectory,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(str(OUT))
    print(
        json.dumps(
            {
                "before_velocity_exact_count": payload["before_velocity_exact_count"],
                "before_terminal_exact_count": payload["before_terminal_exact_count"],
                "before_ar": before_ar,
                "before_rollout": before_rollout,
                "best_step_payload": best_payload,
                "after_velocity_exact_count": payload["after_velocity_exact_count"],
                "after_terminal_exact_count": payload["after_terminal_exact_count"],
                "after_ar": after_ar,
                "after_rollout": after_rollout,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
