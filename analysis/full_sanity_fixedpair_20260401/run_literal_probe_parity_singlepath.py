import argparse
import copy
import inspect
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path("/home/yektefai/PhylaFlow")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.full_sanity_fixedpair_20260401.probe_current_mainline_ckpt import (
    build_dataset,
    set_seed,
)
from analysis.full_sanity_fixedpair_20260401.probe_discrete_phase_terminal_head import (
    TerminalHead,
    ar_eval,
    build_direct_set_loss,
    build_path_ar_samples,
    discrete_phase_rollout,
    eval_terminal,
    extract_tree_phase_feature,
    target_splits_from_velocity_sample,
)
from run.TrainingModule import (
    _build_autoregressive_replay_batch,
    _build_legacy_velocity_oracle_sample,
    _build_pair_oracle_orthant_velocity_label_map,
    _topology_key,
)
from model.model import return_model
from run.TrainingModule import TrainingModule
from utils.bhv_utils import return_tree_boundary_merge_paths


DEFAULT_START_JSON = ROOT / (
    "analysis/full_sanity_fixedpair_20260401/"
    "randomstart20_singlepair_discretephase_probe_single_start_20260416.json"
)
DEFAULT_TARGET_JSON = ROOT / (
    "analysis/full_sanity_fixedpair_20260401/"
    "randomstart20_singlepair_discretephase_probe_single_target_20260416.json"
)
DEFAULT_ANCHORS_JSON = ROOT / (
    "analysis/full_sanity_fixedpair_20260401/"
    "randomstart20_singlepair_discretephase_probe_singlepath_velocity_anchors_20260416.json"
)


def _load_tree_json(path: Path):
    payload = json.loads(path.read_text())
    return str(payload["tree"])


def _build_family_canonical_samples(start_tree, target_tree, anchors):
    num_leaves = int(anchors[0]["num_leaves"])
    canonical_map, _ = _build_pair_oracle_orthant_velocity_label_map(
        start_tree, target_tree
    )
    family_samples = {}

    start_key = _topology_key(start_tree)
    if start_key in canonical_map:
        family_samples["O0"] = {
            **dict(canonical_map[start_key]),
            "num_leaves": num_leaves,
        }

    a1_sample = next((x for x in anchors if str(x.get("anchor_family")) == "A1"), None)
    if a1_sample is not None:
        a1_key = tuple(_topology_key(str(a1_sample["newick_tree"])))
        if a1_key in canonical_map:
            family_samples["A1"] = {
                **dict(canonical_map[a1_key]),
                "num_leaves": num_leaves,
            }

    boundary_paths = return_tree_boundary_merge_paths(start_tree, target_tree)
    if len(boundary_paths) >= 2:
        third_orthant_source = str(boundary_paths[1]["end_newick"])
        family_samples["O2"] = _build_legacy_velocity_oracle_sample(
            third_orthant_source,
            target_tree,
            timepoint=2.0,
            num_leaves=num_leaves,
        )
    return family_samples


def _build_velocity_samples(anchors, family_canonical_samples):
    out = []
    for idx, sample in enumerate(anchors):
        family = str(sample.get("anchor_family", f"anchor_{idx}"))
        name = str(sample.get("source_checkpoint", sample.get("path_index", idx)))
        train_sample = dict(sample)
        canonical_sample = family_canonical_samples.get(family)
        if canonical_sample is not None:
            train_sample["velocity"] = dict(canonical_sample["velocity"])
            if canonical_sample.get("velocity_next_boundary_tree") is not None:
                train_sample["velocity_next_boundary_tree"] = canonical_sample.get(
                    "velocity_next_boundary_tree"
                )
            target = target_splits_from_velocity_sample(canonical_sample)
        else:
            target = target_splits_from_velocity_sample(sample)
        out.append((family, name, train_sample, target))
    return out


def _build_terminal_samples(boundary_paths):
    samples = {}
    last_idx = len(boundary_paths) - 1
    for idx, path in enumerate(boundary_paths):
        samples[f"path{idx}_end"] = {
            "newick": str(path["end_newick"]),
            "phase": int(idx),
            "label": bool(idx == last_idx),
        }
    return samples


def _build_ar_samples(target_tree, boundary_paths):
    samples = []
    counts = {}
    for idx, path in enumerate(boundary_paths):
        path_samples = build_path_ar_samples(path, target_tree, phase_idx=idx)
        counts[f"path{idx}"] = int(len(path_samples))
        samples.extend(path_samples)
    counts["total"] = int(len(samples))
    return samples, counts


def _derive_default_out(config_path: Path, ckpt_path: Path):
    stem = (
        f"{config_path.stem}__{ckpt_path.parent.name}__{ckpt_path.stem}"
        "__literal_probe_parity.json"
    )
    return ROOT / "analysis/full_sanity_fixedpair_20260401" / stem


def _build_module(config, dataset, ckpt_path: Path, device: str, strict_load: bool):
    trainer_cfg = dict(config.get("trainer", {}))
    sig = inspect.signature(TrainingModule.__init__)
    kwargs = {
        "model": return_model(config),
        "dataset": dataset,
        "lr": float(trainer_cfg.get("lr", 1e-4)),
        "record": False,
        "epochs": int(trainer_cfg.get("epochs", 1)),
    }
    for key, value in trainer_cfg.items():
        if key in kwargs:
            continue
        if key in sig.parameters:
            kwargs[key] = value
    module = TrainingModule(**kwargs)
    module.legacy_first_hit_gather_only = bool(
        trainer_cfg.get("legacy_first_hit_gather_only", False)
    )
    state = torch.load(ckpt_path, map_location="cpu")
    load_result = module.load_state_dict(state["state_dict"], strict=strict_load)
    module.to(torch.device(device))
    module.eval()
    return module, {
        "strict_load": bool(strict_load),
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
    }


def run_literal_probe(
    config_path: Path,
    ckpt_path: Path,
    start_json: Path,
    target_json: Path,
    anchors_json: Path,
    out_path: Path,
    device: str,
    lr: float,
    steps: int,
    max_phases: int,
    max_events: int,
    strict_load: bool,
):
    config = yaml.safe_load(config_path.read_text())
    set_seed(int(config["trainer"].get("seed", 42)))
    dataset = build_dataset(config)
    module, load_info = _build_module(config, dataset, ckpt_path, device, strict_load)
    module.record = False
    module.autoregressive_use_time = True
    module.autoregressive_stop_after_merge_weight = 0.0

    start_tree = _load_tree_json(start_json)
    target_tree = _load_tree_json(target_json)
    anchors = json.loads(anchors_json.read_text())
    family_canonical_samples = _build_family_canonical_samples(
        start_tree, target_tree, anchors
    )
    velocity_samples = _build_velocity_samples(anchors, family_canonical_samples)
    boundary_paths = return_tree_boundary_merge_paths(start_tree, target_tree)
    terminal_samples = _build_terminal_samples(boundary_paths)
    ar_samples, ar_counts = _build_ar_samples(target_tree, boundary_paths)
    ar_batch = _build_autoregressive_replay_batch(module, ar_samples)

    with torch.no_grad():
        feature_dim = int(
            extract_tree_phase_feature(
                module,
                next(iter(terminal_samples.values()))["newick"],
                next(iter(terminal_samples.values()))["phase"],
            ).numel()
        )
    terminal_head = TerminalHead(feature_dim).to(module.device)

    def eval_velocity():
        module.eval()
        infos = {}
        exact_count = 0
        for family, name, sample, target in velocity_samples:
            _, info = build_direct_set_loss(module, sample, target)
            infos.setdefault(family, {})[name] = info
            exact_count += int(info["exact"])
        return infos, int(exact_count)

    before_velocity_infos, before_velocity_exact = eval_velocity()
    before_ar = ar_eval(module, ar_batch)
    before_terminal_infos, before_terminal_exact, before_terminal_loss = eval_terminal(
        module, terminal_head, terminal_samples
    )
    before_rollout = discrete_phase_rollout(
        module,
        terminal_head,
        start_tree,
        target_tree,
        max_phases=max_phases,
        max_events=max_events,
    )

    best_state = copy.deepcopy(module.model.state_dict())
    best_terminal_state = copy.deepcopy(terminal_head.state_dict())
    best_payload = None
    best_score = (-before_velocity_exact, -before_terminal_exact, before_rollout["final_rf"])
    trajectory = []

    optimizer = torch.optim.Adam(
        list(module.model.parameters()) + list(terminal_head.parameters()),
        lr=lr,
    )
    for step in range(1, int(steps) + 1):
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
            term_losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
            )
        terminal_loss = torch.stack(term_losses).mean()

        ar_logs = module.step(ar_batch, autoregressive=True)
        ar_loss = ar_logs["loss"]
        total_loss = vel_loss + terminal_loss + ar_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(module.model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0 or step == int(steps):
            velocity_infos, velocity_exact = eval_velocity()
            terminal_infos, terminal_exact, terminal_loss_eval = eval_terminal(
                module, terminal_head, terminal_samples
            )
            ar_info = ar_eval(module, ar_batch)
            rollout = discrete_phase_rollout(
                module,
                terminal_head,
                start_tree,
                target_tree,
                max_phases=max_phases,
                max_events=max_events,
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
        module,
        terminal_head,
        start_tree,
        target_tree,
        max_phases=max_phases,
        max_events=max_events,
    )

    payload = {
        "config": str(config_path),
        "base_checkpoint": str(ckpt_path),
        "start_json": str(start_json),
        "target_json": str(target_json),
        "anchors_json": str(anchors_json),
        "optimizer": {"lr": float(lr), "steps": int(steps)},
        "rollout_limits": {"max_phases": int(max_phases), "max_events": int(max_events)},
        "checkpoint_load": load_info,
        "velocity_phase_targets": {
            family: {name: info["target_splits"] for name, info in fam_infos.items()}
            for family, fam_infos in before_velocity_infos.items()
        },
        "family_canonical_samples": {
            family: {
                "newick_tree": str(sample["newick_tree"]),
                "timepoint": float(sample["timepoint"]),
                "target_splits": target_splits_from_velocity_sample(sample),
            }
            for family, sample in family_canonical_samples.items()
        },
        "ar_counts": ar_counts,
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
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--start-json", default=str(DEFAULT_START_JSON))
    parser.add_argument("--target-json", default=str(DEFAULT_TARGET_JSON))
    parser.add_argument("--anchors-json", default=str(DEFAULT_ANCHORS_JSON))
    parser.add_argument("--out")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-phases", type=int, default=6)
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument(
        "--strict-load",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out) if args.out else _derive_default_out(config_path, ckpt_path)

    payload = run_literal_probe(
        config_path=config_path,
        ckpt_path=ckpt_path,
        start_json=Path(args.start_json),
        target_json=Path(args.target_json),
        anchors_json=Path(args.anchors_json),
        out_path=out_path,
        device=args.device,
        lr=float(args.lr),
        steps=int(args.steps),
        max_phases=int(args.max_phases),
        max_events=int(args.max_events),
        strict_load=bool(args.strict_load),
    )
    print(str(out_path))
    print(
        json.dumps(
            {
                "before_velocity_exact_count": payload["before_velocity_exact_count"],
                "before_terminal_exact_count": payload["before_terminal_exact_count"],
                "before_ar": payload["before_ar"],
                "before_rollout": payload["before_rollout"],
                "best_step_payload": payload["best_step_payload"],
                "after_velocity_exact_count": payload["after_velocity_exact_count"],
                "after_terminal_exact_count": payload["after_terminal_exact_count"],
                "after_ar": payload["after_ar"],
                "after_rollout": payload["after_rollout"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
