import argparse
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path("/home/yektefai/PhylaFlow")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.full_sanity_fixedpair_20260401.probe_current_mainline_ckpt import (
    build_dataset,
    set_seed,
)
from run.TrainingModule import _build_legacy_velocity_oracle_sample
from utils.bhv_utils import (
    return_sampled_tree_orthant_velocity,
    return_tree_boundary_merge_paths,
)
from utils.random_tree import Tree


DEFAULT_TEMPLATE = ROOT / (
    "configs/"
    "randomstart20_singlepair_discretephase_terminal_probeparity_singlepath_"
    "fromscratch_rawfh_splitar_1000_20260416.yaml"
)
DEFAULT_CURRENT_TARGET = ROOT / (
    "analysis/full_sanity_fixedpair_20260401/"
    "randomstart20_singlepair_discretephase_probe_single_target_20260416.json"
)


def _load_current_target(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text())
    return str(payload["tree"])


def _pick_pair(
    dataset,
    *,
    dataset_index: int,
    rng: random.Random,
    exclude_target: str | None,
    posterior_index: int | None,
    min_boundary_paths: int,
    max_start_tries_per_target: int,
):
    train = dataset.dataset_train
    posterior_trees = list(train.return_posterior_trees(int(dataset_index)))
    if not posterior_trees:
        raise RuntimeError("No posterior trees available for parity-case generation.")

    candidate_indices = list(range(len(posterior_trees)))
    if posterior_index is not None:
        candidate_indices = [int(posterior_index)]
    else:
        rng.shuffle(candidate_indices)

    for target_idx in candidate_indices:
        target_tree = str(posterior_trees[int(target_idx)]).strip()
        if not target_tree.endswith(";"):
            target_tree += ";"
        if exclude_target is not None and str(exclude_target).strip() == target_tree:
            continue

        for start_try in range(int(max_start_tries_per_target)):
            start_tree = str(train.sample_random_tree(target_tree)).strip()
            if not start_tree.endswith(";"):
                start_tree += ";"
            boundary_paths = return_tree_boundary_merge_paths(start_tree, target_tree)
            if len(boundary_paths) >= int(min_boundary_paths):
                return {
                    "dataset_index": int(dataset_index),
                    "posterior_index": int(target_idx),
                    "start_try": int(start_try),
                    "start_tree": start_tree,
                    "target_tree": target_tree,
                    "boundary_paths": boundary_paths,
                }

    raise RuntimeError(
        "Unable to sample a fresh pair with enough boundary paths. "
        f"Tried {len(candidate_indices)} target trees."
    )


def _sample_phase_family_anchors(
    phase_source_tree: str,
    target_tree: str,
    *,
    bank_group_key: str,
    anchor_family: str,
    phase_idx: int,
    num_leaves: int,
    count: int,
):
    local_paths = return_tree_boundary_merge_paths(phase_source_tree, target_tree)
    if not local_paths:
        raise RuntimeError(f"No local boundary paths available for family {anchor_family}.")
    local_time = float(local_paths[0]["global_time"])
    if local_time <= 0.0:
        fractions = [0.0] * int(count)
    elif int(count) == 1:
        fractions = [0.0]
    else:
        fractions = [0.0, 0.25, 0.5, 0.75][: int(count)]
        while len(fractions) < int(count):
            fractions.append(fractions[-1])

    anchors = []
    next_boundary_tree = str(local_paths[0]["start_newick"])
    for idx, frac in enumerate(fractions):
        u = 0.0
        if float(frac) > 0.0 and local_time > 0.0:
            u = min(float(frac) * local_time * 0.95, max(local_time - 1e-8, 0.0))
        sampled_tree, sampled_velocity = return_sampled_tree_orthant_velocity(
            phase_source_tree,
            target_tree,
            u,
            legacy_training_semantics=False,
        )
        anchors.append(
            {
                "anchor_family": str(anchor_family),
                "source_checkpoint": f"{anchor_family.lower()}_{idx}",
                "path_index": int(phase_idx),
                "timepoint": float(phase_idx),
                "num_leaves": int(num_leaves),
                "newick_tree": str(sampled_tree),
                "target_tree": str(target_tree),
                "velocity": {int(k): float(v) for k, v in sampled_velocity.items()},
                "velocity_next_boundary_tree": next_boundary_tree,
                "local_anchor_time": float(u),
                "local_next_boundary_time": float(local_time),
                "bank_group_key": str(bank_group_key),
            }
        )
    return anchors


def _build_anchor_payloads(
    start_tree: str,
    target_tree: str,
    *,
    bank_group_key: str,
    o0_count: int,
    a1_count: int,
    o2_count: int,
    full_path_count: int | None = None,
):
    boundary_paths = return_tree_boundary_merge_paths(start_tree, target_tree)
    if len(boundary_paths) < 3:
        raise RuntimeError(
            f"Need at least 3 boundary paths for the parity case, got {len(boundary_paths)}."
        )
    num_leaves = int(Tree(start_tree).n_leaves)

    if full_path_count is not None:
        def _phase_family_name(phase_idx: int) -> str:
            if int(phase_idx) == 0:
                return "O0"
            if int(phase_idx) == 1:
                return "A1"
            if int(phase_idx) == 2:
                return "O2"
            return f"P{int(phase_idx)}"

        anchors = []
        phase_sources = [str(start_tree)]
        phase_sources.extend(str(path["end_newick"]) for path in boundary_paths[:-1])
        for phase_idx, phase_source in enumerate(phase_sources):
            family = _phase_family_name(int(phase_idx))
            phase_anchors = _sample_phase_family_anchors(
                phase_source,
                target_tree,
                bank_group_key=str(bank_group_key),
                anchor_family=family,
                phase_idx=int(phase_idx),
                num_leaves=num_leaves,
                count=int(full_path_count),
            )
            canonical = _build_legacy_velocity_oracle_sample(
                phase_source,
                target_tree,
                timepoint=float(phase_idx),
                num_leaves=num_leaves,
            )
            if phase_anchors and canonical is not None:
                phase_anchors[0].update(
                    {
                        "velocity": dict(canonical.get("velocity", {})),
                        "velocity_next_boundary_tree": canonical.get(
                            "velocity_next_boundary_tree"
                        ),
                        "source_checkpoint": None,
                    }
                )
            anchors.extend(phase_anchors)

        return {
            "num_leaves": int(num_leaves),
            "boundary_path_count": int(len(boundary_paths)),
            "anchors": anchors,
            "full_path_anchor_count": int(full_path_count),
        }

    o0 = _build_legacy_velocity_oracle_sample(
        start_tree,
        target_tree,
        timepoint=0.0,
        num_leaves=num_leaves,
    )
    o0["anchor_family"] = "O0"
    o0["path_index"] = 0
    o0["source_checkpoint"] = None
    o0["bank_group_key"] = str(bank_group_key)
    o0_anchors = _sample_phase_family_anchors(
        start_tree,
        target_tree,
        bank_group_key=str(bank_group_key),
        anchor_family="O0",
        phase_idx=0,
        num_leaves=num_leaves,
        count=int(o0_count),
    )
    if o0_anchors:
        o0_anchors[0].update(
            {
                "velocity": dict(o0.get("velocity", {})),
                "velocity_next_boundary_tree": o0.get("velocity_next_boundary_tree"),
                "source_checkpoint": None,
            }
        )

    phase1_source = str(boundary_paths[0]["end_newick"])
    a1_anchors = _sample_phase_family_anchors(
        phase1_source,
        target_tree,
        bank_group_key=str(bank_group_key),
        anchor_family="A1",
        phase_idx=1,
        num_leaves=num_leaves,
        count=int(a1_count),
    )

    phase2_source = str(boundary_paths[1]["end_newick"])
    o2 = _build_legacy_velocity_oracle_sample(
        phase2_source,
        target_tree,
        timepoint=2.0,
        num_leaves=num_leaves,
    )
    o2["anchor_family"] = "O2"
    o2["path_index"] = 2
    o2["source_checkpoint"] = None
    o2["bank_group_key"] = str(bank_group_key)
    o2_anchors = _sample_phase_family_anchors(
        phase2_source,
        target_tree,
        bank_group_key=str(bank_group_key),
        anchor_family="O2",
        phase_idx=2,
        num_leaves=num_leaves,
        count=int(o2_count),
    )
    if o2_anchors:
        o2_anchors[0].update(
            {
                "velocity": dict(o2.get("velocity", {})),
                "velocity_next_boundary_tree": o2.get("velocity_next_boundary_tree"),
                "source_checkpoint": None,
            }
        )

    return {
        "num_leaves": int(num_leaves),
        "boundary_path_count": int(len(boundary_paths)),
        "anchors": [*o0_anchors, *a1_anchors, *o2_anchors],
    }


def _update_config_paths(config, *, case_name: str, start_json: Path, target_json: Path, anchors_json: Path):
    cfg = json.loads(json.dumps(config))
    cfg["trainer"]["checkpoint_dir"] = (
        f"./checkpoints/full_sanity_fixedpair_20260401/{case_name}"
    )
    cfg["trainer"]["sample_metrics_trace_path"] = str(
        ROOT / "analysis/full_sanity_fixedpair_20260401" / f"{case_name}_metrics.jsonl"
    )
    cfg["data"]["overfit_full_path_control_extra_velocity_samples_json_path"] = str(
        anchors_json
    )
    cfg["data"]["overfit_full_path_control_mode"] = True
    cfg["data"]["overfit_full_path_control_use_discrete_phase_time"] = True
    cfg["data"]["overfit_fixed_pair"] = True
    cfg["data"]["overfit_fixed_pair_start_tree_json_path"] = str(start_json)
    cfg["data"]["overfit_fixed_pair_target_tree_json_path"] = str(target_json)
    cfg["data"]["overfit_fixed_pair_start_tree_json_paths"] = None
    cfg["data"]["overfit_fixed_pair_target_tree_json_paths"] = None
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--exclude-target-json", type=Path, default=DEFAULT_CURRENT_TARGET)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--posterior-index", type=int, default=None)
    parser.add_argument("--pair-seed", type=int, default=314159)
    parser.add_argument("--o0-count", type=int, default=4)
    parser.add_argument("--a1-count", type=int, default=4)
    parser.add_argument("--o2-count", type=int, default=4)
    parser.add_argument(
        "--full-path-anchor-count",
        type=int,
        default=None,
        help=(
            "If set, generate this many anchors for every phase of the oracle path "
            "instead of only O0/A1/O2."
        ),
    )
    parser.add_argument("--min-boundary-paths", type=int, default=3)
    parser.add_argument("--max-start-tries-per-target", type=int, default=200)
    args = parser.parse_args()

    config = yaml.safe_load(args.template_config.read_text())
    set_seed(int(config["trainer"].get("seed", 42)))
    dataset = build_dataset(config)
    rng = random.Random(int(args.pair_seed))

    pair = _pick_pair(
        dataset,
        dataset_index=int(args.dataset_index),
        rng=rng,
        exclude_target=_load_current_target(args.exclude_target_json),
        posterior_index=args.posterior_index,
        min_boundary_paths=int(args.min_boundary_paths),
        max_start_tries_per_target=int(args.max_start_tries_per_target),
    )

    anchor_payload = _build_anchor_payloads(
        pair["start_tree"],
        pair["target_tree"],
        bank_group_key=str(args.case_name),
        o0_count=int(args.o0_count),
        a1_count=int(args.a1_count),
        o2_count=int(args.o2_count),
        full_path_count=(
            None
            if args.full_path_anchor_count is None
            else int(args.full_path_anchor_count)
        ),
    )

    out_dir = ROOT / "analysis/full_sanity_fixedpair_20260401"
    start_json = out_dir / f"{args.case_name}_start.json"
    target_json = out_dir / f"{args.case_name}_target.json"
    anchors_json = out_dir / f"{args.case_name}_velocity_anchors.json"
    config_yaml = ROOT / "configs" / f"{args.case_name}.yaml"
    manifest_json = out_dir / f"{args.case_name}_manifest.json"

    start_json.write_text(
        json.dumps(
            {
                "tree": pair["start_tree"],
                "group_key": str(args.case_name),
            },
            indent=2,
        )
    )
    target_json.write_text(
        json.dumps(
            {
                "tree": pair["target_tree"],
                "group_key": str(args.case_name),
            },
            indent=2,
        )
    )
    anchors_json.write_text(json.dumps(anchor_payload["anchors"], indent=2))

    case_config = _update_config_paths(
        config,
        case_name=str(args.case_name),
        start_json=start_json,
        target_json=target_json,
        anchors_json=anchors_json,
    )
    config_yaml.write_text(yaml.safe_dump(case_config, sort_keys=False))

    manifest = {
        "case_name": str(args.case_name),
        "dataset_index": int(pair["dataset_index"]),
        "posterior_index": int(pair["posterior_index"]),
        "start_try": int(pair["start_try"]),
        "pair_seed": int(args.pair_seed),
        "boundary_path_count": int(anchor_payload["boundary_path_count"]),
        "num_leaves": int(anchor_payload["num_leaves"]),
        "anchor_count": int(len(anchor_payload["anchors"])),
        "start_json": str(start_json),
        "target_json": str(target_json),
        "anchors_json": str(anchors_json),
        "config_yaml": str(config_yaml),
    }
    if anchor_payload.get("full_path_anchor_count") is not None:
        manifest["full_path_anchor_count"] = int(anchor_payload["full_path_anchor_count"])
    manifest_json.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
