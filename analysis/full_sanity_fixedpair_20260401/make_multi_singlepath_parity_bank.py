import argparse
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path("/home/yektefai/PhylaFlow")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.full_sanity_fixedpair_20260401.make_singlepath_parity_case import (
    DEFAULT_CURRENT_TARGET,
    DEFAULT_TEMPLATE,
    _build_anchor_payloads,
    _load_current_target,
    _pick_pair,
)
from analysis.full_sanity_fixedpair_20260401.probe_current_mainline_ckpt import (
    build_dataset,
    set_seed,
)
from utils.metric_utils import canonicalize_topology_newick


def _case_file_paths(case_name: str):
    out_dir = ROOT / "analysis/full_sanity_fixedpair_20260401"
    return {
        "start_json": out_dir / f"{case_name}_start.json",
        "target_json": out_dir / f"{case_name}_target.json",
        "anchors_json": out_dir / f"{case_name}_velocity_anchors.json",
        "manifest_json": out_dir / f"{case_name}_manifest.json",
    }


def _write_case_files(case_name: str, pair, anchor_payload):
    paths = _case_file_paths(case_name)
    paths["start_json"].write_text(
        json.dumps({"tree": pair["start_tree"], "group_key": case_name}, indent=2)
    )
    paths["target_json"].write_text(
        json.dumps({"tree": pair["target_tree"], "group_key": case_name}, indent=2)
    )
    paths["anchors_json"].write_text(json.dumps(anchor_payload["anchors"], indent=2))
    manifest = {
        "case_name": str(case_name),
        "dataset_index": int(pair["dataset_index"]),
        "posterior_index": int(pair["posterior_index"]),
        "start_try": int(pair["start_try"]),
        "boundary_path_count": int(anchor_payload["boundary_path_count"]),
        "num_leaves": int(anchor_payload["num_leaves"]),
        "start_json": str(paths["start_json"]),
        "target_json": str(paths["target_json"]),
        "anchors_json": str(paths["anchors_json"]),
    }
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2))
    return manifest, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--exclude-target-json", type=Path, default=DEFAULT_CURRENT_TARGET)
    parser.add_argument("--bank-name", required=True)
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--pair-seed", type=int, default=271828)
    parser.add_argument("--o0-count", type=int, default=4)
    parser.add_argument("--a1-count", type=int, default=4)
    parser.add_argument("--o2-count", type=int, default=4)
    parser.add_argument("--min-boundary-paths", type=int, default=3)
    parser.add_argument("--max-start-tries-per-target", type=int, default=200)
    parser.add_argument(
        "--require-unique-target-topologies",
        action="store_true",
        help="Use at most one representative target tree per unique posterior topology.",
    )
    args = parser.parse_args()

    template_config = yaml.safe_load(args.template_config.read_text())
    set_seed(int(template_config["trainer"].get("seed", 42)))
    dataset = build_dataset(template_config)
    rng = random.Random(int(args.pair_seed))
    exclude_target = _load_current_target(args.exclude_target_json)

    seen_targets = set()
    if exclude_target is not None:
        seen_targets.add(str(exclude_target).strip())

    manifests = []
    combined_anchor_samples = []
    start_paths = []
    target_paths = []
    pair_attempt = 0
    target_index_schedule = None
    if bool(args.require_unique_target_topologies):
        posterior_trees = list(dataset.dataset_train.return_posterior_trees(int(args.dataset_index)))
        topology_to_indices = {}
        for idx, tree in enumerate(posterior_trees):
            key = canonicalize_topology_newick(str(tree).strip())
            topology_to_indices.setdefault(key, []).append(int(idx))
        target_index_schedule = [
            indices[0] for _, indices in sorted(topology_to_indices.items(), key=lambda item: item[1][0])
        ]
        rng.shuffle(target_index_schedule)
        if exclude_target is not None:
            exclude_key = canonicalize_topology_newick(str(exclude_target).strip())
            target_index_schedule = [
                idx
                for idx in target_index_schedule
                if canonicalize_topology_newick(str(posterior_trees[idx]).strip()) != exclude_key
            ]
        if int(args.num_cases) > len(target_index_schedule):
            raise RuntimeError(
                f"Requested {args.num_cases} cases but only {len(target_index_schedule)} unique target topologies exist."
            )

    while len(manifests) < int(args.num_cases):
        case_idx = len(manifests)
        case_name = f"{args.bank_name}_case{case_idx:02d}"
        posterior_index = None
        if target_index_schedule is not None:
            posterior_index = int(target_index_schedule[case_idx])
        pair = _pick_pair(
            dataset,
            dataset_index=int(args.dataset_index),
            rng=random.Random(int(args.pair_seed) + pair_attempt),
            exclude_target=None,
            posterior_index=posterior_index,
            min_boundary_paths=int(args.min_boundary_paths),
            max_start_tries_per_target=int(args.max_start_tries_per_target),
        )
        pair_attempt += 1
        target_key = str(pair["target_tree"]).strip()
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)

        anchor_payload = _build_anchor_payloads(
            pair["start_tree"],
            pair["target_tree"],
            bank_group_key=case_name,
            o0_count=int(args.o0_count),
            a1_count=int(args.a1_count),
            o2_count=int(args.o2_count),
        )
        manifest, case_paths = _write_case_files(case_name, pair, anchor_payload)
        manifests.append(manifest)
        combined_anchor_samples.extend(anchor_payload["anchors"])
        start_paths.append(str(case_paths["start_json"]))
        target_paths.append(str(case_paths["target_json"]))

    bank_analysis_dir = ROOT / "analysis/full_sanity_fixedpair_20260401"
    combined_anchors_path = bank_analysis_dir / f"{args.bank_name}_velocity_anchors.json"
    combined_manifest_path = bank_analysis_dir / f"{args.bank_name}_manifest.json"
    config_path = ROOT / "configs" / f"{args.bank_name}.yaml"
    metrics_path = bank_analysis_dir / f"{args.bank_name}_metrics.jsonl"

    combined_anchors_path.write_text(json.dumps(combined_anchor_samples, indent=2))

    config = json.loads(json.dumps(template_config))
    config["trainer"]["checkpoint_dir"] = (
        f"./checkpoints/full_sanity_fixedpair_20260401/{args.bank_name}"
    )
    config["trainer"]["sample_metrics_trace_path"] = str(metrics_path)
    config["trainer"]["sample_metrics_num_pairs"] = int(args.num_cases)
    config["trainer"]["sampling_random_fixed_pair_bank_use_at_sampling"] = True

    data_cfg = config["data"]
    data_cfg["overfit_full_path_control_extra_velocity_samples_json_path"] = str(
        combined_anchors_path
    )
    data_cfg["overfit_full_path_control_mode"] = True
    data_cfg["overfit_full_path_control_use_discrete_phase_time"] = True
    data_cfg["overfit_fixed_pair"] = True
    data_cfg["overfit_fixed_pair_start_tree_json_path"] = None
    data_cfg["overfit_fixed_pair_target_tree_json_path"] = None
    data_cfg["overfit_fixed_pair_start_tree_json_paths"] = start_paths
    data_cfg["overfit_fixed_pair_target_tree_json_paths"] = target_paths
    data_cfg["overfit_fixed_pair_group_by_json_metadata"] = True
    data_cfg["overfit_fixed_pair_reference_tree_from_target_bank"] = False
    data_cfg["overfit_virtual_epoch_size"] = int(args.num_cases)
    data_cfg["overfit_fixed_pair_cache_virtual_index_selection"] = True
    data_cfg["overfit_full_path_control_seed"] = int(args.pair_seed)

    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    payload = {
        "bank_name": str(args.bank_name),
        "num_cases": int(args.num_cases),
        "dataset_index": int(args.dataset_index),
        "pair_seed": int(args.pair_seed),
        "combined_anchors_json": str(combined_anchors_path),
        "config_yaml": str(config_path),
        "metrics_path": str(metrics_path),
        "cases": manifests,
    }
    combined_manifest_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
