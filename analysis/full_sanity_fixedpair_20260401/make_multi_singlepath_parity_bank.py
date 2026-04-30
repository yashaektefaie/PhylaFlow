import argparse
import json
import math
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
        "anchor_count": int(len(anchor_payload["anchors"])),
        "start_json": str(paths["start_json"]),
        "target_json": str(paths["target_json"]),
        "anchors_json": str(paths["anchors_json"]),
    }
    if anchor_payload.get("full_path_anchor_count") is not None:
        manifest["full_path_anchor_count"] = int(anchor_payload["full_path_anchor_count"])
    if pair.get("topology_key") is not None:
        manifest["topology_key"] = str(pair["topology_key"])
    if pair.get("topology_count") is not None:
        manifest["topology_count"] = int(pair["topology_count"])
    if pair.get("topology_probability") is not None:
        manifest["topology_probability"] = float(pair["topology_probability"])
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2))
    return manifest, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--exclude-target-json", type=Path, default=DEFAULT_CURRENT_TARGET)
    parser.add_argument(
        "--no-exclude-target",
        action="store_true",
        help="Do not exclude the default current target topology when building the bank.",
    )
    parser.add_argument("--bank-name", required=True)
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--pair-seed", type=int, default=271828)
    parser.add_argument("--o0-count", type=int, default=4)
    parser.add_argument("--a1-count", type=int, default=4)
    parser.add_argument("--o2-count", type=int, default=4)
    parser.add_argument(
        "--full-path-anchor-count",
        type=int,
        default=None,
        help=(
            "If set, generate this many anchors for every phase of every case "
            "instead of only O0/A1/O2."
        ),
    )
    parser.add_argument("--min-boundary-paths", type=int, default=3)
    parser.add_argument("--max-start-tries-per-target", type=int, default=200)
    parser.add_argument(
        "--require-unique-target-topologies",
        action="store_true",
        help="Use at most one representative target tree per unique posterior topology.",
    )
    parser.add_argument(
        "--weight-target-topologies-by-posterior-frequency",
        action="store_true",
        help=(
            "Allocate repeated target topologies in proportion to their frequency in "
            "the posterior tree sample. This allows duplicate target topologies in the bank."
        ),
    )
    parser.add_argument(
        "--ensure-all-topologies-represented",
        action="store_true",
        help=(
            "When using weighted topology allocation, give every unique posterior "
            "topology at least one case before distributing remaining cases by frequency."
        ),
    )
    parser.add_argument(
        "--target-schedule-manifest",
        type=Path,
        default=None,
        help=(
            "If set, reuse the source manifest's posterior_index/topology schedule "
            "instead of computing a fresh target allocation."
        ),
    )
    parser.add_argument(
        "--target-schedule-multiplier",
        type=int,
        default=1,
        help="Repeat each source manifest target schedule entry this many times.",
    )
    parser.add_argument(
        "--target-schedule-offset",
        type=int,
        default=0,
        help="Skip this many entries from the expanded target schedule.",
    )
    parser.add_argument(
        "--target-schedule-limit",
        type=int,
        default=None,
        help="Use at most this many entries from the expanded target schedule.",
    )
    parser.add_argument(
        "--case-index-offset",
        type=int,
        default=0,
        help="Offset emitted case indices and pair-attempt seeds by this amount.",
    )
    args = parser.parse_args()

    template_config = yaml.safe_load(args.template_config.read_text())
    set_seed(int(template_config["trainer"].get("seed", 42)))
    dataset = build_dataset(template_config)
    rng = random.Random(int(args.pair_seed))
    exclude_target = None
    if not bool(args.no_exclude_target):
        exclude_target = _load_current_target(args.exclude_target_json)

    seen_targets = set()
    if exclude_target is not None:
        seen_targets.add(str(exclude_target).strip())
    seen_start_target_pairs = set()

    manifests = []
    combined_anchor_samples = []
    start_paths = []
    target_paths = []
    pair_attempt = int(args.case_index_offset)
    target_index_schedule = None
    target_schedule_entries = None
    posterior_trees = list(
        dataset.dataset_train.return_posterior_trees(int(args.dataset_index))
    )
    topology_to_indices = {}
    for idx, tree in enumerate(posterior_trees):
        key = canonicalize_topology_newick(str(tree).strip())
        topology_to_indices.setdefault(key, []).append(int(idx))

    if args.target_schedule_manifest is not None:
        if bool(args.require_unique_target_topologies) or bool(
            args.weight_target_topologies_by_posterior_frequency
        ):
            raise RuntimeError(
                "Use either --target-schedule-manifest or target allocation flags, not both."
            )
        source_payload = json.loads(args.target_schedule_manifest.read_text())
        source_cases = list(source_payload.get("cases") or [])
        if not source_cases:
            raise RuntimeError(
                f"{args.target_schedule_manifest} does not contain any manifest cases."
            )
        target_schedule_entries = []
        for source_case in source_cases:
            if source_case.get("posterior_index") is None:
                raise RuntimeError(
                    f"{args.target_schedule_manifest} has a case without posterior_index."
                )
            entry = {
                "posterior_index": int(source_case["posterior_index"]),
                "topology_key": str(source_case.get("topology_key", "")),
                "topology_count": int(source_case.get("topology_count", 0)),
                "topology_probability": float(
                    source_case.get("topology_probability", 0.0)
                ),
            }
            for _ in range(int(args.target_schedule_multiplier)):
                target_schedule_entries.append(dict(entry))
        full_target_schedule_count = len(target_schedule_entries)
        if int(args.target_schedule_offset) < 0:
            raise RuntimeError("--target-schedule-offset must be nonnegative.")
        start = int(args.target_schedule_offset)
        stop = None
        if args.target_schedule_limit is not None:
            if int(args.target_schedule_limit) < 0:
                raise RuntimeError("--target-schedule-limit must be nonnegative.")
            stop = start + int(args.target_schedule_limit)
        target_schedule_entries = target_schedule_entries[start:stop]
        if len(target_schedule_entries) != int(args.num_cases):
            raise RuntimeError(
                f"Schedule manifest expansion created {len(target_schedule_entries)} cases, "
                f"after slicing {full_target_schedule_count} full entries, but "
                f"--num-cases is {int(args.num_cases)}."
            )
        rng.shuffle(target_schedule_entries)
    elif bool(args.weight_target_topologies_by_posterior_frequency):
        if bool(args.require_unique_target_topologies):
            raise RuntimeError(
                "Use either --require-unique-target-topologies or "
                "--weight-target-topologies-by-posterior-frequency, not both."
            )
        exclude_key = None
        if exclude_target is not None:
            exclude_key = canonicalize_topology_newick(str(exclude_target).strip())
        weighted_items = []
        for topo_key, indices in sorted(
            topology_to_indices.items(), key=lambda item: item[1][0]
        ):
            if exclude_key is not None and topo_key == exclude_key:
                continue
            weighted_items.append(
                {
                    "topology_key": str(topo_key),
                    "posterior_index": int(indices[0]),
                    "topology_count": int(len(indices)),
                }
            )
        total_count = int(sum(item["topology_count"] for item in weighted_items))
        if total_count <= 0:
            raise RuntimeError("No posterior topology mass available for weighted allocation.")
        base_alloc = [0 for _ in weighted_items]
        if bool(args.ensure_all_topologies_represented):
            if int(args.num_cases) < len(weighted_items):
                raise RuntimeError(
                    f"Requested {args.num_cases} cases but need at least "
                    f"{len(weighted_items)} to cover every topology."
                )
            base_alloc = [1 for _ in weighted_items]
        extra_cases = int(args.num_cases) - int(sum(base_alloc))
        scaled_counts = [
            (float(item["topology_count"]) / float(total_count)) * extra_cases
            for item in weighted_items
        ]
        allocated = [base + int(math.floor(count)) for base, count in zip(base_alloc, scaled_counts)]
        remaining = int(args.num_cases) - int(sum(allocated))
        if remaining > 0:
            remainder_order = sorted(
                range(len(weighted_items)),
                key=lambda idx: (
                    scaled_counts[idx] - math.floor(scaled_counts[idx]),
                    weighted_items[idx]["topology_count"],
                ),
                reverse=True,
            )
            for idx in remainder_order[:remaining]:
                allocated[idx] += 1
        target_schedule_entries = []
        for item, alloc in zip(weighted_items, allocated):
            if int(alloc) <= 0:
                continue
            prob = float(item["topology_count"]) / float(total_count)
            for _ in range(int(alloc)):
                target_schedule_entries.append(
                    {
                        "posterior_index": int(item["posterior_index"]),
                        "topology_key": str(item["topology_key"]),
                        "topology_count": int(item["topology_count"]),
                        "topology_probability": prob,
                    }
                )
        if len(target_schedule_entries) != int(args.num_cases):
            raise RuntimeError(
                f"Weighted allocation created {len(target_schedule_entries)} cases, "
                f"expected {int(args.num_cases)}."
            )
        rng.shuffle(target_schedule_entries)
    elif bool(args.require_unique_target_topologies):
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
        local_case_idx = len(manifests)
        case_idx = int(args.case_index_offset) + local_case_idx
        case_name = f"{args.bank_name}_case{case_idx:02d}"
        posterior_index = None
        topology_metadata = {}
        if target_index_schedule is not None:
            posterior_index = int(target_index_schedule[local_case_idx])
            topology_key = canonicalize_topology_newick(
                str(posterior_trees[posterior_index]).strip()
            )
            topology_metadata = {
                "topology_key": str(topology_key),
                "topology_count": int(len(topology_to_indices[topology_key])),
                "topology_probability": float(len(topology_to_indices[topology_key]))
                / float(len(posterior_trees)),
            }
        elif target_schedule_entries is not None:
            entry = target_schedule_entries[local_case_idx]
            posterior_index = int(entry["posterior_index"])
            topology_metadata = {
                "topology_key": str(entry["topology_key"]),
                "topology_count": int(entry["topology_count"]),
                "topology_probability": float(entry["topology_probability"]),
            }
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
        start_target_key = (str(pair["start_tree"]).strip(), target_key)
        if start_target_key in seen_start_target_pairs:
            continue
        if target_schedule_entries is None and target_key in seen_targets:
            continue
        seen_start_target_pairs.add(start_target_key)
        if target_schedule_entries is None:
            seen_targets.add(target_key)

        anchor_payload = _build_anchor_payloads(
            pair["start_tree"],
            pair["target_tree"],
            bank_group_key=case_name,
            o0_count=int(args.o0_count),
            a1_count=int(args.a1_count),
            o2_count=int(args.o2_count),
            full_path_count=(
                None
                if args.full_path_anchor_count is None
                else int(args.full_path_anchor_count)
            ),
        )
        pair.update(topology_metadata)
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
    model_cfg = config.get("model", {})
    for key in [
        "first_hit_head_num_cases",
        "autoregressive_num_cases",
        "velocity_terminal_head_num_cases",
        "branch_relax_num_cases",
    ]:
        if key in model_cfg and model_cfg.get(key) is not None:
            model_cfg[key] = int(args.num_cases)
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
        "full_path_anchor_count": (
            None
            if args.full_path_anchor_count is None
            else int(args.full_path_anchor_count)
        ),
        "weighted_by_topology_frequency": bool(
            args.weight_target_topologies_by_posterior_frequency
        ),
        "source_target_schedule_manifest": (
            None
            if args.target_schedule_manifest is None
            else str(args.target_schedule_manifest)
        ),
        "target_schedule_multiplier": int(args.target_schedule_multiplier),
        "combined_anchors_json": str(combined_anchors_path),
        "config_yaml": str(config_path),
        "metrics_path": str(metrics_path),
        "cases": manifests,
    }
    combined_manifest_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
