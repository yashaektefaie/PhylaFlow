#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path("/home/yektefai/PhylaFlow")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import TreeDataset  # noqa: E402


SHORT_ROOT = "/home/yektefai/30272299/short_run_data_DS1-8"
DEFAULT_TEMPLATE = (
    ROOT
    / "configs"
    / "ds1_short_multipair78_discretephase_terminal_probeparity_fromscratch_rawfh_splitar_balancedanchors_signvel_phasefh_6000_20260417.yaml"
)
GENERATOR = (
    ROOT
    / "analysis"
    / "full_sanity_fixedpair_20260401"
    / "make_multi_singlepath_parity_bank.py"
)
ANALYSIS_DIR = ROOT / "analysis" / "full_sanity_fixedpair_20260401"
CONFIG_DIR = ROOT / "configs"

METRIC_LOG_EXACT_KEYS = [
    "loss",
    "train/velocity_loss_unscaled",
    "train/velocity_loss_regression_unscaled",
    "train/velocity_loss_auxiliary_unscaled",
    "train/velocity_loss_scaled",
    "train/terminal_loss_unscaled",
    "train/terminal_loss_scaled",
    "train/autoregressive_loss_unscaled",
    "train/autoregressive_loss_scaled",
    "train/probe_parity_joint_loss_scaled",
    "sample_metrics/num_pairs",
    "sample_metrics/rf_norm_mean",
    "sample_metrics/rf_norm_best",
    "sample_metrics/rf_norm_worst",
    "sample_metrics/rf_norm_min",
    "sample_metrics/rf_norm_max",
    "sample_metrics/short_kl_divergence_topological",
    "sample_metrics/short_kl_divergence_tree_topology",
    "sample_metrics/golden_kl_divergence_topological",
    "sample_metrics/golden_kl_divergence_tree_topology",
    "sample_metrics/short_support_rate",
    "sample_metrics/golden_support_rate",
    "sample_metrics/sampled_topology_unique_count",
    "sample_metrics/sampled_topology_mode_mass",
    "sample_metrics/sampled_topology_entropy",
    "sample_metrics/sampled_topology_entropy_normalized",
]


def _dataset_unique_topology_count(dataset_id: str, trprobs_sample_count_per_file: int) -> int:
    dataset = TreeDataset(
        nexus_root="unused",
        mrbayes_root="unused",
        posterior_trprobs_root=SHORT_ROOT,
        posterior_dataset_id=str(dataset_id),
        trprobs_sample_count_per_file=int(trprobs_sample_count_per_file),
    )
    trees = dataset.return_posterior_trees(str(dataset_id))
    return int(len(Counter(trees)))


def _write_temp_template(base_config_path: Path, dataset_id: str) -> Path:
    config = yaml.safe_load(base_config_path.read_text())
    config["data"]["short_run_dataset_id"] = str(dataset_id)
    config["data"]["short_run_root"] = str(SHORT_ROOT)
    fd, path_str = tempfile.mkstemp(prefix=f"{dataset_id.lower()}_", suffix="_template.yaml")
    Path(path_str).write_text(yaml.safe_dump(config, sort_keys=False))
    Path(path_str).chmod(0o644)
    return Path(path_str)


def _patch_wandb_clean_config(config_path: Path, dataset_id: str, num_cases: int, target_steps: int) -> None:
    config = yaml.safe_load(config_path.read_text())
    bank_name = config_path.stem
    epochs = max(1, int(math.ceil(float(target_steps) / float(num_cases))))
    trainer_cfg = config["trainer"]
    trainer_cfg["record"] = True
    trainer_cfg["epochs"] = epochs
    trainer_cfg["wandb_project"] = "DSoverfit"
    trainer_cfg["wandb_name"] = bank_name
    trainer_cfg["wandb_group"] = str(dataset_id)
    trainer_cfg["wandb_tags"] = [
        str(dataset_id),
        "batch1",
        "topofreqcover",
        "fullsupport",
        f"{num_cases}case",
    ]
    trainer_cfg["metric_log_exact_keys"] = list(METRIC_LOG_EXACT_KEYS)
    trainer_cfg["metric_log_prefixes"] = []

    data_cfg = config["data"]
    data_cfg["short_run_dataset_id"] = str(dataset_id)
    data_cfg["short_run_root"] = str(SHORT_ROOT)
    data_cfg["batch_size"] = 1
    data_cfg["num_workers"] = 0
    data_cfg["pin_memory"] = False
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


def _run_generator(
    *,
    template_config: Path,
    bank_name: str,
    num_cases: int,
    pair_seed: int,
    o0_count: int,
    a1_count: int,
    o2_count: int,
    min_boundary_paths: int,
    max_start_tries_per_target: int,
) -> None:
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--template-config",
        str(template_config),
        "--bank-name",
        str(bank_name),
        "--num-cases",
        str(int(num_cases)),
        "--dataset-index",
        "0",
        "--pair-seed",
        str(int(pair_seed)),
        "--o0-count",
        str(int(o0_count)),
        "--a1-count",
        str(int(a1_count)),
        "--o2-count",
        str(int(o2_count)),
        "--min-boundary-paths",
        str(int(min_boundary_paths)),
        "--max-start-tries-per-target",
        str(int(max_start_tries_per_target)),
        "--weight-target-topologies-by-posterior-frequency",
        "--ensure-all-topologies-represented",
        "--no-exclude-target",
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[f"DS{i}" for i in range(1, 9)],
    )
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--trprobs-sample-count", type=int, default=1000)
    parser.add_argument(
        "--support-multiplier",
        type=int,
        default=3,
        help="Bank size multiplier relative to number of unique posterior topologies.",
    )
    parser.add_argument("--target-steps", type=int, default=6000)
    parser.add_argument("--pair-seed", type=int, default=271828)
    parser.add_argument("--o0-count", type=int, default=4)
    parser.add_argument("--a1-count", type=int, default=4)
    parser.add_argument("--o2-count", type=int, default=4)
    parser.add_argument("--min-boundary-paths", type=int, default=3)
    parser.add_argument("--max-start-tries-per-target", type=int, default=200)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip bank/config emission when the target config already exists.",
    )
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    summary = []

    for dataset_id in [str(item) for item in args.datasets]:
        unique_topologies = _dataset_unique_topology_count(
            dataset_id,
            trprobs_sample_count_per_file=int(args.trprobs_sample_count),
        )
        num_cases = int(unique_topologies) * int(args.support_multiplier)
        bank_name = (
            f"{dataset_id.lower()}_short_multipair{num_cases}_topofreqcover_"
            f"discretephase_terminal_probeparity_wandbclean_"
            f"{int(args.target_steps)}_{stamp}"
        )
        config_path = CONFIG_DIR / f"{bank_name}.yaml"
        manifest_path = ANALYSIS_DIR / f"{bank_name}_manifest.json"
        metrics_path = ANALYSIS_DIR / f"{bank_name}_metrics.jsonl"

        if bool(args.skip_existing) and config_path.exists():
            summary.append(
                {
                    "dataset_id": dataset_id,
                    "unique_topologies": int(unique_topologies),
                    "num_cases": int(num_cases),
                    "epochs": int(math.ceil(float(args.target_steps) / float(num_cases))),
                    "config_yaml": str(config_path),
                    "manifest_json": str(manifest_path),
                    "metrics_path": str(metrics_path),
                    "status": "existing",
                }
            )
            continue

        temp_template = _write_temp_template(args.template_config, dataset_id)
        try:
            _run_generator(
                template_config=temp_template,
                bank_name=bank_name,
                num_cases=num_cases,
                pair_seed=int(args.pair_seed),
                o0_count=int(args.o0_count),
                a1_count=int(args.a1_count),
                o2_count=int(args.o2_count),
                min_boundary_paths=int(args.min_boundary_paths),
                max_start_tries_per_target=int(args.max_start_tries_per_target),
            )
        finally:
            temp_template.unlink(missing_ok=True)
        _patch_wandb_clean_config(
            config_path=config_path,
            dataset_id=dataset_id,
            num_cases=num_cases,
            target_steps=int(args.target_steps),
        )

        summary.append(
            {
                "dataset_id": dataset_id,
                "unique_topologies": int(unique_topologies),
                "num_cases": int(num_cases),
                "epochs": int(math.ceil(float(args.target_steps) / float(num_cases))),
                "config_yaml": str(config_path),
                "manifest_json": str(manifest_path),
                "metrics_path": str(metrics_path),
                "status": "generated",
            }
        )

    summary_path = ANALYSIS_DIR / f"all_ds_weighted_fullsupport_wandbclean_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary_json": str(summary_path), "datasets": summary}, indent=2))


if __name__ == "__main__":
    main()
