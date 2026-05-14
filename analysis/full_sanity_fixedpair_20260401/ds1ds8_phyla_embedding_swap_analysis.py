#!/usr/bin/env python
"""Evaluate DS1-DS8 checkpoint behavior under swapped Phyla embeddings.

For each target dataset, this script builds one fixed unseen-start set, then
samples from the same checkpoint while forcing the Phyla embedding bank to DS1,
DS2, ... where the embedding bank has enough leaves for the target trees.  The
generated trees are scored against the target dataset's short/golden posterior
using the existing sample-metrics split/topology KL code.
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from ete3 import Tree as EteTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import PhylaDataModule
from model.model import return_model
from run.TrainingModule import TrainingModule
from run.run import (
    _configure_torch_runtime,
    _get_dataset_ids_from_config,
    _load_model_init_checkpoint,
    _set_global_seed,
)
from utils.metric_utils import calculate_norm_rf


DEFAULT_BASE_CONFIG = Path(
    "/ewsc/yektefai/30272299/launch_configs_ewsc/configs/"
    "local_ds1ds8_smallbank_exactanchors_phy256_leafglobal_cladehead_"
    "metricprobe64_fh64_aradd_mlp2cap_s128_lr2e3_ds2eval_mrbayes20k_20260505.yaml"
)
DEFAULT_CHECKPOINT = Path(
    "/ewsc/yektefai/phylaflow/checkpoints/full_sanity_fixedpair_20260401/"
    "ds1ds8_smallbank_exactanchors_phy256_leafglobal_cladehead_metricprobe64_"
    "fh64_aradd_mlp2cap_s128_lr2e3_ds2eval_mrbayes20k_20260505/"
    "2026-05-05_06-54-39/sample-metrics-epoch=1-step=009500.ckpt"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ewsc/yektefai/phylaflow/analysis/full_sanity_fixedpair_20260401/"
    "ds1ds8_step9500_phyla_embedding_swap_20260505"
)
CONFIG_ROOT = Path("/ewsc/yektefai/30272299/launch_configs_ewsc/configs")
DEFAULT_SAMPLE_CONFIGS = {
    "DS1": CONFIG_ROOT
    / "local_ds1_1280bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_"
    "unseeneval234_mrbayes20k_20260501.yaml",
    "DS2": CONFIG_ROOT
    / "local_ds2_210bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_"
    "unseeneval42_mrbayes20k_20260501.yaml",
    "DS3": CONFIG_ROOT
    / "local_ds3_1215bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_"
    "unseeneval243_mrbayes20k_20260501.yaml",
    "DS4": CONFIG_ROOT
    / "local_ds4_573bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_"
    "unseeneval573_mrbayes20k_smallbank_20260502.yaml",
    "DS5": CONFIG_ROOT
    / "local_ds5_525bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_"
    "unseeneval525_mrbayes20k_smallbank_20260502.yaml",
    "DS6": CONFIG_ROOT
    / "local_ds6_219bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_"
    "unseeneval219_mrbayes20k_smallbank_20260502.yaml",
    "DS7": CONFIG_ROOT
    / "local_ds7_1344bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_"
    "unseeneval1344_mrbayes20k_smallbank_20260502.yaml",
    "DS8": CONFIG_ROOT
    / "local_ds8_1122bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_"
    "unseeneval1122_mrbayes20k_smallbank_20260502.yaml",
}


def _load_yaml(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _split_ids(config):
    ids = _get_dataset_ids_from_config(config)
    rng = random.Random(42)
    rng.shuffle(ids)
    if len(ids) < 2:
        return ids, ids
    split = int(0.8 * len(ids))
    return ids[:split], ids[split:]


def _training_module_kwargs(config, model, dataset):
    trainer_cfg = dict(config.get("trainer") or {})
    signature = inspect.signature(TrainingModule.__init__)
    kwargs = {
        name: trainer_cfg[name]
        for name in signature.parameters
        if name not in {"self", "model", "dataset"} and name in trainer_cfg
    }
    kwargs["model"] = model
    kwargs["dataset"] = dataset
    kwargs.setdefault("logger", None)
    return kwargs


def _apply_sample_config(base_config, sample_config, output_dir, num_pairs):
    config = copy.deepcopy(base_config)
    sample_trainer_cfg = dict(sample_config.get("trainer") or {})
    trainer_cfg = config.setdefault("trainer", {})
    data_cfg = config.setdefault("data", {})
    data_cfg["sample_metrics_config_path"] = str(sample_config["_path"])

    for key in (
        "sample_metrics_num_pairs",
        "sample_metrics_unseen_start_seed",
        "sample_metrics_unseen_pair_selection_mode",
        "sample_metrics_unseen_start_max_duplicate_tries",
        "sample_metrics_unseen_start_metric_encoder_path",
        "sample_metrics_mrbayes20k_dataset_pickle_path",
        "sample_metrics_mrbayes20k_golden_root",
    ):
        if key in sample_trainer_cfg:
            trainer_cfg[key] = sample_trainer_cfg[key]

    trainer_cfg["record"] = False
    trainer_cfg["sample_metrics_trace_path"] = str(output_dir / "metrics.jsonl")
    trainer_cfg["sample_metrics_tree_dump_enabled"] = True
    trainer_cfg["sample_metrics_tree_dump_dir"] = str(output_dir / "generated_trees")
    trainer_cfg["sample_metrics_checkpoint_enabled"] = False
    trainer_cfg["sample_metrics_relaxed_likelihood_enabled"] = False
    trainer_cfg["sample_metrics_mrbayes20k_enabled"] = False
    trainer_cfg["sample_metrics_unseen_start_eval"] = True
    trainer_cfg["sample_metrics_unseen_pair_selection_mode"] = trainer_cfg.get(
        "sample_metrics_unseen_pair_selection_mode",
        "random_bank",
    )
    if num_pairs is not None:
        trainer_cfg["sample_metrics_num_pairs"] = int(num_pairs)
    return config


def _newick_leaf_count(tree):
    return len(EteTree(str(tree), format=1).get_leaves())


def _max_numeric_leaf_index(tree):
    values = []
    for leaf in EteTree(str(tree), format=1).iter_leaves():
        try:
            values.append(int(str(leaf.name)))
        except ValueError:
            return None
    return max(values) if values else None


def _embedding_leaf_capacity(module, dataset_id):
    tensor = getattr(module, "phyla_precomputed_dataset_id_to_tensor", {}).get(
        str(dataset_id).upper()
    )
    if tensor is None:
        return 0
    return int(tensor.size(0))


def _sample_once_forced_embedding(module, pair, forced_embedding_id):
    forced_pair = dict(pair)
    forced_pair["dataset_id"] = str(forced_embedding_id).upper()
    sample_kwargs = module._build_harness_sample_kwargs(forced_pair, train=True)
    if sample_kwargs.get("phyla_embeddings") is None:
        # Off-diagonal swaps intentionally reuse the target tree topology with a
        # different dataset bank.  Biological taxon names will not match there,
        # so fall back to numeric leaf positions when the bank has capacity.
        forced_pair["name_mapping"] = None
        sample_kwargs = module._build_harness_sample_kwargs(forced_pair, train=True)
    if sample_kwargs.get("phyla_embeddings") is None:
        raise RuntimeError(
            f"Could not resolve Phyla embeddings for {forced_embedding_id} "
            f"on {pair.get('n_leaves')} leaves."
        )
    sampled_trees, _, _, _, _, trace = module.sample(
        [pair["start_tree"]],
        **sample_kwargs,
    )
    sampled_tree = sampled_trees[0]
    return {
        "rf_norm": float(calculate_norm_rf(sampled_tree, pair["target_tree"])),
        "start_rf_norm": float(
            calculate_norm_rf(pair["start_tree"], pair["target_tree"])
        ),
        "_start_tree": str(pair["start_tree"]),
        "_original_start_tree": (
            str(pair["original_start_tree"])
            if pair.get("original_start_tree") is not None
            else None
        ),
        "_sampled_tree": str(sampled_tree),
        "_target_tree": str(pair["target_tree"]),
        "_n_leaves": int(pair["n_leaves"]),
        "_bank_group_key": pair.get("bank_group_key"),
        "_source_bank_index": pair.get("source_bank_index"),
        "forced_phyla_embedding_id": str(forced_embedding_id).upper(),
        "stopped_for_repeated_topology": float(
            1.0 if trace.get("stopped_for_repeated_topology", False) else 0.0
        ),
        "stopped_for_no_valid_merge": float(
            1.0 if trace.get("stopped_for_no_valid_merge", False) else 0.0
        ),
        "skipped_no_valid_boundary_revisits": float(
            trace.get("skipped_no_valid_boundary_revisits", 0.0)
        ),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def run_target(base_config, checkpoint, target_id, sample_config_path, condition_ids, args):
    target_dir = args.output_root / target_id
    target_dir.mkdir(parents=True, exist_ok=True)
    sample_config = _load_yaml(sample_config_path)
    sample_config["_path"] = str(sample_config_path)
    config = _apply_sample_config(base_config, sample_config, target_dir, args.num_pairs)
    with (target_dir / "effective_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    _set_global_seed(config.get("trainer", {}).get("seed"))
    train_ids, test_ids = _split_ids(config)
    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)
    model = return_model(config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _load_model_init_checkpoint(model, str(checkpoint), device)
    model.to(device)
    module = TrainingModule(**_training_module_kwargs(config, model, dataset))
    module.legacy_first_hit_gather_only = bool(
        config.get("trainer", {}).get("legacy_first_hit_gather_only", False)
    )
    module._sample_metrics_can_write_artifacts = lambda: True
    module.to(device)
    module.eval()

    dataset_split = module._sample_metrics_dataset_split(train=True)
    pairs = module._sample_metrics_unseen_bank_pairs(dataset_split, train=True)
    start_trees = [pair["start_tree"] for pair in pairs]
    max_leaf_count = max((_newick_leaf_count(tree) for tree in start_trees), default=0)
    max_numeric_leaf = max(
        (_max_numeric_leaf_index(tree) or 0 for tree in start_trees),
        default=0,
    )
    embeddings, embedding_stats = module._sample_metrics_encode_metric_starts(
        start_trees
    )
    if embeddings is None:
        raise RuntimeError(f"Could not encode start trees for {target_id}.")

    starts_path = target_dir / "fixed_unseen_start_pairs.jsonl"
    with starts_path.open("w", encoding="utf-8") as handle:
        for index, pair in enumerate(pairs):
            payload = {
                "index": index,
                "target_dataset_id": target_id,
                "source_bank_index": pair.get("source_bank_index"),
                "bank_group_key": pair.get("bank_group_key"),
                "n_leaves": pair.get("n_leaves"),
                "start_tree": pair.get("start_tree"),
                "target_tree": pair.get("target_tree"),
                "original_start_tree": pair.get("original_start_tree"),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    target_rows = []
    for condition_id in condition_ids:
        condition_id = str(condition_id).upper()
        condition_dir = target_dir / f"condition_{condition_id}"
        condition_dir.mkdir(parents=True, exist_ok=True)
        capacity = _embedding_leaf_capacity(module, condition_id)
        row_base = {
            "target_dataset_id": target_id,
            "condition_phyla_embedding_id": condition_id,
            "num_pairs": len(pairs),
            "max_leaf_count": max_leaf_count,
            "max_numeric_leaf_index": max_numeric_leaf,
            "embedding_leaf_capacity": capacity,
            "status": "pending",
        }
        if capacity < max_numeric_leaf:
            row = dict(row_base)
            row["status"] = "skipped_insufficient_embedding_leaves"
            target_rows.append(row)
            with (condition_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(_json_safe(row), handle, indent=2, sort_keys=True)
                handle.write("\n")
            continue

        module.sample_metrics_tree_dump_dir = str(condition_dir / "generated_trees")
        previous_label = getattr(module, "_sample_metrics_tree_dump_label", None)
        module._sample_metrics_tree_dump_label = (
            f"target_{target_id}_condition_{condition_id}"
        )
        rows = []
        started = time.time()
        replacements = module._sample_metrics_replace_frozen_start_case_tables(
            embeddings
        )
        try:
            with torch.inference_mode():
                for pair in pairs:
                    rows.append(_sample_once_forced_embedding(module, pair, condition_id))
        finally:
            module._sample_metrics_restore_frozen_start_case_tables(replacements)
            if previous_label is None:
                try:
                    delattr(module, "_sample_metrics_tree_dump_label")
                except AttributeError:
                    pass
            else:
                module._sample_metrics_tree_dump_label = previous_label

        metrics = module._summarize_sample_compare_harness_rows(rows, train=True)
        start_metrics = {
            f"start_{key}": value
            for key, value in module._posterior_reference_metrics(start_trees, train=True).items()
        }
        elapsed_sec = time.time() - started
        row = dict(row_base)
        row.update(
            {
                "status": "completed",
                "elapsed_sec": elapsed_sec,
                "tree_dump_dir": str(condition_dir / "generated_trees"),
                "golden_kl_divergence_topological": metrics.get(
                    "golden_kl_divergence_topological"
                ),
                "golden_kl_divergence_tree_topology": metrics.get(
                    "golden_kl_divergence_tree_topology"
                ),
                "short_kl_divergence_topological": metrics.get(
                    "short_kl_divergence_topological"
                ),
                "short_kl_divergence_tree_topology": metrics.get(
                    "short_kl_divergence_tree_topology"
                ),
                "golden_support_rate": metrics.get("golden_support_rate"),
                "short_support_rate": metrics.get("short_support_rate"),
                "rf_norm_mean": metrics.get("rf_norm_mean"),
                "rf_norm_best": metrics.get("rf_norm_best"),
                "rf_norm_worst": metrics.get("rf_norm_worst"),
                "sampled_topology_unique_count": metrics.get(
                    "sampled_topology_unique_count"
                ),
                "sampled_topology_mode_mass": metrics.get("sampled_topology_mode_mass"),
            }
        )
        summary = {
            **row,
            "checkpoint": str(checkpoint),
            "sample_config": str(sample_config_path),
            "fixed_start_pairs": str(starts_path),
            "embedding_stats": embedding_stats,
            "metrics": metrics,
            "start_metrics": start_metrics,
        }
        with (condition_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(summary), handle, indent=2, sort_keys=True)
            handle.write("\n")
        target_rows.append(row)

    return target_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--targets", nargs="+", default=[f"DS{i}" for i in range(1, 9)])
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[f"DS{i}" for i in range(1, 9)],
    )
    parser.add_argument("--num-pairs", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    _configure_torch_runtime()
    base_config = _load_yaml(args.base_config)
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for raw_target in args.targets:
        target_id = str(raw_target).upper()
        sample_config_path = DEFAULT_SAMPLE_CONFIGS[target_id]
        rows = run_target(
            base_config,
            args.checkpoint,
            target_id,
            sample_config_path,
            [str(value).upper() for value in args.conditions],
            args,
        )
        all_rows.extend(rows)

    fields = [
        "target_dataset_id",
        "condition_phyla_embedding_id",
        "status",
        "num_pairs",
        "max_leaf_count",
        "max_numeric_leaf_index",
        "embedding_leaf_capacity",
        "elapsed_sec",
        "golden_kl_divergence_topological",
        "golden_kl_divergence_tree_topology",
        "short_kl_divergence_topological",
        "short_kl_divergence_tree_topology",
        "golden_support_rate",
        "short_support_rate",
        "rf_norm_mean",
        "rf_norm_best",
        "rf_norm_worst",
        "sampled_topology_unique_count",
        "sampled_topology_mode_mass",
        "tree_dump_dir",
    ]
    csv_path = args.output_root / "summary_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    json_path = args.output_root / "summary_matrix.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(all_rows), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"summary_csv": str(csv_path), "rows": all_rows}, indent=2))


if __name__ == "__main__":
    main()
