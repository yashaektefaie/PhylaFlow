import inspect
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path("/home/yektefai/PhylaFlow")


def load_config(path: Path):
    return yaml.safe_load(path.read_text())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(config):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from run.run import _get_dataset_ids_from_config
    from data.dataset import PhylaDataModule

    data_cfg = config.get("data", {})
    uses_short_run = any(
        data_cfg.get(key) is not None
        for key in (
            "short_run_dataset_id",
            "short_run_dataset_ids",
            "posterior_dataset_id",
            "posterior_dataset_ids",
            "posterior_trprobs_root",
            "short_run_root",
        )
    )
    if uses_short_run or data_cfg.get("nexus_root") in {None, "", "unused"}:
        ids = _get_dataset_ids_from_config(config)
        dataset = PhylaDataModule(config, train_ids=ids, test_ids=ids)
        dataset.setup(stage=None)
        return dataset

    from analysis.full_sanity_fixedpair_20260327.analyze_current_run_sampler_ablation import (
        build_dataset as _build_dataset,
    )

    dataset = _build_dataset(config)
    dataset.setup(stage=None)
    return dataset


def build_fixed_pair(dataset):
    train_dataset = dataset.dataset_train
    pair = train_dataset.get_overfit_fixed_pair(0)
    if pair is None:
        _ = train_dataset[0]
        pair = train_dataset.get_overfit_fixed_pair(0)
    if pair is None:
        pair_bank = train_dataset._cached_overfit_pair_banks.get(0)
        if pair_bank:
            # For start-bank configs, evaluate consistently on the first banked start.
            pair = pair_bank[0]
    if pair is None:
        raise ValueError("Unable to build overfit fixed pair for probing.")
    return {
        "start_tree": pair.get("random_tree", pair.get("start_tree")),
        "target_tree": pair.get("effective_target_tree", pair.get("target_tree")),
    }


def build_training_sampling_pair(dataset):
    train_dataset = dataset.dataset_train
    if (
        getattr(train_dataset, "overfit_full_path_control_mode", False)
        and getattr(train_dataset, "_frozen_full_path_control_selections", None)
    ):
        sampled = train_dataset[0]
        return {
            "start_tree": sampled.get("start_tree"),
            "target_tree": sampled.get("target_tree"),
            "n_leaves": len(sampled.get("seq_ordering_map", {}))
            or len(sampled.get("num_to_name", {})),
            "max_events": int(sampled.get("fixed_pair_num_events", 1024)),
            "name_mapping": (
                train_dataset.return_nexus_number_to_name(0)
                if hasattr(train_dataset, "return_nexus_number_to_name")
                else None
            ),
        }
    return build_fixed_pair(dataset)


def build_module(config, dataset, ckpt_path: Path, device: str):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from model.model import return_model
    from run.TrainingModule import TrainingModule

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
    module.load_state_dict(state["state_dict"], strict=True)
    module.to(torch.device(device))
    module.eval()
    return module


def run_probe(
    config_path: Path,
    ckpt_path: Path,
    device: str,
    mode: str = "sample",
    bootstrap_module: str | None = None,
):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from utils.metric_utils import calculate_norm_rf

    if bootstrap_module:
        importlib.import_module(bootstrap_module)

    config = load_config(config_path)
    set_seed(int(config["trainer"].get("seed", 42)))
    dataset = build_dataset(config)
    module = build_module(config, dataset, ckpt_path, device)
    if mode == "harness":
        module.train()
        with torch.no_grad():
            metrics = module.sample_compare_harness(train=True)
        return {
            "config": str(config_path),
            "checkpoint": str(ckpt_path),
            "mode": mode,
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
        }

    module.train()
    pair = build_training_sampling_pair(dataset)
    sample_kwargs = module._build_harness_sample_kwargs(pair, train=True)
    with torch.no_grad():
        trees, _, _, _, _, trace = module.sample(
            [pair["start_tree"]],
            **sample_kwargs,
        )
    final_tree = trees[0]
    return {
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "mode": mode,
        "final_rf_norm": float(calculate_norm_rf(final_tree, pair["target_tree"])),
        "start_rf_norm": float(calculate_norm_rf(pair["start_tree"], pair["target_tree"])),
        "final_vs_start_rf_norm": float(
            calculate_norm_rf(final_tree, pair["start_tree"])
        ),
        "num_ar_states": int(len(trace.get("autoregressive", []))),
        "num_velocity_states": int(len(trace.get("velocity", []))),
        "num_topology_changes": int(trace.get("num_topology_changes", 0)),
        "stopped_for_no_valid_merge": bool(trace.get("stopped_for_no_valid_merge", False)),
        "stopped_for_repeated_topology": bool(
            trace.get("stopped_for_repeated_topology", False)
        ),
        "final_tree": final_tree,
        "start_tree": pair["start_tree"],
        "sample_kwargs": {
            "dt_base": float(sample_kwargs["dt_base"]),
            "fixed_dt_sampling": bool(sample_kwargs["fixed_dt_sampling"]),
            "max_steps": None
            if sample_kwargs["max_steps"] is None
            else int(sample_kwargs["max_steps"]),
            "max_events": None
            if sample_kwargs["max_events"] is None
            else int(sample_kwargs["max_events"]),
            "max_autoregressive_merges_per_boundary": int(
                sample_kwargs["max_autoregressive_merges_per_boundary"]
            ),
        },
    }


def main():
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: probe_current_mainline_ckpt.py <config.yaml> <checkpoint.ckpt> "
            "[--device=cuda|cpu] [--mode=sample|harness] [--bootstrap-module=module.path]"
        )
    config_path = Path(sys.argv[1])
    ckpt_path = Path(sys.argv[2])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = "sample"
    bootstrap_module = None
    for arg in sys.argv[3:]:
        if arg.startswith("--device="):
            device = arg.split("=", 1)[1]
        elif arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
        elif arg.startswith("--bootstrap-module="):
            bootstrap_module = arg.split("=", 1)[1]
    result = run_probe(
        config_path,
        ckpt_path,
        device,
        mode=mode,
        bootstrap_module=bootstrap_module,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
