#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ete3 import Tree as EteTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.full_sanity_fixedpair_20260401.multi_ds_branchwarm_cumulative_mh_experiment import (  # noqa: E402
    GenericJCLikelihood,
)
from analysis.full_sanity_fixedpair_20260401.split_guided_start_experiment import (  # noqa: E402
    _ensure_semicolon,
)
from analysis.full_sanity_fixedpair_20260401.train_standalone_branch_relaxer_20260429 import (  # noqa: E402
    BranchDeltaHead,
    StandaloneRelaxer,
    _load_phyla_embedding_bank,
    _small_model_config,
)
from model.model import return_model  # noqa: E402
from run.TrainingModule import (  # noqa: E402
    _branch_relax_entries_for_tree,
    _move_tokenized_batch_to_device,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "analysis/full_sanity_fixedpair_20260401"
    / "standalone_branch_relaxer_ds1_ds8_phyla_leafonly_balanced_nocase_20260429"
    / "best.pt"
)


def _read_tree_list(path: Path) -> list[str]:
    trees: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("(") or line.startswith("[") or line.endswith(";"):
            trees.append(_ensure_semicolon(line))
            continue
        maybe_path = Path(line)
        try:
            exists = maybe_path.exists()
        except OSError:
            exists = False
        if exists:
            trees.append(_ensure_semicolon(maybe_path.read_text().strip()))
        else:
            trees.append(_ensure_semicolon(line))
    if not trees:
        raise ValueError(f"No trees found in {path}")
    return trees


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _checkpoint_args(checkpoint: dict) -> SimpleNamespace:
    raw_args = dict(checkpoint.get("args") or {})

    def _localize_repo_path(value):
        if value is None:
            return value
        value = str(value)
        old_root = "/home/yektefai/PhylaFlow"
        if value == old_root:
            return str(REPO_ROOT)
        if value.startswith(old_root + os.sep):
            return str(REPO_ROOT / value[len(old_root) + 1 :])
        return value

    def _localize_base_config(value):
        value = _localize_repo_path(value)
        if value and not os.path.exists(value):
            directory, filename = os.path.split(value)
            if filename.startswith("local_"):
                alternate = os.path.join(directory, filename[len("local_") :])
                if os.path.exists(alternate):
                    return alternate
        return value

    return SimpleNamespace(
        base_config=_localize_base_config(
            raw_args.get(
                "base_config",
                str(
                    REPO_ROOT
                    / "configs/local_ds1_frozenprobe64_fh16_aradd_scale128x4_lr2e3_20260428.yaml"
                ),
            ),
        ),
        embed_dim=int(raw_args.get("embed_dim", 64)),
        n_layers=int(raw_args.get("n_layers", 2)),
        n_heads=int(raw_args.get("n_heads", 4)),
        dropout=float(raw_args.get("dropout", 0.0)),
        head_hidden_dim=int(raw_args.get("head_hidden_dim", 128)),
        case_dim=int(raw_args.get("case_dim", 0)),
        phyla_dim=int(raw_args.get("phyla_dim", 256)),
        phyla_use_leaf_tokens=bool(raw_args.get("phyla_use_leaf_tokens", True)),
        phyla_use_split_tokens=bool(raw_args.get("phyla_use_split_tokens", False)),
        phyla_embedding_dir=_localize_repo_path(
            raw_args.get(
                "phyla_embedding_dir",
                str(REPO_ROOT / "analysis/full_sanity_fixedpair_20260401/ds_phyla_embeddings_20260428"),
            )
        ),
    )


def _source_mask_for_node(node, *, n_leaves: int) -> int:
    root_leaf = int(n_leaves) - 1
    biological_bits = max(int(n_leaves) - 1, 0)
    full_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
    raw_indices = []
    for leaf in node.iter_leaves():
        value = int(str(leaf.name))
        raw_indices.append(value - 1 if 1 <= value <= int(n_leaves) else value)
    indices = set(raw_indices)
    if root_leaf in indices:
        indices = set(range(int(n_leaves))) - indices
    mask = 0
    for index in indices:
        if 0 <= int(index) < biological_bits:
            mask |= 1 << int(index)
    return int(mask if mask else full_mask)


def _apply_relaxer_preserve_topology(
    relaxer,
    sample,
    device,
    *,
    scale: float = 1.0,
    edge_floor: float = 1e-8,
    phyla_bank=None,
):
    relaxer.eval()
    newick = str(sample["newick_tree"])
    tokenized = relaxer.model.tokenizer([newick])
    tokenized = _move_tokenized_batch_to_device(tokenized, device)
    phyla_embeddings = None
    if phyla_bank:
        phyla_embeddings = phyla_bank[str(sample["dataset_id"]).upper()]
    with torch.inference_mode():
        edge_outputs = relaxer.model(
            tokenized,
            torch.tensor([4.0], dtype=torch.float32, device=device),
            phyla_embeddings=phyla_embeddings,
            return_leafs_only=False,
            return_edges_only=True,
            return_edge_features=True,
        )
        _edge_values, _edge_pad_mask, edge_features = edge_outputs
        edge_split_masks = tokenized[-1]
        entries, _lengths, n_leaves, _mapping = _branch_relax_entries_for_tree(
            relaxer,
            newick,
            edge_split_masks[0],
            labels=None,
        )
        if not entries:
            return newick, {"applied": False}
        features = torch.stack(
            [edge_features[0, entry["edge_index"]] for entry in entries],
            dim=0,
        )
        numeric = torch.tensor(
            [entry["numeric"] for entry in entries],
            dtype=torch.float32,
            device=device,
        )
        case_indices = torch.full(
            (len(entries),),
            int(sample["case_index"]),
            dtype=torch.long,
            device=device,
        )
        deltas = relaxer.head(features, numeric, case_indices).detach().cpu().numpy()

    delta_by_source_mask = {
        int(entry.get("source_mask", entry["mask"])): float(delta)
        for entry, delta in zip(entries, deltas)
    }
    tree = EteTree(_ensure_semicolon(newick), format=1, quoted_node_names=True)
    applied = 0
    max_abs_delta = 0.0
    for node in tree.traverse("postorder"):
        if node.is_root():
            continue
        try:
            source_mask = _source_mask_for_node(node, n_leaves=int(n_leaves))
        except Exception:
            continue
        delta = delta_by_source_mask.get(int(source_mask))
        if delta is None:
            continue
        before = float(node.dist)
        after = max(before + float(scale) * float(delta), float(edge_floor))
        node.dist = after
        applied += 1
        max_abs_delta = max(max_abs_delta, abs(after - before))
    return tree.write(format=1), {
        "applied": bool(applied),
        "applied_edge_count": int(applied),
        "max_abs_delta": float(max_abs_delta),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="DS1")
    parser.add_argument("--label", required=True)
    parser.add_argument("--input-tree-list", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--relax-scale", type=float, default=1.0)
    args = parser.parse_args()

    dataset_id = str(args.dataset_id).upper()
    input_path = Path(args.input_tree_list).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = _read_tree_list(input_path)
    device = torch.device(args.device)
    checkpoint = torch.load(Path(args.checkpoint).resolve(), map_location=device)
    ckpt_args = _checkpoint_args(checkpoint)
    cfg = _small_model_config(ckpt_args.base_config, ckpt_args)
    model = return_model(cfg).to(device)
    head = BranchDeltaHead(
        int(model.embed_dim),
        hidden_dim=int(ckpt_args.head_hidden_dim),
        case_dim=int(ckpt_args.case_dim),
        num_cases=len(trees),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    head.load_state_dict(checkpoint["head"])
    relaxer = StandaloneRelaxer(model, head).to(device)
    relaxer.eval()

    phyla_bank = _load_phyla_embedding_bank(
        [dataset_id],
        ckpt_args.phyla_embedding_dir,
        device,
    )
    scorer = GenericJCLikelihood(dataset_id=dataset_id)

    rows = []
    relaxed_trees: list[str] = []
    for index, tree in enumerate(trees):
        before = scorer.log_likelihood(tree)
        relaxed_tree, info = _apply_relaxer_preserve_topology(
            relaxer,
            {
                "newick_tree": tree,
                "dataset_id": dataset_id,
                "case_index": index,
            },
            device,
            scale=float(args.relax_scale),
            phyla_bank=phyla_bank,
        )
        relaxed_tree = _ensure_semicolon(relaxed_tree)
        after = scorer.log_likelihood(relaxed_tree)
        relaxed_trees.append(relaxed_tree)
        rows.append(
            {
                "index": int(index),
                "before_log_likelihood": float(before),
                "after_log_likelihood": float(after),
                "delta_log_likelihood": float(after - before),
                "applied": bool(info.get("applied")),
                "max_abs_delta": float(info.get("max_abs_delta", 0.0)),
            }
        )

    before_values = [row["before_log_likelihood"] for row in rows]
    after_values = [row["after_log_likelihood"] for row in rows]
    delta_values = [row["delta_log_likelihood"] for row in rows]
    summary = {
        "label": str(args.label),
        "dataset_id": dataset_id,
        "input_tree_list": str(input_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "tree_count": int(len(rows)),
        "relax_scale": float(args.relax_scale),
        "applied_count": int(sum(1 for row in rows if row["applied"])),
        "before_log_likelihood": _summary(before_values),
        "after_log_likelihood": _summary(after_values),
        "delta_log_likelihood": _summary(delta_values),
        "num_improved": int(sum(1 for value in delta_values if value > 0.0)),
        "num_worse": int(sum(1 for value in delta_values if value < 0.0)),
    }

    relaxed_path = out_dir / f"{args.label}_relaxed_trees.txt"
    rows_path = out_dir / f"{args.label}_per_tree.jsonl"
    summary_path = out_dir / f"{args.label}_summary.json"
    relaxed_path.write_text("\n".join(relaxed_trees) + "\n")
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
