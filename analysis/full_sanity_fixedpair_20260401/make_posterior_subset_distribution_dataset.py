#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from ete3 import Tree as EteTree

ROOT = Path("/home/yektefai/PhylaFlow")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import TreeDataset  # noqa: E402
from utils.random_tree import Tree as RandomTree  # noqa: E402


def _sorted_numeric_names(names):
    return sorted((str(name) for name in names), key=lambda x: int(x))


def _dummy_nexus_text(num_leaves: int, nchar: int = 128) -> str:
    rows = "\n".join(f"    {i}  {'A' * nchar}" for i in range(num_leaves))
    return (
        "#NEXUS\n\n"
        "BEGIN DATA;\n"
        f"    DIMENSIONS NTAX={num_leaves} NCHAR={nchar};\n"
        "    FORMAT DATATYPE=DNA MISSING=? GAP=-;\n"
        "    MATRIX\n"
        f"{rows}\n"
        "    ;\n"
        "END;\n"
    )


def _tree_line(idx: int, newick: str) -> str:
    return f"tree STATE_{idx} = {newick}"


def _induce_and_relabel(newick: str, chosen_original: list[str], relabel: dict[str, str]) -> str:
    tree = EteTree(newick, format=1)
    tree.prune(chosen_original, preserve_branch_length=True)
    for leaf in tree.iter_leaves():
        leaf.name = relabel[str(leaf.name)]
    out = tree.write(format=1)
    if not out.endswith(";"):
        out += ";"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-nexus-root", required=True)
    parser.add_argument("--source-mrbayes-root", required=True)
    parser.add_argument("--num-leaves", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--num-random-starts", type=int, default=64)
    args = parser.parse_args()

    source_ds = TreeDataset(args.source_nexus_root, args.source_mrbayes_root)
    if len(source_ds) == 0:
        raise ValueError("Source dataset is empty.")

    posterior_trees = source_ds.return_posterior_trees(0)
    if not posterior_trees:
        raise ValueError("Source dataset returned no posterior trees.")

    first_tree = EteTree(posterior_trees[0], format=1)
    full_leaf_names = _sorted_numeric_names(leaf.name for leaf in first_tree.iter_leaves())
    if args.num_leaves >= len(full_leaf_names):
        raise ValueError(
            f"Requested {args.num_leaves} leaves, but source tree only has {len(full_leaf_names)} leaves."
        )

    rng = random.Random(int(args.seed))
    chosen_original = _sorted_numeric_names(
        rng.sample(full_leaf_names, int(args.num_leaves))
    )
    relabel = {old: str(i) for i, old in enumerate(chosen_original)}

    induced_trees = [
        _induce_and_relabel(newick, chosen_original, relabel)
        for newick in posterior_trees
    ]

    analysis_dir = ROOT / "analysis" / "full_sanity_fixedpair_20260401"
    dataset_dir = analysis_dir / f"{args.tag}_dataset"
    nexus_dir = dataset_dir / "nexus"
    runs_dir = dataset_dir / "runs" / args.tag
    target_bank_dir = analysis_dir / f"{args.tag}_target_bank"
    start_bank_dir = analysis_dir / f"{args.tag}_random_start_bank"
    nexus_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    target_bank_dir.mkdir(parents=True, exist_ok=True)
    start_bank_dir.mkdir(parents=True, exist_ok=True)

    (nexus_dir / f"{args.tag}.nex").write_text(_dummy_nexus_text(int(args.num_leaves)))

    run1_lines = ["#NEXUS", "begin trees;"]
    run2_lines = ["#NEXUS", "begin trees;"]
    for idx, tree_newick in enumerate(induced_trees):
        line = _tree_line(idx, tree_newick)
        if idx % 2 == 0:
            run1_lines.append(line)
        else:
            run2_lines.append(line)
    run1_lines.append("end;")
    run2_lines.append("end;")
    (runs_dir / f"{args.tag}_DNA.run1.t").write_text("\n".join(run1_lines) + "\n")
    (runs_dir / f"{args.tag}_DNA.run2.t").write_text("\n".join(run2_lines) + "\n")

    generated_ds = TreeDataset(str(nexus_dir), str(dataset_dir / "runs"))
    generated_posterior = generated_ds.return_posterior_trees(0)
    target_json_paths = []
    for idx, tree_newick in enumerate(generated_posterior):
        path = target_bank_dir / f"target_{idx:04d}.json"
        path.write_text(
            json.dumps(
                {
                    "target_tree": tree_newick,
                    "selected_original_labels": chosen_original,
                    "source_index": idx,
                },
                indent=2,
            )
            + "\n"
        )
        target_json_paths.append(str(path))

    start_json_paths = []
    for idx in range(int(args.num_random_starts)):
        random.seed(int(args.seed) + idx)
        start_newick = str(RandomTree(num_leaves=int(args.num_leaves), random=True))
        if not start_newick.endswith(";"):
            start_newick += ";"
        path = start_bank_dir / f"start_{idx:04d}.json"
        path.write_text(
            json.dumps(
                {
                    "start_tree": start_newick,
                    "seed": int(args.seed) + idx,
                },
                indent=2,
            )
            + "\n"
        )
        start_json_paths.append(str(path))

    summary = {
        "tag": args.tag,
        "num_leaves": int(args.num_leaves),
        "seed": int(args.seed),
        "selected_original_labels": chosen_original,
        "source_posterior_count": len(posterior_trees),
        "generated_posterior_count": len(generated_posterior),
        "num_random_starts": int(args.num_random_starts),
        "dataset_dir": str(dataset_dir),
        "target_bank_dir": str(target_bank_dir),
        "start_bank_dir": str(start_bank_dir),
        "target_json_paths": target_json_paths,
        "start_json_paths": start_json_paths,
    }
    summary_path = analysis_dir / f"{args.tag}_distribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
