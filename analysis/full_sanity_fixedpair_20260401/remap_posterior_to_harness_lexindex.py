#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path("/home/yektefai/PhylaFlow")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ete3 import Tree as EteTree  # noqa: E402

from data.dataset import TreeDataset  # noqa: E402
from utils.metric_utils import calculate_norm_rf  # noqa: E402


def build_harness_lexicographic_ordering_map(reference_tree_newick: str) -> Dict[str, str]:
    tree = EteTree(reference_tree_newick, format=1)
    leaves = tree.get_leaves()
    leaves.sort(key=lambda leaf: str(leaf.name))
    return {str(leaf.name): str(i) for i, leaf in enumerate(leaves)}


def build_numeric_to_harness_lexicographic_ordering_map(
    reference_tree_newick: str,
) -> Dict[str, str]:
    tree = EteTree(reference_tree_newick, format=1)
    original_names = [str(leaf.name) for leaf in tree.get_leaves()]
    numeric_sorted = sorted(original_names, key=lambda name: int(str(name)))
    lex_sorted = sorted(original_names, key=lambda name: str(name))
    original_to_numeric = {str(name): str(i) for i, name in enumerate(numeric_sorted)}
    original_to_lex = {str(name): str(i) for i, name in enumerate(lex_sorted)}
    return {
        original_to_numeric[str(name)]: original_to_lex[str(name)]
        for name in original_names
    }


def remap_tree_with_ordering_map(
    tree_newick: str,
    ordering_map: Dict[str, str],
) -> str:
    tree = EteTree(tree_newick, format=1)
    for leaf in tree.get_leaves():
        original_name = str(leaf.name)
        mapped_name = ordering_map.get(original_name)
        if mapped_name is None:
            raise KeyError(
                f"Leaf {original_name!r} is missing from the harness ordering map."
            )
        leaf.name = str(mapped_name)
    return tree.write(format=1)


def remap_posterior_trees_to_harness_lexindex(
    posterior_trees: Iterable[str],
    reference_tree_newick: str | None = None,
) -> Tuple[List[str], Dict[str, str]]:
    trees = [str(tree) for tree in posterior_trees]
    if not trees:
        return [], {}
    ordering_map = build_harness_lexicographic_ordering_map(
        reference_tree_newick or trees[0]
    )
    remapped = [remap_tree_with_ordering_map(tree, ordering_map) for tree in trees]
    return remapped, ordering_map


def load_posterior_trees(
    posterior_root: str,
    dataset_id: str,
    trprobs_sample_count_per_file: int = 1000,
) -> List[str]:
    dataset = TreeDataset(
        nexus_root="unused",
        mrbayes_root="unused",
        posterior_trprobs_root=str(posterior_root),
        posterior_dataset_id=str(dataset_id),
        trprobs_sample_count_per_file=int(trprobs_sample_count_per_file),
    )
    return dataset.return_posterior_trees(str(dataset_id))


def _lookup_nested_key(payload: Any, dotted_key: str) -> Any:
    current = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            raise KeyError(
                f"Cannot descend into {part!r}; current value is not a dict."
            )
        current = current[part]
    return current


def _rf_summary(compare_tree: str, posterior_trees: List[str]) -> Dict[str, float]:
    if not posterior_trees:
        return {
            "mean_rf_norm": float("nan"),
            "min_rf_norm": float("nan"),
            "max_rf_norm": float("nan"),
        }
    values = [float(calculate_norm_rf(compare_tree, tree)) for tree in posterior_trees]
    return {
        "mean_rf_norm": float(sum(values) / len(values)),
        "min_rf_norm": float(min(values)),
        "max_rf_norm": float(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior-root", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--trprobs-sample-count", type=int, default=1000)
    parser.add_argument(
        "--compare-json",
        default=None,
        help="Optional JSON artifact containing a tree to compare against the posterior.",
    )
    parser.add_argument(
        "--compare-key",
        default="harness.final_tree",
        help="Dotted key inside --compare-json to extract the comparison Newick.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Prints to stdout either way.",
    )
    args = parser.parse_args()

    raw_posterior = load_posterior_trees(
        posterior_root=str(args.posterior_root),
        dataset_id=str(args.dataset_id),
        trprobs_sample_count_per_file=int(args.trprobs_sample_count),
    )
    remapped_posterior, ordering_map = remap_posterior_trees_to_harness_lexindex(
        raw_posterior
    )

    payload: Dict[str, Any] = {
        "posterior_root": str(Path(args.posterior_root).resolve()),
        "dataset_id": str(args.dataset_id),
        "num_posterior_trees": int(len(raw_posterior)),
        "ordering_map": ordering_map,
        "raw_posterior_example": raw_posterior[0] if raw_posterior else None,
        "remapped_posterior_example": remapped_posterior[0] if remapped_posterior else None,
    }

    if args.compare_json:
        compare_payload = json.loads(Path(args.compare_json).read_text())
        compare_tree = str(_lookup_nested_key(compare_payload, str(args.compare_key)))
        payload["comparison"] = {
            "compare_json": str(Path(args.compare_json).resolve()),
            "compare_key": str(args.compare_key),
            "against_raw_posterior": _rf_summary(compare_tree, raw_posterior),
            "against_remapped_posterior": _rf_summary(compare_tree, remapped_posterior),
        }

    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
