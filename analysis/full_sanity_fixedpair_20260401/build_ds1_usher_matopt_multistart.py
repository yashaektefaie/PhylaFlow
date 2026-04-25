#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List

ROOT = Path("/home/yektefai/PhylaFlow")
ANALYSIS_DIR = ROOT / "analysis/full_sanity_fixedpair_20260401"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from build_ds1_usher_matopt_start import (  # noqa: E402
    _consensus_reference,
    _extract_newick_string_from_mat,
    _parse_translate_block,
    _run,
    _sanitize_sample_sequence,
)


def _prepare_common_inputs(
    *,
    ds1_pickle: Path,
    translation_source: Path,
    output_dir: Path,
    fa_to_vcf: Path,
) -> tuple[List[str], Dict[str, str], Path]:
    seqs: Dict[str, str] = pickle.load(open(ds1_pickle, "rb"))
    translate = _parse_translate_block(translation_source)
    ordered_names = [translate[idx] for idx in sorted(translate)]
    raw_sequences = [seqs[name].upper() for name in ordered_names]
    numeric_sequences = {
        str(idx): _sanitize_sample_sequence(seqs[name])
        for idx, name in enumerate(ordered_names, start=1)
    }
    reference = _consensus_reference(raw_sequences)

    fasta_path = output_dir / "ds1_numeric_consensus_ref_for_usher.fa"
    with fasta_path.open("w") as handle:
        handle.write(">REF\n")
        handle.write(reference + "\n")
        for idx in range(1, len(ordered_names) + 1):
            handle.write(f">{idx}\n")
            handle.write(numeric_sequences[str(idx)] + "\n")

    vcf_path = output_dir / "ds1_numeric_consensus_ref_for_usher.vcf"
    _run(
        [str(fa_to_vcf), "-ref=REF", str(fasta_path), str(vcf_path)],
        cwd=output_dir,
        log_path=output_dir / "fatovcf.log",
    )
    return ordered_names, numeric_sequences, vcf_path


def _final_parsimony_score(log_path: Path) -> int | None:
    text = log_path.read_text(errors="replace")
    matches = re.findall(r"Final Parsimony score\s+(\d+)", text)
    return int(matches[-1]) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds1-pickle", default="/home/yektefai/30272299/DS1.pickle")
    parser.add_argument(
        "--translation-source",
        default="/home/yektefai/30272299/golden_run_data_DS1-8/DS1/rep_1/DS1.trprobs",
    )
    parser.add_argument("--usher-prefix", default="/tmp/phylaflow-usher")
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/"
            "ds1_usher_matopt_mp_multistart78_20260424"
        ),
    )
    parser.add_argument("--num-starts", type=int, default=78)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--radius", type=int, default=-1)
    parser.add_argument("--max-iterations", type=int, default=1000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    usher_prefix = Path(args.usher_prefix).resolve()
    fa_to_vcf = usher_prefix / "bin/faToVcf"
    usher = usher_prefix / "bin/usher"
    mat_optimize = usher_prefix / "bin/matOptimize"
    for binary in [fa_to_vcf, usher, mat_optimize]:
        if not binary.exists():
            raise FileNotFoundError(binary)

    ordered_names, numeric_sequences, vcf_path = _prepare_common_inputs(
        ds1_pickle=Path(args.ds1_pickle).resolve(),
        translation_source=Path(args.translation_source).resolve(),
        output_dir=output_dir,
        fa_to_vcf=fa_to_vcf,
    )

    labels = sorted(numeric_sequences, key=int)
    all_pairs = list(combinations(labels, 2))
    rng = random.Random(int(args.seed))
    selected_pairs = rng.sample(all_pairs, k=min(int(args.num_starts), len(all_pairs)))

    env = dict(os.environ)
    env["OPAL_PREFIX"] = str(usher_prefix)
    rows = []
    start_tree_paths = []
    for idx, (left, right) in enumerate(selected_pairs):
        run_dir = output_dir / f"start_{idx:04d}_{left}_{right}"
        greedy_dir = run_dir / "usher_greedy"
        greedy_dir.mkdir(parents=True, exist_ok=True)
        cherry_path = run_dir / "start_cherry.nh"
        cherry_path.write_text(f"({left}:1,{right}:1);\n")
        greedy_pb = greedy_dir / "greedy.pb"
        _run(
            [
                str(usher),
                "-t",
                str(cherry_path),
                "-v",
                str(vcf_path),
                "-o",
                str(greedy_pb),
                "-d",
                str(greedy_dir),
                "-u",
                "-T",
                str(args.threads),
            ],
            cwd=run_dir,
            log_path=greedy_dir / "usher.log",
        )

        optimized_pb = run_dir / "matopt_optimized.pb"
        matopt_log = run_dir / "matoptimize.log"
        _run(
            [
                str(mat_optimize),
                "-i",
                str(greedy_pb),
                "-o",
                str(optimized_pb),
                "-T",
                str(args.threads),
                "-r",
                str(args.radius),
                "-N",
                str(args.max_iterations),
                "-n",
            ],
            cwd=run_dir,
            env=env,
            log_path=matopt_log,
        )
        optimized_newick = run_dir / "matopt_optimized.nwk"
        optimized_newick.write_text(
            _extract_newick_string_from_mat(
                optimized_pb,
                expected_leaf_count=len(ordered_names),
            )
            + "\n"
        )
        start_tree_paths.append(str(optimized_newick))
        row = {
            "index": idx,
            "left_label": left,
            "right_label": right,
            "left_taxon": ordered_names[int(left) - 1],
            "right_taxon": ordered_names[int(right) - 1],
            "cherry": str(cherry_path),
            "greedy_mat": str(greedy_pb),
            "optimized_mat": str(optimized_pb),
            "optimized_newick": str(optimized_newick),
            "final_parsimony_score": _final_parsimony_score(matopt_log),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    start_tree_list = output_dir / "start_trees.txt"
    start_tree_list.write_text("\n".join(start_tree_paths) + "\n")
    summary = {
        "num_starts": len(rows),
        "seed": int(args.seed),
        "threads": int(args.threads),
        "radius": int(args.radius),
        "max_iterations": int(args.max_iterations),
        "vcf": str(vcf_path),
        "start_tree_list": str(start_tree_list),
        "ordered_taxa": ordered_names,
        "rows": rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
