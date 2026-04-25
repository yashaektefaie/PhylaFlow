#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path("/home/yektefai/PhylaFlow")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_translate_block(path: Path) -> Dict[int, str]:
    in_translate = False
    mapping: Dict[int, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not in_translate:
            if line.lower().startswith("translate"):
                in_translate = True
            continue
        if not line:
            continue
        done = line.endswith(";")
        line = line.rstrip(";").rstrip(",")
        match = re.match(r"(\d+)\s+(.+)$", line)
        if match:
            mapping[int(match.group(1))] = match.group(2).strip().strip(",")
        if done:
            break
    if not mapping:
        raise ValueError(f"Failed to parse translate block from {path}")
    return mapping


def _consensus_reference(seqs: List[str]) -> str:
    if not seqs:
        raise ValueError("No sequences provided.")
    length = len(seqs[0])
    ref_chars: List[str] = []
    for idx in range(length):
        counts = Counter(seq[idx] for seq in seqs if seq[idx] in "ACGT")
        if counts:
            ref_chars.append(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0])
        else:
            ref_chars.append("A")
    return "".join(ref_chars)


def _sanitize_sample_sequence(seq: str) -> str:
    return "".join(base if base in "ACGT" else "N" for base in seq.upper())


def _closest_pair(seqs_by_label: Dict[str, str]) -> Tuple[str, str, float, int]:
    best: Tuple[float, int, str, str] | None = None
    labels = sorted(seqs_by_label, key=lambda value: int(value))
    for idx, left in enumerate(labels):
        left_seq = seqs_by_label[left]
        for right in labels[idx + 1 :]:
            right_seq = seqs_by_label[right]
            comparable = 0
            diff = 0
            for a, b in zip(left_seq, right_seq):
                if a not in "ACGT" or b not in "ACGT":
                    continue
                comparable += 1
                if a != b:
                    diff += 1
            if comparable == 0:
                continue
            rate = float(diff) / float(comparable)
            score = (rate, diff, left, right)
            if best is None or score < best:
                best = score
    if best is None:
        raise ValueError("Could not choose a closest pair.")
    rate, diff, left, right = best
    return left, right, rate, diff


def _run(cmd: List[str], *, cwd: Path, env: dict | None = None, log_path: Path) -> None:
    with log_path.open("w") as log_file:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {result.returncode}: {' '.join(cmd)}. See {log_path}"
        )


def _extract_newick_string_from_mat(path: Path, expected_leaf_count: int) -> str:
    from ete3 import Tree as EteTree

    raw = path.read_bytes()
    candidates: List[str] = []
    start = 0
    while True:
        start = raw.find(b"(", start)
        if start < 0:
            break
        end = raw.find(b";", start)
        if end < 0:
            break
        chunk = raw[start : end + 1]
        start += 1
        try:
            text = chunk.decode("ascii")
        except UnicodeDecodeError:
            continue
        if any(ord(char) < 32 for char in text):
            continue
        try:
            tree = EteTree(text, format=1)
        except Exception:
            continue
        if len(tree.get_leaf_names()) == expected_leaf_count:
            candidates.append(text)
    if not candidates:
        raise ValueError(f"Could not recover a Newick string from {path}")
    return max(candidates, key=len)


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
            "ds1_usher_matopt_mp_20260424"
        ),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--radius", type=int, default=-1)
    parser.add_argument("--max-iterations", type=int, default=1000)
    args = parser.parse_args()

    ds1_pickle = Path(args.ds1_pickle).resolve()
    translation_source = Path(args.translation_source).resolve()
    usher_prefix = Path(args.usher_prefix).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fa_to_vcf = usher_prefix / "bin" / "faToVcf"
    usher = usher_prefix / "bin" / "usher"
    mat_optimize = usher_prefix / "bin" / "matOptimize"
    mat_utils = usher_prefix / "bin" / "matUtils"
    for binary in [fa_to_vcf, usher, mat_optimize, mat_utils]:
        if not binary.exists():
            raise FileNotFoundError(binary)

    seqs: Dict[str, str] = pickle.load(open(ds1_pickle, "rb"))
    translate = _parse_translate_block(translation_source)
    ordered_names = [translate[idx] for idx in sorted(translate)]
    raw_sequences = [seqs[name].upper() for name in ordered_names]
    if len({len(seq) for seq in raw_sequences}) != 1:
        raise ValueError("DS1 sequences are not all the same length.")

    numeric_sequences = {
        str(idx): _sanitize_sample_sequence(seqs[name])
        for idx, name in enumerate(ordered_names, start=1)
    }
    reference = _consensus_reference(raw_sequences)
    fasta_path = output_dir / "ds1_numeric_consensus_ref_for_usher.fa"
    with fasta_path.open("w") as handle:
        handle.write(">REF\n")
        handle.write(reference + "\n")
        for idx, name in enumerate(ordered_names, start=1):
            handle.write(f">{idx}\n")
            handle.write(numeric_sequences[str(idx)] + "\n")

    vcf_path = output_dir / "ds1_numeric_consensus_ref_for_usher.vcf"
    _run(
        [str(fa_to_vcf), "-ref=REF", str(fasta_path), str(vcf_path)],
        cwd=output_dir,
        log_path=output_dir / "fatovcf.log",
    )

    left, right, pair_rate, pair_diff = _closest_pair(numeric_sequences)
    cherry_path = output_dir / "closest_pair_start_cherry.nh"
    cherry_path.write_text(f"({left}:1,{right}:1);\n")

    greedy_dir = output_dir / "usher_greedy"
    greedy_dir.mkdir(exist_ok=True)
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
        cwd=output_dir,
        log_path=greedy_dir / "usher.log",
    )

    optimized_pb = output_dir / "matopt_optimized.pb"
    env = dict(os.environ)
    env["OPAL_PREFIX"] = str(usher_prefix)
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
        cwd=output_dir,
        env=env,
        log_path=output_dir / "matoptimize.log",
    )

    optimized_newick = output_dir / "matopt_optimized.nwk"
    extraction_method = "matutils_extract"
    extraction_error = None
    try:
        _run(
            [str(mat_utils), "extract", "-i", str(optimized_pb), "-t", str(optimized_newick)],
            cwd=output_dir,
            log_path=output_dir / "matutils_extract.log",
        )
    except Exception as exc:  # noqa: BLE001
        extraction_method = "protobuf_embedded_newick"
        extraction_error = str(exc)
        optimized_newick.write_text(
            _extract_newick_string_from_mat(
                optimized_pb,
                expected_leaf_count=len(ordered_names),
            )
            + "\n"
        )

    summary = {
        "ds1_pickle": str(ds1_pickle),
        "translation_source": str(translation_source),
        "usher_prefix": str(usher_prefix),
        "ordered_taxa": ordered_names,
        "fasta": str(fasta_path),
        "vcf": str(vcf_path),
        "start_cherry": str(cherry_path),
        "closest_pair": {
            "left_label": left,
            "right_label": right,
            "left_taxon": ordered_names[int(left) - 1],
            "right_taxon": ordered_names[int(right) - 1],
            "hamming_rate_ignore_missing": pair_rate,
            "hamming_diff_ignore_missing": pair_diff,
        },
        "greedy_mat": str(greedy_pb),
        "optimized_mat": str(optimized_pb),
        "optimized_newick": str(optimized_newick),
        "optimized_newick_extraction_method": extraction_method,
        "optimized_newick_extraction_error": extraction_error,
        "threads": int(args.threads),
        "radius": int(args.radius),
        "max_iterations": int(args.max_iterations),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
