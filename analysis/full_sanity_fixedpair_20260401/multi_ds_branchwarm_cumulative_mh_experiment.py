#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import pickle
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from ete3 import Tree as EteTree

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "analysis/full_sanity_fixedpair_20260401"
DATA_ROOT = Path(os.environ.get("PHYLAFLOW_30272299_ROOT", "/ewsc/yektefai/30272299"))
GOLDEN_ROOT = DATA_ROOT / "golden_run_data_DS1-8"
SOURCE_DIR = ANALYSIS_DIR / "ds2_ds3_ds4_ds6_ds7_ds8_checkpoint_samples_20260426"
SPLIT_DIVERSE_DIR = ANALYSIS_DIR / "splitkl_diverse_mrbayes_100k_20260427"
CHECKPOINT_MB_DIR = ANALYSIS_DIR / "checkpoint_mrbayes_generation_curves_100k_20260427"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.full_sanity_fixedpair_20260401.benchmark_mrbayes_fixed_start_generic import (  # noqa: E402
    _build_alignment_nexus,
    _collect_tree_files,
    _extract_newicks,
    _load_posterior_trees,
    _parse_translate_block,
    _sanitize_start_tree,
    _tree_distribution_metrics_from_counts,
)
try:
    from analysis.full_sanity_fixedpair_20260401.split_guided_likelihood_mh_experiment import (  # noqa: E402
        IUPAC_MASKS,
        _mixture_proposal_log_probs,
    )
except ModuleNotFoundError:
    IUPAC_MASKS = {
        "A": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "C": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        "G": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
        "T": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "U": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "?": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        "-": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        "N": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }

    def _mixture_proposal_log_probs(*_args, **_kwargs):
        raise RuntimeError(
            "split_guided_likelihood_mh_experiment.py is unavailable; "
            "mixture proposal MH is not available in this checkout"
        )
try:
    from analysis.full_sanity_fixedpair_20260401.split_guided_local_search_experiment import (  # noqa: E402
        _leaf_sort_key,
        _nni_neighbors,
        _strip_internal_names,
    )
except ModuleNotFoundError:
    def _leaf_sort_key(value):
        try:
            return (0, int(value))
        except Exception:
            return (1, str(value))

    def _nni_neighbors(*_args, **_kwargs):
        raise RuntimeError(
            "split_guided_local_search_experiment.py is unavailable in this checkout"
        )

    def _strip_internal_names(tree):
        for node in tree.traverse():
            if not node.is_leaf():
                node.name = ""
        return tree

try:
    from analysis.full_sanity_fixedpair_20260401.split_guided_proxy_mh_experiment import (  # noqa: E402
        _choose_from_log_probs,
        _split_log_weights,
        _tree_split_set,
    )
except ModuleNotFoundError:
    def _choose_from_log_probs(*_args, **_kwargs):
        raise RuntimeError(
            "split_guided_proxy_mh_experiment.py is unavailable in this checkout"
        )

    def _split_log_weights(*_args, **_kwargs):
        raise RuntimeError(
            "split_guided_proxy_mh_experiment.py is unavailable in this checkout"
        )

    def _tree_split_set(*_args, **_kwargs):
        raise RuntimeError(
            "split_guided_proxy_mh_experiment.py is unavailable in this checkout"
        )
from analysis.full_sanity_fixedpair_20260401.split_guided_start_experiment import (  # noqa: E402
    _ensure_semicolon,
    _split_counts,
    _split_kl_from_counts,
)
from utils.metric_utils import canonicalize_topology_newick  # noqa: E402


Split = Tuple[str, ...]


SPLITKL_GUIDE_LISTS = {
    "DS4": SOURCE_DIR
    / "ds4_best_splitkl_step038000_trees"
    / "ds4_best_splitkl_step038000_sampled_start_trees.txt",
    "DS7": SOURCE_DIR
    / "ds7_best_splitkl_topokl_step010000_trees"
    / "ds7_best_splitkl_topokl_step010000_sampled_start_trees.txt",
    "DS8": SOURCE_DIR
    / "ds8_best_splitkl_topokl_step006000_trees"
    / "ds8_best_splitkl_topokl_step006000_sampled_start_trees.txt",
}


def _read_tree_list(path: Path) -> List[str]:
    trees: List[str] = []
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


def _split_diverse_start_list(dataset_id: str) -> Path:
    path = (
        SPLIT_DIVERSE_DIR
        / dataset_id.lower()
        / "starts"
        / "splitkl_diverse_start_trees.txt"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _guide_tree_list(dataset_id: str) -> Path:
    path = SPLITKL_GUIDE_LISTS[dataset_id]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _dataset_paths(dataset_id: str) -> tuple[Path, Path]:
    dataset_pickle = DATA_ROOT / f"{dataset_id}.pickle"
    golden_root = GOLDEN_ROOT / dataset_id
    if not dataset_pickle.exists():
        raise FileNotFoundError(dataset_pickle)
    if not golden_root.exists():
        raise FileNotFoundError(golden_root)
    return dataset_pickle, golden_root


def _num_taxa(dataset_id: str) -> int:
    _dataset_pickle, golden_root = _dataset_paths(dataset_id)
    translate = _parse_translate_block(golden_root / "rep_1" / f"{dataset_id}.trprobs")
    return len(translate)


def _encode_sequence(sequence: str) -> np.ndarray:
    return np.asarray(
        [
            IUPAC_MASKS.get(str(raw_base).upper(), IUPAC_MASKS["?"])
            for raw_base in sequence
        ],
        dtype=np.float64,
    )


class GenericJCLikelihood:
    def __init__(
        self,
        *,
        dataset_id: str,
        branch_length_floor: float = 1e-8,
    ) -> None:
        dataset_pickle, golden_root = _dataset_paths(dataset_id)
        translation_path = golden_root / "rep_1" / f"{dataset_id}.trprobs"

        with dataset_pickle.open("rb") as handle:
            sequences: Dict[str, str] = pickle.load(handle)
        translate = _parse_translate_block(translation_path)
        ordered_taxa = [translate[index] for index in sorted(translate)]
        missing = [name for name in ordered_taxa if name not in sequences]
        if missing:
            raise ValueError(f"{dataset_pickle} is missing taxa: {missing}")

        lengths = {len(sequences[name]) for name in ordered_taxa}
        if len(lengths) != 1:
            raise ValueError(f"Sequences are not aligned to one length: {sorted(lengths)}")

        self.n_sites = int(next(iter(lengths)))
        self.branch_length_floor = float(branch_length_floor)
        self.transition_cache: Dict[float, np.ndarray] = {}
        self.leaf_vectors: Dict[str, np.ndarray] = {
            str(index): _encode_sequence(sequences[name])
            for index, name in enumerate(ordered_taxa, start=1)
        }

    def _leaf_vector(self, name: str) -> np.ndarray:
        key = str(name)
        cached = self.leaf_vectors.get(key)
        if cached is not None:
            return cached
        try:
            one_based = str(int(key) + 1)
        except Exception:
            one_based = ""
        cached = self.leaf_vectors.get(one_based)
        if cached is None:
            raise ValueError(f"Unknown leaf label {name!r}; expected numeric labels")
        return cached

    def _transition(self, branch_length: float) -> np.ndarray:
        length = max(float(branch_length), self.branch_length_floor)
        cached = self.transition_cache.get(length)
        if cached is not None:
            return cached
        decay = math.exp(-4.0 * length / 3.0)
        same = 0.25 + 0.75 * decay
        different = 0.25 - 0.25 * decay
        matrix = np.full((4, 4), different, dtype=np.float64)
        np.fill_diagonal(matrix, same)
        self.transition_cache[length] = matrix
        return matrix

    def log_likelihood(self, newick: str) -> float:
        tree = EteTree(_ensure_semicolon(newick), format=1, quoted_node_names=True)
        values: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        zeros = np.zeros(self.n_sites, dtype=np.float64)

        for node in tree.traverse("postorder"):
            if node.is_leaf():
                values[id(node)] = (self._leaf_vector(str(node.name)), zeros)
                continue

            clv = np.ones((self.n_sites, 4), dtype=np.float64)
            log_scale = np.zeros(self.n_sites, dtype=np.float64)
            for child in node.children:
                child_clv, child_log_scale = values[id(child)]
                transition = self._transition(float(child.dist))
                contribution = child_clv @ transition.T
                clv *= contribution
                log_scale += child_log_scale

            site_scale = np.max(clv, axis=1)
            site_scale = np.maximum(site_scale, np.finfo(np.float64).tiny)
            clv = clv / site_scale[:, None]
            log_scale += np.log(site_scale)
            values[id(node)] = (clv, log_scale)

        root_clv, root_log_scale = values[id(tree)]
        site_likelihood = 0.25 * np.sum(root_clv, axis=1)
        site_likelihood = np.maximum(site_likelihood, np.finfo(np.float64).tiny)
        return float(np.sum(np.log(site_likelihood) + root_log_scale))


def _jc_branch_warmup_nexus(
    *,
    alignment_nexus_text: str,
    start_tree: str,
    filename_prefix: str,
    ngen: int,
    samplefreq: int,
    printfreq: int,
    num_taxa: int,
) -> str:
    safe_start = _sanitize_start_tree(start_tree, num_taxa=int(num_taxa))
    return (
        "#NEXUS\n\n"
        + alignment_nexus_text
        + "\nBEGIN TREES;\n"
        + f"    TREE init = {safe_start}\n"
        + "END;\n\n"
        + "BEGIN MRBAYES;\n"
        + "    set autoclose=yes nowarn=yes quitonerror=yes;\n"
        + "    lset nst=1 rates=equal;\n"
        + "    prset statefreqpr=fixed(equal);\n"
        + "    startvals tau=init;\n"
        + "    startvals v=init;\n"
        + f"    mcmcp filename={filename_prefix} nruns=1 nchains=1 ngen={ngen} "
        + f"samplefreq={samplefreq} printfreq={printfreq} diagnfreq={printfreq} "
        + "checkpoint=no append=no starttree=current nperts=0;\n"
        + "    propset extspr(tau,v)$prob=0 exttbr(tau,v)$prob=0 "
        + "nni(tau,v)$prob=0 parsspr(tau,v)$prob=0;\n"
        + "    mcmc;\n"
        + "END;\n"
    )


def _run_one_branch_warmup(task: Mapping[str, object]) -> dict:
    run_dir = Path(str(task["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    tree_path = run_dir / "warmed_final.nwk"
    if tree_path.exists() and not bool(task.get("force", False)):
        return {
            "run_index": int(task["run_index"]),
            "tree": tree_path.read_text().strip(),
        }

    nexus_path = run_dir / "branch_warmup.nex"
    prefix = run_dir / "run"
    nexus_path.write_text(
        _jc_branch_warmup_nexus(
            alignment_nexus_text=str(task["alignment_nexus_text"]),
            start_tree=str(task["start_tree"]),
            filename_prefix="run",
            ngen=int(task["ngen"]),
            samplefreq=int(task["samplefreq"]),
            printfreq=int(task["printfreq"]),
            num_taxa=int(task["num_taxa"]),
        )
    )
    log_path = run_dir / "stdout.log"
    with log_path.open("w") as log_file:
        result = subprocess.run(
            [str(task["mrbayes_bin"]), str(nexus_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(run_dir),
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"MrBayes branch warmup failed for {run_dir}; see {log_path}")

    trees: List[str] = []
    for tree_file in _collect_tree_files(prefix):
        trees.extend(_extract_newicks(tree_file))
    if not trees:
        raise RuntimeError(f"No branch-warmed tree samples found under {run_dir}")
    final_tree = _ensure_semicolon(trees[-1])
    tree_path.write_text(final_tree + "\n")
    return {"run_index": int(task["run_index"]), "tree": final_tree}


def _run_branch_warmups(
    *,
    dataset_id: str,
    start_trees: Sequence[str],
    work_dir: Path,
    ngen: int,
    samplefreq: int,
    printfreq: int,
    max_workers: int,
    mrbayes_bin: Path,
    force: bool,
) -> List[str]:
    dataset_pickle, golden_root = _dataset_paths(dataset_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    alignment_path = work_dir / f"{dataset_id}_jc_alignment.nex"
    alignment_text, ordered_taxa = _build_alignment_nexus(
        dataset_pickle=dataset_pickle,
        translation_source=golden_root / "rep_1" / f"{dataset_id}.trprobs",
        output_path=alignment_path,
    )
    tasks = [
        {
            "run_index": idx,
            "run_dir": work_dir / f"run_{idx:04d}",
            "alignment_nexus_text": alignment_text,
            "start_tree": tree,
            "ngen": int(ngen),
            "samplefreq": int(samplefreq),
            "printfreq": int(printfreq),
            "num_taxa": len(ordered_taxa),
            "mrbayes_bin": str(mrbayes_bin),
            "force": bool(force),
        }
        for idx, tree in enumerate(start_trees)
    ]
    warmed: List[str | None] = [None] * len(tasks)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        future_to_idx = {
            executor.submit(_run_one_branch_warmup, task): int(task["run_index"])
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            result = future.result()
            warmed[int(result["run_index"])] = str(result["tree"])
            completed += 1
            if completed == len(tasks) or completed % max(1, len(tasks) // 10) == 0:
                print(
                    json.dumps(
                        {
                            "dataset": dataset_id,
                            "stage": "branch_warmup",
                            "completed": completed,
                            "total": len(tasks),
                        }
                    ),
                    flush=True,
                )
    return [str(tree) for tree in warmed if tree is not None]


def _sort_tree_children(tree: EteTree) -> None:
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            continue
        node.children.sort(
            key=lambda child: ",".join(
                sorted(
                    (str(name) for name in child.get_leaf_names()),
                    key=lambda x: _leaf_sort_key(x),
                )
            )
        )


def _state_key(newick: str) -> str:
    tree = EteTree(_ensure_semicolon(newick), format=1, quoted_node_names=True)
    leaves = list(tree.iter_leaves())
    if leaves:
        tree.set_outgroup(min(leaves, key=lambda leaf: _leaf_sort_key(str(leaf.name))))
    tree.resolve_polytomy(default_dist=1e-6, default_support=0.0, recursive=True)
    _strip_internal_names(tree)
    for node in tree.traverse():
        if node.up is not None and (
            not math.isfinite(float(node.dist)) or float(node.dist) <= 0.0
        ):
            node.dist = 1e-6
    _sort_tree_children(tree)
    return _ensure_semicolon(tree.write(format=1))


class BranchStateCache:
    def __init__(
        self,
        split_logp: Mapping[Split, float],
        *,
        floor: float,
        likelihood: GenericJCLikelihood,
    ) -> None:
        self.split_logp = dict(split_logp)
        self.floor_logp = math.log(float(floor))
        self.likelihood = likelihood
        self.key_to_tree: Dict[str, str] = {}
        self.key_to_topology_key: Dict[str, str] = {}
        self.topology_to_tree: Dict[str, str] = {}
        self.topology_to_splits: Dict[str, frozenset[Split]] = {}
        self.topology_to_score: Dict[str, float] = {}
        self.key_to_log_likelihood: Dict[str, float] = {}
        self.key_to_neighbor_keys: Dict[str, List[str]] = {}

    def key(self, tree: str) -> str:
        key = _state_key(tree)
        self.key_to_tree.setdefault(key, key)
        topology_key = self.key_to_topology_key.get(key)
        if topology_key is None:
            topology_key = canonicalize_topology_newick(key)
            self.key_to_topology_key[key] = topology_key
            self.topology_to_tree.setdefault(topology_key, key)
        return key

    def topology_key(self, key: str) -> str:
        return self.key_to_topology_key[key]

    def splits(self, key: str) -> frozenset[Split]:
        topology_key = self.topology_key(key)
        cached = self.topology_to_splits.get(topology_key)
        if cached is not None:
            return cached
        cached = _tree_split_set(self.topology_to_tree[topology_key])
        self.topology_to_splits[topology_key] = cached
        return cached

    def score(self, key: str) -> float:
        topology_key = self.topology_key(key)
        cached = self.topology_to_score.get(topology_key)
        if cached is not None:
            return cached
        score = sum(
            self.split_logp.get(split, self.floor_logp) for split in self.splits(key)
        )
        self.topology_to_score[topology_key] = float(score)
        return float(score)

    def log_target(self, key: str) -> float:
        cached = self.key_to_log_likelihood.get(key)
        if cached is not None:
            return cached
        value = self.likelihood.log_likelihood(self.key_to_tree[key])
        self.key_to_log_likelihood[key] = float(value)
        return float(value)

    def neighbors(self, key: str) -> List[str]:
        cached = self.key_to_neighbor_keys.get(key)
        if cached is not None:
            return cached
        seen_topologies: set[str] = set()
        neighbor_keys: List[str] = []
        for neighbor in _nni_neighbors(self.key_to_tree[key]):
            neighbor_key = self.key(neighbor)
            topology_key = self.topology_key(neighbor_key)
            if topology_key in seen_topologies:
                continue
            seen_topologies.add(topology_key)
            neighbor_keys.append(neighbor_key)
        self.key_to_neighbor_keys[key] = neighbor_keys
        return neighbor_keys


class CumulativeCounts:
    def __init__(self, cache: BranchStateCache) -> None:
        self.cache = cache
        self.tree_counts: Counter = Counter()
        self.split_counts: Counter = Counter()
        self.sample_count = 0

    def add_keys(self, keys: Sequence[str]) -> None:
        for key in keys:
            topology_key = self.cache.topology_key(key)
            self.tree_counts[topology_key] += 1
            self.split_counts.update(self.cache.splits(key))
            self.sample_count += 1

    def metrics(self, *, posterior_counts: Counter, posterior_split_counts: Counter) -> dict:
        metrics = _tree_distribution_metrics_from_counts(
            posterior_counts,
            self.tree_counts,
        )
        metrics.update(_split_kl_from_counts(posterior_split_counts, self.split_counts))
        return metrics


def _proposal_log_lookup(
    log_probs: Mapping[str, float],
    target_key: str,
    cache: BranchStateCache,
) -> float | None:
    direct = log_probs.get(target_key)
    if direct is not None:
        return float(direct)
    target_topology = cache.topology_key(target_key)
    matches = [
        float(value)
        for key, value in log_probs.items()
        if cache.topology_key(key) == target_topology
    ]
    if not matches:
        return None
    return matches[0]


def _run_cumulative_branch_state_mh(
    *,
    start_trees: Sequence[str],
    guide_trees: Sequence[str],
    dataset_id: str,
    posterior_counts: Counter,
    posterior_split_counts: Counter,
    iterations: int,
    checkpoints: Sequence[int],
    sample_every: int,
    lam: float,
    proposal_mode: str,
    rho_guided: float,
    proposal_prob: float,
    seed: int,
    floor: float,
) -> List[dict]:
    rng = random.Random(seed)
    split_logp = _split_log_weights(guide_trees, floor=floor)
    likelihood = GenericJCLikelihood(dataset_id=dataset_id)
    cache = BranchStateCache(split_logp, floor=floor, likelihood=likelihood)
    current_keys = [cache.key(tree) for tree in start_trees]
    checkpoint_set = set(int(value) for value in checkpoints)
    cumulative = CumulativeCounts(cache)
    cumulative.add_keys(current_keys)

    rows: List[dict] = []
    accepted = 0
    proposed = 0
    skipped = 0
    reverse_topology_fallbacks = 0

    def save(iteration: int) -> None:
        log_likelihoods = [cache.log_target(key) for key in current_keys]
        row = {
            "dataset": dataset_id,
            "method": (
                "branch-warm cumulative uniform NNI MH"
                if proposal_mode == "uniform"
                else "branch-warm cumulative split-guided mixed NNI MH"
            ),
            "start_source": "splitkl_diverse",
            "guide_source": "splitkl_terminal" if proposal_mode != "uniform" else "",
            "proposal_mode": proposal_mode,
            "lambda_split": float(lam),
            "rho_guided": float(rho_guided),
            "proposal_prob": float(proposal_prob),
            "iteration": int(iteration),
            "mh_samples_per_chain": int(iteration // max(1, int(sample_every)) + 1),
            "accepted": int(accepted),
            "proposed": int(proposed),
            "skipped": int(skipped),
            "accept_rate": float(accepted) / float(proposed) if proposed else 0.0,
            "proposal_attempt_rate": (
                float(proposed) / float(proposed + skipped) if (proposed + skipped) else 0.0
            ),
            "reverse_topology_fallbacks": int(reverse_topology_fallbacks),
            "mean_current_log_likelihood": (
                float(sum(log_likelihoods)) / float(len(log_likelihoods))
                if log_likelihoods
                else float("nan")
            ),
            "state_cache": int(len(cache.key_to_tree)),
            "topology_cache": int(len(cache.topology_to_tree)),
            "neighbor_cache": int(len(cache.key_to_neighbor_keys)),
            "log_likelihood_cache": int(len(cache.key_to_log_likelihood)),
        }
        row.update(
            cumulative.metrics(
                posterior_counts=posterior_counts,
                posterior_split_counts=posterior_split_counts,
            )
        )
        rows.append(row)

    if 0 in checkpoint_set:
        save(0)

    for iteration in range(1, int(iterations) + 1):
        next_keys: List[str] = []
        for current_key in current_keys:
            if rng.random() >= float(proposal_prob):
                skipped += 1
                next_keys.append(current_key)
                continue
            neighbor_keys = cache.neighbors(current_key)
            if not neighbor_keys:
                skipped += 1
                next_keys.append(current_key)
                continue

            forward_log_probs = _mixture_proposal_log_probs(
                current_key=current_key,
                neighbor_keys=neighbor_keys,
                cache=cache,
                lam=lam,
                proposal_mode=proposal_mode,
                rho_guided=rho_guided,
            )
            proposal_key, log_q_forward = _choose_from_log_probs(forward_log_probs, rng)
            reverse_neighbors = cache.neighbors(proposal_key)
            reverse_log_probs = _mixture_proposal_log_probs(
                current_key=proposal_key,
                neighbor_keys=reverse_neighbors,
                cache=cache,
                lam=lam,
                proposal_mode=proposal_mode,
                rho_guided=rho_guided,
            )
            direct_reverse = reverse_log_probs.get(current_key)
            log_q_reverse = _proposal_log_lookup(reverse_log_probs, current_key, cache)
            proposed += 1
            if direct_reverse is None and log_q_reverse is not None:
                reverse_topology_fallbacks += 1
            if log_q_reverse is None:
                next_keys.append(current_key)
                continue

            log_alpha = (
                cache.log_target(proposal_key)
                - cache.log_target(current_key)
                + float(log_q_reverse)
                - float(log_q_forward)
            )
            if math.log(max(rng.random(), 1e-300)) < min(0.0, log_alpha):
                next_keys.append(proposal_key)
                accepted += 1
            else:
                next_keys.append(current_key)

        current_keys = next_keys
        if iteration % int(sample_every) == 0:
            cumulative.add_keys(current_keys)
        if iteration in checkpoint_set:
            save(iteration)
        if iteration % 25 == 0 or iteration == int(iterations):
            print(
                json.dumps(
                    {
                        "dataset": dataset_id,
                        "stage": "cumulative_mh",
                        "iteration": iteration,
                        "proposal_mode": proposal_mode,
                        "rho_guided": rho_guided,
                        "lambda_split": lam,
                        "proposal_prob": proposal_prob,
                        "accepted": accepted,
                        "proposed": proposed,
                        "accept_rate": float(accepted) / float(proposed)
                        if proposed
                        else 0.0,
                        "state_cache": len(cache.key_to_tree),
                        "topology_cache": len(cache.topology_to_tree),
                    }
                ),
                flush=True,
            )

    return rows


def _load_baseline_rows(dataset_id: str) -> List[dict]:
    dataset_dir = CHECKPOINT_MB_DIR / dataset_id.lower()
    split_diverse_curve = (
        SPLIT_DIVERSE_DIR / dataset_id.lower() / "splitkl_diverse_g100000_curve.json"
    )
    initial_by_slug = _load_initial_metric_rows(dataset_id)
    candidates = [
        ("raw random MrBayes", "random", dataset_dir / "random_g100000_curve.json"),
        (
            "PhylaFlow split-KL start MrBayes",
            "splitkl_terminal",
            next(iter(sorted(dataset_dir.glob("*splitkl*_g100000_curve.json"))), None),
        ),
        ("split-diverse start MrBayes", "splitkl_diverse", split_diverse_curve),
    ]
    rows: List[dict] = []
    for label, initial_slug, path in candidates:
        if path is None or not Path(path).exists():
            continue
        data = json.loads(Path(path).read_text())
        final = data.get("final_cumulative_by_generation", {})
        initial = data.get("initial_starts", {})
        initial_override = initial_by_slug.get(initial_slug, {})
        best = data.get("best_cumulative", {})
        rows.append(
            {
                "dataset": dataset_id,
                "baseline": label,
                "path": str(path),
                "initial_tree_kl": initial_override.get(
                    "kl_divergence_tree_topology",
                    initial.get("kl_divergence_tree_topology"),
                ),
                "initial_split_kl": initial_override.get("split_kl", initial.get("split_kl")),
                "final_tree_kl": final.get("kl_divergence_tree_topology"),
                "best_tree_kl": best.get("kl_divergence_tree_topology"),
                "final_generation": data.get("ngen"),
                "samplefreq": data.get("samplefreq"),
                "num_runs": data.get("num_runs"),
            }
        )
    return rows


def _load_initial_metric_rows(dataset_id: str) -> Dict[str, dict]:
    path = SPLIT_DIVERSE_DIR / dataset_id.lower() / "initial_metrics.tsv"
    if not path.exists():
        return {}
    lines = [line.rstrip("\n").split("\t") for line in path.read_text().splitlines() if line]
    if not lines:
        return {}
    header = lines[0]
    rows: Dict[str, dict] = {}
    for parts in lines[1:]:
        row = dict(zip(header, parts))
        slug = row.get("slug")
        if not slug:
            continue
        parsed: dict = dict(row)
        for key in [
            "kl_divergence_tree_topology",
            "split_kl",
            "posterior_topology_support_recall",
            "support_rate_samples",
            "n_unique_sampled_topologies",
            "posterior_split_support_recall",
        ]:
            if key in parsed and parsed[key] != "":
                parsed[key] = float(parsed[key])
        rows[str(slug)] = parsed
    return rows


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return str(value)


def _write_outputs(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cumulative_mh_metrics.json").write_text(
        json.dumps(list(rows), indent=2) + "\n"
    )
    (output_dir / "baseline_summary.json").write_text(
        json.dumps(list(baseline_rows), indent=2) + "\n"
    )

    fields = [
        "dataset",
        "method",
        "proposal_mode",
        "lambda_split",
        "rho_guided",
        "proposal_prob",
        "iteration",
        "mh_samples_per_chain",
        "sample_count",
        "kl_divergence_tree_topology",
        "split_kl",
        "support_rate_samples",
        "posterior_topology_support_recall",
        "n_unique_sampled_topologies",
        "sampled_topology_mode_mass",
        "mean_current_log_likelihood",
        "accept_rate",
        "accepted",
        "proposed",
        "proposal_attempt_rate",
    ]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(_format_value(row.get(field, "")) for field in fields))
    (output_dir / "cumulative_mh_metrics.tsv").write_text("\n".join(lines) + "\n")

    baseline_fields = [
        "dataset",
        "baseline",
        "initial_tree_kl",
        "initial_split_kl",
        "final_tree_kl",
        "best_tree_kl",
        "final_generation",
        "samplefreq",
        "num_runs",
        "path",
    ]
    baseline_lines = ["\t".join(baseline_fields)]
    for row in baseline_rows:
        baseline_lines.append(
            "\t".join(_format_value(row.get(field, "")) for field in baseline_fields)
        )
    (output_dir / "baseline_summary.tsv").write_text(
        "\n".join(baseline_lines) + "\n"
    )

    summary_lines = [
        "# DS4/DS7/DS8 branch-warm cumulative MH probe",
        "",
        "The MH rows are cumulative samples from the chain states, not just final/current states.",
        "",
        "## Existing 100K MrBayes baselines",
        "",
        "| dataset | baseline | initial TreeKL | initial split-KL | final TreeKL |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in baseline_rows:
        summary_lines.append(
            "| {dataset} | {baseline} | {initial:.6f} | {split:.6f} | {final:.6f} |".format(
                dataset=row["dataset"],
                baseline=row["baseline"],
                initial=float(row.get("initial_tree_kl") or float("nan")),
                split=float(row.get("initial_split_kl") or float("nan")),
                final=float(row.get("final_tree_kl") or float("nan")),
            )
        )
    summary_lines.extend(
        [
            "",
            "## New cumulative MH arms",
            "",
            "| dataset | method | iter | samples | TreeKL | split-KL | support | recall | unique | accept | proposed |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        summary_lines.append(
            "| {dataset} | {method} | {iteration} | {samples:.0f} | {tree:.6f} | "
            "{split:.6f} | {support:.6f} | {recall:.6f} | {unique:.0f} | "
            "{accept:.6f} | {proposed} |".format(
                dataset=row["dataset"],
                method=row["method"],
                iteration=int(row["iteration"]),
                samples=float(row["sample_count"]),
                tree=float(row["kl_divergence_tree_topology"]),
                split=float(row["split_kl"]),
                support=float(row["support_rate_samples"]),
                recall=float(row["posterior_topology_support_recall"]),
                unique=float(row["n_unique_sampled_topologies"]),
                accept=float(row.get("accept_rate", 0.0)),
                proposed=int(row.get("proposed", 0)),
            )
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["DS4", "DS7", "DS8"])
    parser.add_argument(
        "--output-dir",
        default=str(ANALYSIS_DIR / "ds4_ds7_ds8_branchwarm_cumulative_mh_20260427"),
    )
    parser.add_argument("--posterior-samples-per-rep", type=int, default=1000)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--warmup-ngen", type=int, default=1000)
    parser.add_argument("--warmup-samplefreq", type=int, default=500)
    parser.add_argument("--warmup-printfreq", type=int, default=500)
    parser.add_argument("--warmup-work-root", default="/tmp/bwcmh_20260427")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--mrbayes-bin",
        default="/opt/conda/envs/phylaflow-mrbayes/bin/mb",
    )
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[0, 50, 100, 250])
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--floor", type=float, default=1e-6)
    parser.add_argument("--lambda-split", type=float, default=0.25)
    parser.add_argument("--rho-guided", type=float, default=0.05)
    parser.add_argument("--proposal-prob", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--force-warmup", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []
    all_baselines: List[dict] = []

    for dataset_index, dataset_id in enumerate(args.datasets):
        dataset_id = str(dataset_id).upper()
        if dataset_id not in SPLITKL_GUIDE_LISTS:
            raise ValueError(f"No split-KL guide list configured for {dataset_id}")
        print(json.dumps({"dataset": dataset_id, "stage": "load_inputs"}), flush=True)
        start_trees = _read_tree_list(_split_diverse_start_list(dataset_id))
        guide_trees = _read_tree_list(_guide_tree_list(dataset_id))
        if int(args.max_runs) > 0:
            start_trees = start_trees[: int(args.max_runs)]
            guide_trees = guide_trees[: max(int(args.max_runs), min(len(guide_trees), 200))]

        dataset_output_dir = output_dir / dataset_id.lower()
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        (dataset_output_dir / "input_start_trees.txt").write_text(
            "\n".join(start_trees) + "\n"
        )
        (dataset_output_dir / "input_guide_trees.txt").write_text(
            "\n".join(guide_trees) + "\n"
        )

        print(json.dumps({"dataset": dataset_id, "stage": "load_posterior"}), flush=True)
        _dataset_pickle, golden_root = _dataset_paths(dataset_id)
        posterior_trees = _load_posterior_trees(
            golden_root=golden_root,
            dataset_id=dataset_id,
            per_file_sample_count=int(args.posterior_samples_per_rep),
        )
        metric_cache: Dict[str, str] = {}
        posterior_counts = Counter()
        for tree in posterior_trees:
            key = metric_cache.get(tree)
            if key is None:
                key = canonicalize_topology_newick(tree)
                metric_cache[tree] = key
            posterior_counts[key] += 1
        _posterior_universe, posterior_split_counts = _split_counts(posterior_trees)

        print(json.dumps({"dataset": dataset_id, "stage": "branch_warmup_start"}), flush=True)
        warmed_trees = _run_branch_warmups(
            dataset_id=dataset_id,
            start_trees=start_trees,
            work_dir=Path(args.warmup_work_root) / dataset_id.lower(),
            ngen=int(args.warmup_ngen),
            samplefreq=int(args.warmup_samplefreq),
            printfreq=int(args.warmup_printfreq),
            max_workers=int(args.max_workers),
            mrbayes_bin=Path(args.mrbayes_bin),
            force=bool(args.force_warmup),
        )
        (dataset_output_dir / "branch_warmed_start_trees.txt").write_text(
            "\n".join(warmed_trees) + "\n"
        )

        method_specs = [
            {
                "proposal_mode": "uniform",
                "lambda_split": 0.0,
                "rho_guided": 0.0,
                "seed_offset": 0,
            },
            {
                "proposal_mode": "mixed",
                "lambda_split": float(args.lambda_split),
                "rho_guided": float(args.rho_guided),
                "seed_offset": 1000,
            },
        ]
        for method_index, spec in enumerate(method_specs):
            print(
                json.dumps(
                    {
                        "dataset": dataset_id,
                        "stage": "cumulative_mh_start",
                        "proposal_mode": spec["proposal_mode"],
                    }
                ),
                flush=True,
            )
            rows = _run_cumulative_branch_state_mh(
                start_trees=warmed_trees,
                guide_trees=guide_trees,
                dataset_id=dataset_id,
                posterior_counts=posterior_counts,
                posterior_split_counts=posterior_split_counts,
                iterations=int(args.iterations),
                checkpoints=args.checkpoints,
                sample_every=int(args.sample_every),
                lam=float(spec["lambda_split"]),
                proposal_mode=str(spec["proposal_mode"]),
                rho_guided=float(spec["rho_guided"]),
                proposal_prob=float(args.proposal_prob),
                seed=int(args.seed) + dataset_index * 10000 + int(spec["seed_offset"]),
                floor=float(args.floor),
            )
            all_rows.extend(rows)
            _write_outputs(
                output_dir=output_dir,
                rows=all_rows,
                baseline_rows=all_baselines,
            )

        all_baselines.extend(_load_baseline_rows(dataset_id))
        _write_outputs(
            output_dir=output_dir,
            rows=all_rows,
            baseline_rows=all_baselines,
        )

    print((output_dir / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
