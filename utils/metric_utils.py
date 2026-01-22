from utils.random_tree import Tree
from utils.bhv_utils import BHVEncoder
from collections import Counter
from typing import Dict, List, Tuple
from scipy.stats import pearsonr
import numpy as np
import math
import random
import subprocess, tempfile, re, pathlib
from Bio import AlignIO
from utils.utils import (
	jensenshannon_loglh_divergence,
	kl_loglh_divergence,
	return_total_tree_length,
)
from ete3 import Tree as EteTree

_LOGLH_RE = re.compile(r"final logLikelihood:\s*([-0-9.eE]+)")

enc = BHVEncoder()


def calculate_norm_rf(t1_nw: str, t2_nw: str) -> float:
	try:
		t1 = EteTree(t1_nw)
		t2 = EteTree(t2_nw)
		rf, max_rf, _, _, _, _, _ = t1.robinson_foulds(t2, unrooted_trees=True)
		return rf / max_rf if max_rf > 0 else 0.0
	except Exception as e:
		return 0.0


def kl_divergence_topological_distributions(
	posterior_trees: List[str],
	sampled_trees: List[str],
	num_leaves: int,
	eps: float = 1e-8,
	alpha: float = 1e-6,
) -> Dict[str, float]:
	"""Compute KL divergence between topological distributions of two sets of trees."""

	full_mask = (1 << num_leaves) - 1

	def return_splits(nw):
		t = Tree(nw)
		enc = BHVEncoder()
		masks, lens = enc.return_BHV_encoding(t)
		return masks

	split_counts_ground_truth = Counter()
	for t in posterior_trees:
		splits = return_splits(t)
		split_counts_ground_truth.update(splits)

	gt_topological_distribution = {
		k: v / sum(split_counts_ground_truth.values())
		for k, v in split_counts_ground_truth.items()
	}

	split_counts_sampled = Counter()
	for t in sampled_trees:
		splits = return_splits(t)
		split_counts_sampled.update(splits)

	sampled_topological_distribution = {
		k: v / sum(split_counts_sampled.values())
		for k, v in split_counts_sampled.items()
	}

	support = set(gt_topological_distribution.keys()).union(
		set(sampled_topological_distribution.keys())
	)
	ZP = 1.0 + alpha * len(support)
	ZQ = 1.0 + alpha * len(support)

	kl = 0.0
	for k in support:
		p = (gt_topological_distribution.get(k, 0.0) + alpha) / ZP
		q = (sampled_topological_distribution.get(k, 0.0) + alpha) / ZQ
		kl += p * (math.log(p / q) / math.log(math.e))
	return {"kl_divergence_topological": kl}


def split_bipartition_frequency_correlation(
	posterior_trees: List[str],
	sampled_trees: List[str],
	num_leaves: int,
	eps: float = 1e-8,
) -> Dict[str, float]:
	"""Compute correlation between split bipartition frequencies of two sets of trees."""

	full_mask = (1 << num_leaves) - 1

	def generate_splits(nw):
		t = Tree(nw)
		enc = BHVEncoder()
		masks, lens = enc.return_BHV_encoding(t)
		return masks

	split_counts_ground_truth = Counter()
	for t in posterior_trees:
		splits = generate_splits(t)
		split_counts_ground_truth.update(splits)

	split_counts_sampled = Counter()
	for t in sampled_trees:
		splits = generate_splits(t)
		split_counts_sampled.update(splits)

	all_splits = set(split_counts_ground_truth.keys()).union(
		set(split_counts_sampled.keys())
	)
	gt_freqs = []
	sampled_freqs = []
	for s in all_splits:
		gt_freqs.append(split_counts_ground_truth.get(s, 0) / len(posterior_trees))
		sampled_freqs.append(split_counts_sampled.get(s, 0) / len(sampled_trees))

	correlation, _ = pearsonr(gt_freqs, sampled_freqs)
	return {"bipartition_frequency_correlation": correlation}


def raxmlng_loglh_batch(
	nexus_path: str,
	newicks: List[str],
	model: str = "JC",
	threads: int = 1,
) -> List[float]:
	"""
	Returns log p(Y | tree, branch_lengths, model) for multiple trees using RAxML-NG --loglh.
	Assumes Newick includes branch lengths.
	"""
	if not newicks:
		return []

	with tempfile.TemporaryDirectory() as td_trees, tempfile.TemporaryDirectory() as td_msa:
		td_trees = pathlib.Path(td_trees)
		td_msa = pathlib.Path(td_msa)

		# ---- Write trees ----
		tree_file = td_trees / "trees.nwk"
		tree_file.write_text("\n".join(t.strip() for t in newicks) + "\n")

		# ---- Convert NEXUS -> FASTA ----
		msa_file = td_msa / "msa.fasta"

		# Option A (recommended): AlignIO.convert
		AlignIO.convert(
			nexus_path,
			"nexus",
			msa_file,
			"fasta",
		)

		cmd = [
			"raxml-ng",
			"--loglh",
			"--msa",
			str(msa_file),
			"--tree",
			str(tree_file),
			"--model",
			model,
			"--threads",
			str(threads),
		]

		p = subprocess.run(cmd, capture_output=True, text=True)
		out = (p.stdout or "") + "\n" + (p.stderr or "")
		if p.returncode != 0:
			raise RuntimeError(f"RAxML-NG failed:\n{out}")

		# Parse all log-likelihoods from output
		loglhs = [float(m.group(1)) for m in _LOGLH_RE.finditer(out)]

		if len(loglhs) != len(newicks):
			raise RuntimeError(
				f"Expected {len(newicks)} loglh values, got {len(loglhs)}:\n{out}"
			)

		return loglhs


def compare_likelihood_distributions(
	nexus_file_path: str,
	true_trees: List[str],
	sampled_trees: List[str],
	threads: int = 1,
) -> Dict[str, float]:
	"""Compare likelihood distributions of true and sampled trees using RAxML-NG."""
	true_loglhs = raxmlng_loglh_batch(
		nexus_path=nexus_file_path, newicks=true_trees, model="JC", threads=threads
	)

	sampled_loglhs = raxmlng_loglh_batch(
		nexus_path=nexus_file_path, newicks=sampled_trees, model="JC", threads=threads
	)

	avg_true_loglh = (
		sum(true_loglhs) / len(true_loglhs) if true_loglhs else float("-inf")
	)
	avg_sampled_loglh = (
		sum(sampled_loglhs) / len(sampled_loglhs) if sampled_loglhs else float("-inf")
	)

	# Difference in average log-likelihoods
	diff_avg_loglh = avg_true_loglh - avg_sampled_loglh
	js_div = jensenshannon_loglh_divergence(true_loglhs, sampled_loglhs, bins=50)
	kl_div = kl_loglh_divergence(true_loglhs, sampled_loglhs, bins=50)

	return {
		"avg_true_loglh": avg_true_loglh,
		"avg_sampled_loglh": avg_sampled_loglh,
		"diff_avg_loglh": diff_avg_loglh,
		"js_divergence_loglh": js_div,
		"kl_divergence_loglh": kl_div,
	}


def compare_branch_length_distributions(
	true_trees: List[str], sampled_trees: List[str]
) -> Dict[str, float]:
	"""Compare branch length distributions of true and sampled trees."""
	true_branch_lengths = []
	for newick in true_trees:
		true_branch_lengths.append(return_total_tree_length(newick))

	sampled_branch_lengths = []
	for newick in sampled_trees:
		sampled_branch_lengths.append(return_total_tree_length(newick))

	js_div = jensenshannon_loglh_divergence(
		true_branch_lengths, sampled_branch_lengths, bins=50
	)
	kl_div = kl_loglh_divergence(true_branch_lengths, sampled_branch_lengths, bins=50)

	return {
		"js_divergence_branch_length": js_div,
		"kl_divergence_branch_length": kl_div,
	}


def load_sample_trprobs(
	path: str, max_trees: int = 1000
) -> Tuple[List[str], Dict[str, str]]:
	"""Load sampled Newick trees and the translation map from a .tprobs file."""
	trees: List[str] = []
	weights: List[float] = []
	translation: Dict[str, str] = {}
	tree_buf: List[str] = []
	in_tree = False
	in_translate = False

	def parse_weight(line: str) -> float | None:
		m = re.search(r"p\s*=\s*([0-9.eE+-]+)", line)
		if m:
			return float(m.group(1))
		m = re.search(r"&W\s*([0-9.eE+-]+)", line)
		if m:
			return float(m.group(1))
		return None

	def finalize_tree(raw: str, weight: float | None) -> None:
		raw = raw.strip()
		while raw.startswith("["):
			end = raw.find("]")
			if end == -1:
				break
			raw = raw[end + 1 :].lstrip()
		raw = raw.rstrip(";").strip()
		if not raw:
			return
		trees.append(raw)
		weights.append(weight if weight is not None else 0.0)

	current_weight = None

	with open(path, "r") as handle:
		for raw_line in handle:
			line = raw_line.strip()
			lower = line.lower()

			if lower.startswith("translate"):
				in_translate = True
				continue
			if in_translate:
				if line:
					match = re.match(r"^(\d+)\s+([^,;]+)", line.rstrip(",;"))
					if match:
						translation[match.group(1)] = match.group(2)
				if ";" in line:
					in_translate = False
				continue

			if lower.startswith("tree "):
				in_tree = True
				current_weight = parse_weight(line)
				after_eq = line.split("=", 1)[-1]
				tree_buf.append(after_eq)
				if ";" in line:
					in_tree = False
			elif in_tree:
				tree_buf.append(line)
				if ";" in line:
					in_tree = False

			if not in_tree and tree_buf:
				tree_str = " ".join(tree_buf)
				tree_buf = []
				finalize_tree(tree_str, current_weight)
				current_weight = None

	if not trees:
		return [], translation

	total = sum(weights)
	if total <= 0.0:
		weights = [1.0 / len(trees)] * len(trees)
	else:
		weights = [w / total for w in weights]

	if max_trees <= 0:
		return [], translation

	sampled_trees = random.choices(trees, weights=weights, k=max_trees)
	result = []
	for t in sampled_trees:
		m = re.search(r"\(.*\)", t.strip())
		if m:
			# Apply translation mapping to convert numeric labels to taxon names
			import pdb

			pdb.set_trace()
			# BELOW will error but not changing cause we may be deleting this anyways
			translated = translate_tree_labels(m.group(0), translation)
			result.append(translated)
	sampled_trees = result

	return sampled_trees, translation


# samples, translation = load_sample_trprobs("./benchmark_data/DS1/rep_1/DS1.trprobs")
# import pdb; pdb.set_trace()
