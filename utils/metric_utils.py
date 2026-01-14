from utils.random_tree import Tree 
from utils.bhv_utils import BHVEncoder
from collections import Counter
from typing import List 
from scipy.stats import pearsonr
import math
import subprocess, tempfile, re, pathlib

_LOGLH_RE = re.compile(r"Final LogLikelihood:\s*([-0-9.eE]+)")

enc = BHVEncoder()


def kl_divergence_topological_distributions(posterior_trees: List[str], 
                                            sampled_trees: List[str], 
                                            num_leaves: int, 
                                            eps: float = 1e-8,
                                            alpha: float = 1e-6) -> float:
    """Compute KL divergence between topological distributions of two sets of trees."""
    
    full_mask = (1 << num_leaves) - 1

    def generate_splits(newick):
        t1 = Tree(newick)
        edge_mask, edge_lengths = enc.return_BHV_encoding(t1)
        splits = []
        for m, l in zip(edge_mask, edge_lengths):
            if l <= eps:
                continue 

            k = int(m).bit_count()
            if k <= 1 or k >= num_leaves - 1:
                continue

            if k > num_leaves / 2:
                m = full_mask ^ int(m)  # Take complement

            splits.append(int(m))
            
        return tuple(sorted(splits))    
    
    split_counts_ground_truth = Counter([generate_splits(t) for t in posterior_trees])
    gt_topological_distribution = {k: v / sum(split_counts_ground_truth.values()) for k, v in split_counts_ground_truth.items()}
    split_counts_sampled = Counter([generate_splits(t) for t in sampled_trees])
    sampled_topological_distribution = {k: v / sum(split_counts_sampled.values()) for k, v in split_counts_sampled.items()}

    support = set(gt_topological_distribution.keys()).union(set(sampled_topological_distribution.keys()))
    ZP = 1.0 + alpha * len(support)
    ZQ = 1.0 + alpha * len(support)

    kl = 0.0
    for k in support:
        p = (gt_topological_distribution.get(k, 0.0) + alpha) / ZP
        q = (sampled_topological_distribution.get(k, 0.0) + alpha) / ZQ
        kl += p * (math.log(p / q) / math.log(math.e))
    return kl

def split_bipartition_frequency_correlation(posterior_trees: List[str], 
                                           sampled_trees: List[str], 
                                           num_leaves: int, 
                                           eps: float = 1e-8) -> float:
    """Compute correlation between split bipartition frequencies of two sets of trees."""
    
    full_mask = (1 << num_leaves) - 1

    def generate_splits(newick):
        t1 = Tree(newick)
        edge_mask, edge_lengths = enc.return_BHV_encoding(t1)
        splits = []
        for m, l in zip(edge_mask, edge_lengths):
            if l <= eps:
                continue 

            k = int(m).bit_count()
            if k <= 1 or k >= num_leaves - 1:
                continue

            if k > num_leaves / 2:
                m = full_mask ^ int(m)  # Take complement

            splits.append(int(m))
            
        return splits    
    
    split_counts_ground_truth = Counter()
    for t in posterior_trees:
        splits = generate_splits(t)
        split_counts_ground_truth.update(splits)
        
    split_counts_sampled = Counter()
    for t in sampled_trees:
        splits = generate_splits(t)
        split_counts_sampled.update(splits)

    all_splits = set(split_counts_ground_truth.keys()).union(set(split_counts_sampled.keys()))
    gt_freqs = []
    sampled_freqs = []
    for s in all_splits:
        gt_freqs.append(split_counts_ground_truth.get(s, 0) / len(posterior_trees))
        sampled_freqs.append(split_counts_sampled.get(s, 0) / len(sampled_trees))

    correlation, _ = pearsonr(gt_freqs, sampled_freqs)
    return correlation

def average_likelihood_plausibility(posterior_trees: List[str], sampled_trees: List[str]) -> float:
    """Compute average likelihood plausibility of sampled trees under posterior trees."""
    from utils.bhv_utils import compute_likelihood_plausibility
    total_plausibility = 0.0
    for sampled_tree in sampled_trees:
        plausibility = compute_likelihood_plausibility(sampled_tree, posterior_trees)
        total_plausibility += plausibility
    average_plausibility = total_plausibility / len(sampled_trees)
    return average_plausibility

def raxmlng_loglh_for_tree(
    msa_path: str,
    newick: str,
    model: str = "JC",
    threads: int = 1,
) -> float:
    """
    Returns log p(Y | tree, branch_lengths, model) using RAxML-NG --loglh.
    Assumes Newick includes branch lengths.
    """
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        tree_file = td / "tree.nwk"
        tree_file.write_text(newick.strip() + "\n")

        cmd = [
            "raxml-ng",
            "--loglh",
            "--msa", msa_path,
            "--tree", str(tree_file),
            "--model", model,
            "--threads", str(threads),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        if p.returncode != 0:
            raise RuntimeError(f"RAxML-NG failed:\n{out}")

        m = _LOGLH_RE.search(out)
        if not m:
            raise RuntimeError(f"Could not parse loglh from RAxML-NG output:\n{out}")

        return float(m.group(1))

def loglh_for_tree_list(msa_path: str, trees: List[str], threads: int = 1) -> List[float]:
    return [raxmlng_loglh_for_tree(msa_path, t, model="JC", threads=threads) for t in trees]

def inference_time_per_tree(total_time: float, num_trees: int) -> float:
    """Compute average inference time per tree."""
    if num_trees == 0:
        return 0.0
    return total_time / num_trees


    