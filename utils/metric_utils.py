from utils.random_tree import Tree 
from utils.bhv_utils import BHVEncoder
from collections import Counter
from typing import Dict, List, Tuple
from scipy.stats import pearsonr
import math
import random
import subprocess, tempfile, re, pathlib

_LOGLH_RE = re.compile(r"Final LogLikelihood:\s*([-0-9.eE]+)")

enc = BHVEncoder()


def translate_tree_labels(newick: str, translation: Dict[str, str]) -> str:
    """Replace numeric labels in a Newick tree with actual taxon names."""
    def replace_label(match):
        label = match.group(1)
        return translation.get(label, label)
    
    # Match labels that appear before : or , or ) 
    pattern = r'(?<=[,(])([0-9]+)(?=[:,)])'
    return re.sub(pattern, replace_label, newick)


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

def raxmlng_loglh_batch(
    msa_path: str,
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
    
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        tree_file = td / "trees.nwk"
        # Write all trees to one file, one per line
        tree_file.write_text("\n".join(t.strip() for t in newicks) + "\n")

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

        # Parse all log-likelihoods from output
        loglhs = [float(m.group(1)) for m in _LOGLH_RE.finditer(out)]
        
        if len(loglhs) != len(newicks):
            raise RuntimeError(
                f"Expected {len(newicks)} loglh values, got {len(loglhs)}:\n{out}"
            )
        
        return loglhs


def raxmlng_loglh_for_tree(
    msa_path: str,
    newick: str,
    model: str = "JC",
    threads: int = 1,
) -> float:
    """
    Returns log p(Y | tree, branch_lengths, model) using RAxML-NG --loglh.
    Assumes Newick includes branch lengths. For multiple trees, use raxmlng_loglh_batch.
    """
    results = raxmlng_loglh_batch(msa_path, [newick], model=model, threads=threads)
    return results[0]


def loglh_for_tree_list(msa_path: str, trees: List[str], threads: int = 1) -> List[float]:
    """Batch version - much faster than one-at-a-time."""
    return raxmlng_loglh_batch(msa_path, trees, model="JC", threads=threads)

def inference_time_per_tree(total_time: float, num_trees: int) -> float:
    """Compute average inference time per tree."""
    if num_trees == 0:
        return 0.0
    return total_time / num_trees


def load_sample_trprobs(path: str, max_trees: int = 1000) -> Tuple[List[str], Dict[str, str]]:
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
            import pdb; pdb.set_trace()
            translated = translate_tree_labels(m.group(0), translation)
            result.append(translated)
    sampled_trees = result

    return sampled_trees, translation

# samples, translation = load_sample_trprobs("./benchmark_data/DS1/rep_1/DS1.trprobs")
# import pdb; pdb.set_trace()