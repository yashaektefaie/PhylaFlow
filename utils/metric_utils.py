from utils.random_tree import Tree 
from utils.bhv_utils import BHVEncoder
from collections import Counter
from typing import List 
import math
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

    

    