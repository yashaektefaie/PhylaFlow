import pickle
import numpy as np
from ete3 import Tree
from utils.metric_utils import kl_divergence_topological_distributions

sampled, posterior = pickle.load(open("samples/sample_trees_100.pkl", "rb"))
sampled_1000, posterior_1000 = pickle.load(open("samples/sample_trees_8000.pkl", "rb"))

kl_div = kl_divergence_topological_distributions(sampled, posterior, num_leaves = 155)
kl_div_1000 = kl_divergence_topological_distributions(sampled_1000, posterior_1000, num_leaves = 155)

def compute_avg_normalized_rf(trees1, trees2):
    """
    Compute average normalized Robinson-Foulds distance between two sets of trees.
    For each tree in trees1, compute RF distance to each tree in trees2, then average.
    """
    rf_distances = []
    n_pairs = min(len(trees1), len(trees2))  # Compare pairwise
    
    for i in range(n_pairs):
        try:
            t1 = Tree(trees1[i])
            t2 = Tree(trees2[i])
            # robinson_foulds returns (rf, max_rf, common_leaves, parts_t1, parts_t2, discarded, edges_t1, edges_t2)
            rf_result = t1.robinson_foulds(t2, unrooted_trees=True)
            rf = rf_result[0]
            max_rf = rf_result[1]
            if max_rf > 0:
                norm_rf = rf / max_rf
            else:
                norm_rf = 0.0
            rf_distances.append(norm_rf)
        except Exception as e:
            print(f"Error computing RF for pair {i}: {e}")
            continue
    
    return np.mean(rf_distances), np.std(rf_distances), len(rf_distances)

# Calculate average normalized RF between sampled and posterior trees
avg_rf, std_rf, n_pairs = compute_avg_normalized_rf(sampled, posterior)
print(f"Average norm-RF (sampled vs posterior, n=100): {avg_rf:.4f} ± {std_rf:.4f} (n_pairs={n_pairs})")

# Calculate average normalized RF between sampled_1000 and posterior_1000 trees
avg_rf_1000, std_rf_1000, n_pairs_1000 = compute_avg_normalized_rf(sampled_1000, posterior_1000)
print(f"Average norm-RF (sampled_1000 vs posterior_1000, n=8000): {avg_rf_1000:.4f} ± {std_rf_1000:.4f} (n_pairs={n_pairs_1000})")

print(f"\nComparison:")
print(f"  Difference: {avg_rf - avg_rf_1000:.4f}")
print(f"  Relative improvement: {((avg_rf - avg_rf_1000) / avg_rf * 100):.2f}%" if avg_rf > 0 else "  N/A")

import pdb; pdb.set_trace()
