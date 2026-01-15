import unittest

from utils.metric_utils import (
    kl_divergence_topological_distributions,
    split_bipartition_frequency_correlation,
)
from utils.random_tree import Tree


class TestMetricUtils(unittest.TestCase):
    def setUp(self):
        self.tree_a = str(Tree(num_leaves=50, random=True))
        self.tree_b = str(Tree(num_leaves=50, random=True))
        self.tree_c = str(Tree(num_leaves=50, random=True))    
        self.num_leaves = 50

    def test_kl_divergence_matches_identity(self):
        posterior = [self.tree_a, self.tree_a, self.tree_b]
        sampled_same = list(posterior)
        sampled_diff = [self.tree_c, self.tree_c, self.tree_c]

        kl_same = kl_divergence_topological_distributions(
            posterior, sampled_same, num_leaves=self.num_leaves
        )
        kl_diff = kl_divergence_topological_distributions(
            posterior, sampled_diff, num_leaves=self.num_leaves
        )

        self.assertLess(kl_same, 1e-8)
        self.assertGreater(kl_diff, 1e-4)

    def test_bipartition_frequency_correlation(self):
        posterior = [str(Tree(num_leaves=self.num_leaves, random=True)) for _ in range(5)]
        sampled_same = list(posterior)
        sampled_diff = [str(Tree(num_leaves=self.num_leaves, random=True)) for _ in range(5)]

        corr_same = split_bipartition_frequency_correlation(
            posterior, sampled_same, num_leaves=self.num_leaves
        )
        corr_diff = split_bipartition_frequency_correlation(
            posterior, sampled_diff, num_leaves=self.num_leaves
        )

        self.assertAlmostEqual(corr_same, 1.0, places=6)
        self.assertLess(corr_diff, 0.5)

if __name__ == "__main__":
    unittest.main()
