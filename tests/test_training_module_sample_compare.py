import unittest
from unittest import mock


class DummyTree:
    def __init__(self, num_leaves=None, random=False):
        self.num_leaves = num_leaves
        self.random = random

    def __str__(self):
        return "(0:0.1,1:0.1,2:0.1);"


class DummyDatasetTrain:
    def return_posterior_trees(self, _id):
        return ["(1:0.1,2:0.1,3:0.1);"] * 3


class DummyDataset:
    def __init__(self):
        self.dataset_train = DummyDatasetTrain()
        self.dataset_val = DummyDatasetTrain()

    def return_number_leaves(self, _id):
        return 3


class TestTrainingModuleSampleCompare(unittest.TestCase):
    def test_sample_compare_runs(self):
        try:
            from run import TrainingModule as tm
        except Exception as exc:
            self.skipTest(f"TrainingModule import failed: {exc}")

        mapping = {0: "A", 1: "B", 2: "C"}
        batch = {
            "nexus_filepaths": ["dummy.nex"],
            "tree_paths": ["dummy.trprobs"],
            "ids": ["ds1"],
            "mappings": [mapping],
            "phyla_embeddings": None,
        }

        module = tm.TrainingModule(
            model=object(),
            dataset=DummyDataset(),
            lr=1e-4,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            verbose=False,
        )

        def fake_sample(*_args, **_kwargs):
            return ["(0:0.1,1:0.1,2:0.1);"]

        with mock.patch.object(tm, "Tree", DummyTree), \
            mock.patch.object(tm.TrainingModule, "sample", fake_sample), \
            mock.patch.object(tm, "compare_likelihood_distributions", lambda *a, **k: {"avg_true_loglh": -1.0}), \
            mock.patch.object(tm, "kl_divergence_topological_distributions", lambda *a, **k: 0.0), \
            mock.patch.object(tm, "split_bipartition_frequency_correlation", lambda *a, **k: 1.0), \
            mock.patch.object(tm, "compare_branch_length_distributions", lambda *a, **k: {"js_divergence_branch_length": 0.0}):
            metrics = module.sample_compare(batch, train=True, num_samples=2)

        self.assertIn("avg_true_loglh", metrics)
        self.assertIn("kl_divergence_topology", metrics)
        self.assertIn("split_bipartition_corr", metrics)
        self.assertIn("js_divergence_branch_length", metrics)


if __name__ == "__main__":
    unittest.main()
