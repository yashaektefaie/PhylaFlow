import unittest
from unittest import mock
from data.dataset import PhylaDataModule
import yaml
from utils.utils import get_possible_ids
import random
from model.model import return_model
from run.TrainingModule import TrainingModule

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        with open("configs/train.yaml", "r") as f:
            config = yaml.safe_load(f)

        ids = get_possible_ids(config['data']['nexus_root'])
        #Random 80-20 train-test split for now
        ran = random.Random(42)
        ran.shuffle(ids)
        train_ids = ids[:int(0.8*len(ids))]
        test_ids = ids[int(0.8*len(ids)):]

        ###TEMPORARY FOR DEBUGGING
        train_ids = test_ids

        dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)
        one = dataset.dataset_train[0]
        two = dataset.dataset_train[0]
        self.batch = dataset.collate_fn([one, two])

        phyla_flow = return_model(config)

        self.training_module = TrainingModule(
            model=phyla_flow,
            lr=config['trainer']['lr'],
            record=config['trainer']['record'],
            epochs=config['trainer']['epochs'],
            dataset=dataset,
            lr_scheduler = 'default',
            num_annealing_steps = 10000,
            num_warmup_steps = 1000,
            deepspeed = False,
            logger = None
        )

    def test_sample_compare_runs(self):
        metrics = self.training_module.sample_compare(self.batch)
        self.assertIn("avg_true_loglh", metrics)
        self.assertIn("kl_divergence_topology", metrics)
        self.assertIn("split_bipartition_corr", metrics)
        self.assertIn("js_divergence_branch_length", metrics)
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    unittest.main()
