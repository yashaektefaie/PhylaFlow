import random
import sys
import types
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
import copy

import torch
from ete3 import Tree as EteTree


def _install_phyla_stub():
    phyla_mod = types.ModuleType("phyla")
    phyla_utils_mod = types.ModuleType("phyla.utils")
    phyla_utils_utils_mod = types.ModuleType("phyla.utils.utils")
    phyla_eval_mod = types.ModuleType("phyla.eval")
    phyla_eval_evo_mod = types.ModuleType("phyla.eval.evo_reasoning_eval")

    class Config:
        def __init__(self):
            self.trainer = types.SimpleNamespace(checkpoint_path="")
            self.eval = types.SimpleNamespace(device="cpu")

    def load_config(config_cls):
        return config_cls()

    def load_model(config=None, random_model=False):
        return {"model": MagicMock()}

    def _encode_sequences_openfold_style(sequences, names):
        batch = {
            "encoded_sequences": torch.zeros((len(sequences), 1), dtype=torch.long),
            "sequence_mask": torch.ones((len(sequences), 1), dtype=torch.long),
            "cls_positions": torch.ones((len(sequences), 1), dtype=torch.bool),
        }
        return batch, None

    phyla_utils_utils_mod.load_config = load_config
    phyla_eval_evo_mod.Config = Config
    phyla_eval_evo_mod.load_model = load_model
    phyla_eval_evo_mod._encode_sequences_openfold_style = (
        _encode_sequences_openfold_style
    )

    phyla_mod.utils = phyla_utils_mod
    phyla_mod.eval = phyla_eval_mod
    phyla_utils_mod.utils = phyla_utils_utils_mod
    phyla_eval_mod.evo_reasoning_eval = phyla_eval_evo_mod

    sys.modules["phyla"] = phyla_mod
    sys.modules["phyla.utils"] = phyla_utils_mod
    sys.modules["phyla.utils.utils"] = phyla_utils_utils_mod
    sys.modules["phyla.eval"] = phyla_eval_mod
    sys.modules["phyla.eval.evo_reasoning_eval"] = phyla_eval_evo_mod


def _install_deepspeed_stub():
    if "deepspeed" in sys.modules:
        return

    ds_mod = types.ModuleType("deepspeed")
    ds_ops_mod = types.ModuleType("deepspeed.ops")
    ds_adam_mod = types.ModuleType("deepspeed.ops.adam")

    class FusedAdam(torch.optim.Adam):
        pass

    ds_adam_mod.FusedAdam = FusedAdam
    ds_ops_mod.adam = ds_adam_mod
    ds_mod.ops = ds_ops_mod

    sys.modules["deepspeed"] = ds_mod
    sys.modules["deepspeed.ops"] = ds_ops_mod
    sys.modules["deepspeed.ops.adam"] = ds_adam_mod


try:
    from deepspeed.ops.adam import FusedAdam as _FusedAdamCheck  # noqa: F401
except Exception:
    _install_deepspeed_stub()

try:
    from phyla.utils.utils import load_config as _LoadConfigCheck  # noqa: F401
    from phyla.eval.evo_reasoning_eval import Config as _ConfigCheck  # noqa: F401
except Exception:
    _install_phyla_stub()

from model.model import TreeDenoiserTokenGT
from data.dataset import TreeDataset
from run.TrainingModule import TrainingModule
from utils.bhv_utils import (
    return_sampled_tree_boundary_decisions,
    return_sampled_tree_orthant_velocity,
)
from utils.random_tree import Tree
from utils.utils import remove_bit


def _detach_tokenized_batch(tokenized):
    out = []
    for item in tokenized:
        if torch.is_tensor(item):
            out.append(item.detach())
        else:
            out.append(item)
    return tuple(out)


def _make_single_velocity_batch(tokenizer, n_leaves, seed):
    random.seed(seed)
    torch.manual_seed(seed)

    start_tree = str(Tree(num_leaves=n_leaves, random=True))
    target_tree = str(Tree(num_leaves=n_leaves, random=True))
    sampled_newick, velocity = return_sampled_tree_orthant_velocity(
        start_tree, target_tree, 0.0
    )
    with torch.no_grad():
        tokenized = _detach_tokenized_batch(tokenizer([sampled_newick]))

    return {
        "tokenized_trees": tokenized,
        "batched_time": torch.tensor([0.0], dtype=torch.float32),
        "phyla_embeddings": None,
        "original_trees": [sampled_newick],
        "batched_velocity": [velocity],
        "num_leaves": [Tree(sampled_newick).n_leaves],
    }


def _leaf_sort_key(name):
    try:
        return (0, int(name))
    except ValueError:
        return (1, str(name))


def _prune_and_renumber_tree_pair(start_newick, target_newick, keep_leaves=12):
    t_start = EteTree(start_newick, format=1)
    t_target = EteTree(target_newick, format=1)

    start_names = {leaf.name for leaf in t_start.get_leaves()}
    target_names = {leaf.name for leaf in t_target.get_leaves()}
    common = sorted(start_names & target_names, key=_leaf_sort_key)
    if len(common) < keep_leaves:
        raise AssertionError(
            f"Not enough shared leaves to prune: have {len(common)}, need {keep_leaves}"
        )

    kept = common[:keep_leaves]
    t_start.prune(kept, preserve_branch_length=True)
    t_target.prune(kept, preserve_branch_length=True)

    renumber_map = {
        old_name: str(i)
        for i, old_name in enumerate(sorted(kept, key=_leaf_sort_key))
    }

    for tree in (t_start, t_target):
        for leaf in tree.get_leaves():
            leaf.name = renumber_map[leaf.name]

    return t_start.write(format=1), t_target.write(format=1)


def _make_batch_from_tree_pair(tokenizer, start_tree, target_tree, time_point=0.0):
    sampled_newick, velocity = return_sampled_tree_orthant_velocity(
        start_tree, target_tree, time_point
    )
    with torch.no_grad():
        tokenized = _detach_tokenized_batch(tokenizer([sampled_newick]))

    return {
        "tokenized_trees": tokenized,
        "batched_time": torch.tensor([float(time_point)], dtype=torch.float32),
        "phyla_embeddings": None,
        "original_trees": [sampled_newick],
        "batched_velocity": [velocity],
        "num_leaves": [Tree(sampled_newick).n_leaves],
    }


def _make_batch_from_tree_pair_with_autoregressive(
    tokenizer,
    start_tree,
    target_tree,
    time_point=0.0,
    max_boundary_attempts=20,
):
    sampled_newick, velocity = return_sampled_tree_orthant_velocity(
        start_tree, target_tree, time_point
    )

    boundary_labels = []
    for _ in range(max_boundary_attempts):
        boundary_labels = return_sampled_tree_boundary_decisions(start_tree, target_tree)
        if boundary_labels:
            break

    if not boundary_labels:
        raise AssertionError(
            "Could not sample autoregressive boundary decisions for the sanity tree pair."
        )

    chosen_boundary = boundary_labels[0]
    with torch.no_grad():
        tokenized = _detach_tokenized_batch(tokenizer([sampled_newick]))
        tokenized_ar = _detach_tokenized_batch(tokenizer([chosen_boundary["newick"]]))

    return {
        "tokenized_trees": tokenized,
        "batched_time": torch.tensor([float(time_point)], dtype=torch.float32),
        "phyla_embeddings": None,
        "original_trees": [sampled_newick],
        "batched_velocity": [velocity],
        "num_leaves": [Tree(sampled_newick).n_leaves],
        "tokenized_autoregressive_trees": tokenized_ar,
        "batched_autoregressive_time": torch.tensor([0.0], dtype=torch.float32),
        "batched_autoregressive_labels": [chosen_boundary["labels"]],
    }


def _gather_supervised_velocity(module, batch):
    with torch.no_grad():
        v_pred, edge_split_masks, _ = module.forward(
            batch["tokenized_trees"],
            batch["batched_time"],
            batch["phyla_embeddings"],
        )

    velocity_labels = batch["batched_velocity"]
    num_leaves = batch["num_leaves"]

    preds = []
    labels = []
    matched_masks = []

    for b_idx, vel_dict in enumerate(velocity_labels):
        if not edge_split_masks[b_idx]:
            continue
        real_max_bit = max(int(m).bit_length() for m in edge_split_masks[b_idx])

        for original_vel, true_vel in vel_dict.items():
            vel = int(original_vel)
            if vel.bit_length() == real_max_bit + 1:
                vel = remove_bit(vel, int(num_leaves[b_idx]) - 1)
            elif vel.bit_length() > real_max_bit + 1:
                continue

            matched_vel = vel
            if matched_vel not in edge_split_masks[b_idx]:
                full_mask = (1 << real_max_bit) - 1
                complement_vel = full_mask ^ matched_vel
                if complement_vel in edge_split_masks[b_idx]:
                    matched_vel = complement_vel
                else:
                    continue

            n_bits = real_max_bit
            k_bits = int(matched_vel).bit_count()
            is_pendant = min(k_bits, n_bits - k_bits) == 1
            if is_pendant:
                continue

            edge_idx = edge_split_masks[b_idx].index(matched_vel)
            preds.append(v_pred[b_idx, edge_idx, 0].detach().cpu())
            labels.append(torch.tensor(float(true_vel), dtype=torch.float32))
            matched_masks.append(int(matched_vel))

    if not preds:
        raise AssertionError("No supervised non-pendant velocity edges were matched.")

    return torch.stack(preds).float(), torch.stack(labels).float(), matched_masks


def _pearson_corr(x, y):
    xm = x - x.mean()
    ym = y - y.mean()
    denom = xm.norm() * ym.norm()
    if float(denom) <= 1e-12:
        return 1.0 if torch.allclose(x, y) else 0.0
    return float((xm * ym).sum() / denom)


def _spearman_corr(x, y):
    xr = torch.empty_like(x)
    yr = torch.empty_like(y)
    xr[torch.argsort(x)] = torch.arange(x.numel(), dtype=x.dtype)
    yr[torch.argsort(y)] = torch.arange(y.numel(), dtype=y.dtype)
    return _pearson_corr(xr, yr)


def _topk_mask_overlap(pred, true, masks, k):
    k = min(int(k), int(pred.numel()))
    if k <= 0:
        return 1.0
    pred_idx = torch.topk(pred.abs(), k=k).indices.tolist()
    true_idx = torch.topk(true.abs(), k=k).indices.tolist()
    pred_masks = {masks[i] for i in pred_idx}
    true_masks = {masks[i] for i in true_idx}
    return len(pred_masks & true_masks) / float(k)


def _velocity_metrics(module, batch, topk=3):
    pred, true, masks = _gather_supervised_velocity(module, batch)
    mse = float(torch.mean((pred - true) ** 2))
    cosine = float(torch.nn.functional.cosine_similarity(pred, true, dim=0))
    pearson = _pearson_corr(pred, true)
    spearman = _spearman_corr(pred, true)

    # Tiny near-zero velocities are numerically unstable for sign comparisons.
    moving = true.abs() > 1e-3
    if int(moving.sum()) > 0:
        sign_acc = float((torch.sign(pred[moving]) == torch.sign(true[moving])).float().mean())
    else:
        sign_acc = 1.0

    topk_overlap = _topk_mask_overlap(pred, true, masks, k=topk)
    return {
        "mse": mse,
        "cosine": cosine,
        "pearson": pearson,
        "spearman": spearman,
        "sign_acc": sign_acc,
        "topk_overlap": topk_overlap,
        "n_supervised_edges": int(pred.numel()),
    }


class _OptimizerProxy:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()


_ORIGINAL_TENSOR_TO = torch.Tensor.to


def _tensor_to_cpu_for_cuda(self, *args, **kwargs):
    if args:
        device = args[0]
        if isinstance(device, str) and device.startswith("cuda"):
            args = ("cpu",) + args[1:]
        elif isinstance(device, torch.device) and device.type == "cuda":
            args = (torch.device("cpu"),) + args[1:]

    if "device" in kwargs:
        device = kwargs["device"]
        if isinstance(device, str) and device.startswith("cuda"):
            kwargs["device"] = "cpu"
        elif isinstance(device, torch.device) and device.type == "cuda":
            kwargs["device"] = torch.device("cpu")

    return _ORIGINAL_TENSOR_TO(self, *args, **kwargs)


class TestTrainingSanity(unittest.TestCase):
    def test_overfit_single_velocity_vector(self):
        random.seed(123)
        torch.manual_seed(123)
        device = torch.device("cpu")

        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=32,
            n_layers=1,
            n_heads=4,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=4,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=1,
            phyla_dim=16,
        ).to(device)

        module = TrainingModule(
            model=model,
            dataset=MagicMock(),
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            velocity_loss_mode="plain",
            velocity_sign_eps=1e-3,
        ).to(device)

        batch = _make_single_velocity_batch(
            tokenizer=model.tokenizer,
            n_leaves=10,
            seed=2024,
        )
        bootstrap_metrics = _velocity_metrics(module, batch, topk=3)
        self.assertGreaterEqual(
            bootstrap_metrics["n_supervised_edges"],
            6,
            "Not enough supervised internal edges to run a robust velocity-overfit sanity check.",
        )

        initial = _velocity_metrics(module, batch, topk=3)

        optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)
        max_steps = 120

        for step in range(max_steps):
            module.train()
            optimizer.zero_grad(set_to_none=True)
            logs = module.step(batch, autoregressive=False)
            loss = logs["loss"]
            self.assertTrue(torch.isfinite(loss).item(), "Training loss became non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.model.parameters(), max_norm=1.0)
            optimizer.step()

            if (step + 1) % 10 == 0:
                probe = _velocity_metrics(module, batch, topk=3)
                if (
                    probe["cosine"] > 0.99
                    and probe["pearson"] > 0.99
                    and probe["topk_overlap"] == 1.0
                    and probe["sign_acc"] > 0.90
                ):
                    break

        final = _velocity_metrics(module, batch, topk=3)

        self.assertLess(
            final["mse"],
            initial["mse"],
            "Overfit sanity check did not reduce velocity MSE.",
        )
        self.assertLess(
            final["mse"],
            max(3e-3, initial["mse"] * 0.1),
            f"MSE did not improve enough (initial={initial['mse']:.6f}, final={final['mse']:.6f})",
        )
        self.assertGreater(
            final["cosine"], 0.99, f"Cosine similarity too low: {final['cosine']:.6f}"
        )
        self.assertGreater(
            final["pearson"], 0.99, f"Pearson correlation too low: {final['pearson']:.6f}"
        )
        self.assertGreater(
            final["spearman"],
            0.95,
            f"Spearman correlation too low: {final['spearman']:.6f}",
        )
        self.assertGreater(
            final["sign_acc"], 0.95, f"Sign accuracy too low: {final['sign_acc']:.6f}"
        )
        self.assertEqual(
            final["topk_overlap"],
            1.0,
            f"Top-k velocity mask overlap not perfect: {final['topk_overlap']:.3f}",
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_random_sanity_check_is_deterministic_for_tree_and_velocity(
        self, _mock_build_index
    ):
        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )

        # random_sanity_check enforces fixed tree sources regardless tfiles input.
        real_one = ds.load_posterior_trees_from_tfiles([])[0]
        real_two = ds.load_posterior_trees_from_tfiles([])[0]
        self.assertEqual(real_one, real_two)

        rand_one = ds.sample_random_tree(real_one)
        rand_two = ds.sample_random_tree(real_one)
        self.assertEqual(rand_one, rand_two)

        # overfit_velocity_zero uses t=0.0, so sampled tree/velocity should be stable.
        sample_newick_1, velocity_1 = return_sampled_tree_orthant_velocity(
            rand_one, real_one, 0.0
        )
        sample_newick_2, velocity_2 = return_sampled_tree_orthant_velocity(
            rand_two, real_two, 0.0
        )

        self.assertEqual(sample_newick_1, sample_newick_2)
        self.assertEqual(set(velocity_1.keys()), set(velocity_2.keys()))
        for k in velocity_1:
            self.assertAlmostEqual(
                float(velocity_1[k]),
                float(velocity_2[k]),
                places=8,
                msg=f"Velocity mismatch for split {k}",
            )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_overfit_velocity_on_random_sanity_tree_pair(self, _mock_build_index):
        random.seed(321)
        torch.manual_seed(321)
        device = torch.device("cpu")

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )

        real_tree = ds.load_posterior_trees_from_tfiles([])[0]
        random_tree = ds.sample_random_tree(real_tree)
        # random_tree_small, real_tree_small = _prune_and_renumber_tree_pair(
        #     random_tree, real_tree, keep_leaves=8
        # )

        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=64,
            n_layers=2,
            n_heads=4,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=2,
            phyla_dim=16,
        ).to(device)

        module = TrainingModule(
            model=model,
            dataset=MagicMock(),
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            velocity_loss_mode="plain",
            velocity_sign_eps=1e-3,
        ).to(device)

        batch = _make_batch_from_tree_pair(
            tokenizer=model.tokenizer,
            start_tree=random_tree,
            target_tree=real_tree,
            time_point=0.0,
        )

        bootstrap_metrics = _velocity_metrics(module, batch, topk=3)
        self.assertGreaterEqual(
            bootstrap_metrics["n_supervised_edges"],
            3,
            "Not enough supervised internal edges in random_sanity_check pair.",
        )

        initial = _velocity_metrics(module, batch, topk=3)
        optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)
        best_metrics = dict(initial)
        best_state = copy.deepcopy(module.model.state_dict())

        for step in range(500):
            module.train()
            optimizer.zero_grad(set_to_none=True)
            logs = module.step(batch, autoregressive=False)
            loss = logs["loss"]
            self.assertTrue(torch.isfinite(loss).item(), "Training loss became non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.model.parameters(), max_norm=1.0)
            optimizer.step()

            if (step + 1) % 10 == 0:
                probe = _velocity_metrics(module, batch, topk=3)
                if (
                    probe["spearman"] > best_metrics["spearman"]
                    or (
                        probe["spearman"] == best_metrics["spearman"]
                        and probe["cosine"] >= best_metrics["cosine"]
                    )
                ):
                    best_metrics = dict(probe)
                    best_state = copy.deepcopy(module.model.state_dict())
                if (
                    probe["cosine"] > 0.99
                    and probe["pearson"] > 0.99
                    and probe["spearman"] > 0.95
                    and probe["topk_overlap"] == 1.0
                    and probe["sign_acc"] > 0.90
                ):
                    break

        module.model.load_state_dict(best_state)
        final = _velocity_metrics(module, batch, topk=3)

        self.assertLess(final["mse"], initial["mse"])
        self.assertLess(
            final["mse"],
            max(3e-3, initial["mse"] * 0.1),
            f"MSE did not improve enough (initial={initial['mse']:.6f}, final={final['mse']:.6f})",
        )
        self.assertGreater(final["cosine"], 0.95)
        self.assertGreater(final["pearson"], 0.95)
        self.assertGreater(final["spearman"], 0.95)
        self.assertGreater(final["sign_acc"], 0.90)
        self.assertEqual(final["topk_overlap"], 1.0)

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_training_step_with_autoregressive_still_converges_velocity(
        self, _mock_build_index
    ):
        random.seed(777)
        torch.manual_seed(777)
        device = torch.device("cpu")

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )
        real_tree = ds.load_posterior_trees_from_tfiles([])[0]
        random_tree = ds.sample_random_tree(real_tree)

        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=64,
            n_layers=2,
            n_heads=4,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=2,
            phyla_dim=16,
        ).to(device)

        dataset_stub = MagicMock()
        dataset_stub.msa_distance = True
        dataset_stub.chosen_tree = (0, 0, 1)

        module = TrainingModule(
            model=model,
            dataset=dataset_stub,
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            velocity_loss_mode="plain",
            velocity_sign_eps=1e-3,
            training_step_velocity_weight=100.0,
            training_step_autoregressive_weight=0.1,
        ).to(device)

        batch = _make_batch_from_tree_pair_with_autoregressive(
            tokenizer=model.tokenizer,
            start_tree=random_tree,
            target_tree=real_tree,
            time_point=0.0,
        )
        dataset_stub.chosen_tree = (0, int(batch["num_leaves"][0]), 1)

        bootstrap_metrics = _velocity_metrics(module, batch, topk=3)
        self.assertGreaterEqual(
            bootstrap_metrics["n_supervised_edges"],
            3,
            "Not enough supervised internal edges in random_sanity_check pair.",
        )

        initial = _velocity_metrics(module, batch, topk=3)
        best_metrics = dict(initial)
        best_state = copy.deepcopy(module.model.state_dict())

        optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)
        module.optimizers = MagicMock(return_value=_OptimizerProxy(optimizer))
        module.manual_backward = lambda loss: loss.backward()
        module.clip_gradients = lambda _opt, gradient_clip_val, gradient_clip_algorithm: torch.nn.utils.clip_grad_norm_(
            module.model.parameters(), max_norm=float(gradient_clip_val)
        )
        module.log = MagicMock()

        with patch.object(torch.Tensor, "to", _tensor_to_cpu_for_cuda):
            for step in range(500):
                module.train()
                loss = module.training_step(batch, step)
                self.assertTrue(
                    torch.isfinite(loss).item(),
                    "training_step loss became non-finite.",
                )

                if (step + 1) % 5 == 0:
                    probe = _velocity_metrics(module, batch, topk=3)
                    if (
                        probe["mse"] < best_metrics["mse"]
                        or (
                            probe["mse"] == best_metrics["mse"]
                            and probe["cosine"] >= best_metrics["cosine"]
                        )
                    ):
                        best_metrics = dict(probe)
                        best_state = copy.deepcopy(module.model.state_dict())

                    if (
                        probe["cosine"] > 0.95
                        and probe["pearson"] > 0.95
                        and probe["spearman"] > 0.90
                        and probe["topk_overlap"] == 1.0
                        and probe["sign_acc"] > 0.85
                    ):
                        break

        module.model.load_state_dict(best_state)
        final = _velocity_metrics(module, batch, topk=3)

        self.assertLess(final["mse"], initial["mse"])
        self.assertLess(
            final["mse"],
            max(2e-2, initial["mse"] * 0.25),
            f"MSE did not improve enough (initial={initial['mse']:.6f}, final={final['mse']:.6f})",
        )
        self.assertGreater(final["cosine"], 0.90)
        self.assertGreater(final["pearson"], 0.90)
        self.assertGreater(final["spearman"], 0.90)
        self.assertGreater(final["sign_acc"], 0.80)
        self.assertEqual(final["topk_overlap"], 1.0)


if __name__ == "__main__":
    unittest.main()
