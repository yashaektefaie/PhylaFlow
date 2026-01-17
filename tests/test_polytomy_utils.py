import pickle
import unittest
from pathlib import Path

import torch

from utils.utils import get_batch_polytomy_indices


class TestGetBatchPolytomyIndicesDebugBatch(unittest.TestCase):
    def _load_debug_batch(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "debug_batch.pkl"
        self.assertTrue(path.exists(), f"Missing debug batch at {path}")

        if torch.cuda.is_available():
            return torch.load(path, weights_only=False)
        else:
            # Force CUDA tensors to load on CPU even on CPU-only machines
            import io
            with open(path, "rb") as f:
                buffer = io.BytesIO(f.read())

            class CPUUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == "torch.storage" and name == "_load_from_bytes":
                        return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
                    return super().find_class(module, name)

            return CPUUnpickler(buffer).load()

    def _extract_edge_data(self, batch):
        if isinstance(batch, dict):
            if "edge_split_masks" in batch and "edge_mask" in batch:
                return batch["edge_split_masks"], batch["edge_mask"], batch.get("num_leaves")
            if "tokenized_trees" in batch:
                tokenized = batch["tokenized_trees"]
                edge_mask = tokenized[-2]
                edge_split_masks = tokenized[-1]
                num_leaves = batch.get("num_leaves")
                if num_leaves is None and len(tokenized) >= 4:
                    leaf_mask = tokenized[3]
                    if isinstance(leaf_mask, torch.Tensor):
                        num_leaves = [int(v.item()) for v in leaf_mask.sum(dim=1)]
                return edge_split_masks, edge_mask, num_leaves

        if isinstance(batch, (list, tuple)) and len(batch) >= 7:
            edge_mask = batch[-2]
            edge_split_masks = batch[-1]
            leaf_mask = batch[3]
            num_leaves = None
            if isinstance(leaf_mask, torch.Tensor):
                num_leaves = [int(v.item()) for v in leaf_mask.sum(dim=1)]
            return edge_split_masks, edge_mask, num_leaves

        raise AssertionError(
            f"Unsupported debug batch structure: {type(batch)}"
        )

    def _normalize_num_leaves(self, num_leaves, edge_split_masks):
        if isinstance(num_leaves, torch.Tensor):
            if num_leaves.numel() == 1:
                return int(num_leaves.item())
            return int(num_leaves.max().item())
        if isinstance(num_leaves, (list, tuple)):
            return int(max(num_leaves))
        if isinstance(num_leaves, int):
            return num_leaves

        max_bit = 0
        for splits in edge_split_masks:
            for split in splits:
                max_bit = max(max_bit, int(split).bit_length())
        return max_bit

    def _normalize_edge_split_masks(self, edge_split_masks):
        if isinstance(edge_split_masks, torch.Tensor):
            return [edge_split_masks[b] for b in range(edge_split_masks.size(0))]
        return edge_split_masks

    def _normalize_edge_mask(self, edge_mask):
        if isinstance(edge_mask, torch.Tensor):
            return edge_mask
        return torch.tensor(edge_mask)

    def test_debug_batch_polytomy_indices_not_empty(self):
        batch = self._load_debug_batch()
        edge_split_masks, edge_mask, num_leaves = self._extract_edge_data(batch)
        edge_split_masks = self._normalize_edge_split_masks(edge_split_masks)
        edge_mask = self._normalize_edge_mask(edge_mask)
        num_leaves = self._normalize_num_leaves(num_leaves, edge_split_masks)

        batch_polytomy_index, _ = get_batch_polytomy_indices(
            edge_split_masks,
            edge_mask,
            min_children=3,
            include_root=True,
            num_leaves=num_leaves,
        )

        full_mask = (1 << num_leaves) - 1
        root_bit  = 1 << (num_leaves - 1)

        root_mask = full_mask ^ root_bit
        print("Root mask: ", [i for i in range(int(root_mask).bit_length()) if (int(root_mask) >> i) & 1])

        for test in _[0][0]: 
            print([i for i in range(int(test).bit_length()) if (int(test) >> i) & 1])

        group_count = sum(len(groups) for groups in batch_polytomy_index)
        total_indices = sum(
            int(group.numel())
            for groups in batch_polytomy_index
            for group in groups
        )

        self.assertGreater(
            group_count,
            0,
            "Expected at least one polytomy group for debug_batch.pkl",
        )
        self.assertGreater(
            total_indices,
            0,
            "Expected polytomy groups to contain at least one index",
        )


if __name__ == "__main__":
    unittest.main()
