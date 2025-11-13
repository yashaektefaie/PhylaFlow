"""Dataset and DataModule skeletons for PhylaFlow.

This version targets a common layout for Nexus alignments and MrBayes outputs:

data_root/
    nexus/                     # directory of source Nexus files (one per ID)
        <id>.nex | <id>.nexus
    runs/                      # directory containing MrBayes outputs per ID
        <id>/
            <id>_DNA.run1.t       # tree samples (we'll index .t files)
            <id>_DNA.run2.t
            ... other MrBayes files ...

"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class TreeDataset(Dataset):
    """Dataset mapping IDs to Nexus sequences and MrBayes tree files.

    Layout assumptions (configurable):
    - nexus_root contains files: <id>.nex or <id>.nexus
    - mrbayes_root contains subdirs per <id> with one or more .t files
      (e.g., <id>_DNA.run1.t, <id>_DNA.run2.t)

    No parsing is performed here by default; this class only builds an index
    and returns paths with placeholders. Fill in parse methods as needed.

    Args:
        nexus_root: Directory with Nexus source files.
        mrbayes_root: Directory with MrBayes output folders.
        prefer_run: Which run's .t to prefer ("run1", "run2", "any").
        transform: Optional callable applied to each sample dict.
        cache: If True, eagerly parse and cache sequences/trees (TODO).
    """

    def __init__(
        self,
        nexus_root: str,
        mrbayes_root: str,
    ) -> None:
        self.nexus_root = nexus_root
        self.mrbayes_root = mrbayes_root


        # Internal containers
        self._ids: List[str] = []  # populated by build_index()
        self._index: List[Dict[str, Any]] = []  # list of sample metadata dicts
        self._id_to_idx: Dict[str, int] = {}
        self._trees_cache: Dict[str, Any] = {}  # TODO: store parsed tree objects
        self._seqs_cache: Dict[str, Any] = {}  # TODO: store parsed sequences

        # Build index immediately; optionally preload
        self.build_index()
        import pdb; pdb.set_trace()


    def __len__(self) -> int:  # Required for torch Dataset
        return len(self._ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:  # Required for torch Dataset
        meta = self._index[index]


        sample = {
            "id": meta["id"],
            "nexus_path": meta["nexus_path"],
            "tree_paths": meta["tree_paths"],  # list of .t files, may be 1
            # Placeholders for parsed content:
            "sequences": None,  # TODO: call self.parse_nexus(meta["nexus_path"]) if needed
            "trees": None,      # TODO: call self.parse_tree_file(path) for each tree file
        }

        return sample

    # --- Additional helper / domain-specific methods (placeholders) ---
    def parse_tree_file(self, path: str) -> Any:
        """Parse a tree file (e.g., Newick) and return structured object.
        TODO: implement actual parsing logic.
        """
        # TODO: open path and parse content
        raise NotImplementedError("parse_tree_file not implemented")

    def parse_nexus(self, path: str) -> Any:
        """Parse sequences from a Nexus file at path.
        TODO: implement using, e.g., BioPython (Bio.Nexus or AlignIO).
        """
        # TODO: open path and parse sequences/taxa
        raise NotImplementedError("parse_nexus not implemented")

    def build_index(self) -> None:
        """Scan nexus_root and mrbayes_root to build ID->paths mapping.

        Strategy:
        - Accept .nex or .nexus as nexus files.
        - ID := basename without extension.
        - For each ID, look for mrbayes_root/ID directory and collect .t files.
        - If prefer_run is set to "run1" or "run2", prefer matching .t file.
        - Otherwise include all .t files.
        """
        nexus_exts = {".nex", ".nexus"}
        if not os.path.isdir(self.nexus_root):
            raise Exception(f"Nexus root is not a directory: {self.nexus_root}")

        ids: List[str] = []
        id_to_nexus: Dict[str, str] = {}

        for name in os.listdir(self.nexus_root):
            base, ext = os.path.splitext(name)
            if ext.lower() in nexus_exts:
                ids.append(base)
                id_to_nexus[base] = os.path.join(self.nexus_root, name)

        ids.sort()

        index: List[Dict[str, Any]] = []
        for id_ in ids:
            run_dir = os.path.join(self.mrbayes_root, id_)
            tree_paths: List[str] = []
            if os.path.isdir(run_dir):
                # collect .t files
                t_files = [f for f in os.listdir(run_dir) if f.endswith(".t")]
                tree_paths = [os.path.join(run_dir, f) for f in sorted(t_files)]

            meta = {
                "id": id_,
                "nexus_path": id_to_nexus[id_],
                "tree_paths": tree_paths,  # may be empty if runs missing
            }
            index.append(meta)

        self._ids = ids
        self._index = index
        self._id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}

    def preload(self) -> None:
        """Preload all trees and sequences into memory if caching.
        TODO: iterate over index and call parse helpers.
        """
        # TODO: implement preload logic
        pass


class PhylaDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for managing TreeDataset splits.

    Responsibilities:
    - prepare_data(): download / generate raw data (non-distributed)
    - setup(stage): create Train/Val/Test datasets (distributed safe)
    - train_dataloader()/val_dataloader()/test_dataloader()/predict_dataloader()
    """

    def __init__(
        self,
        data_root: str,
        nexus_subdir: str = "nexus",
        mrbayes_subdir: str = "runs",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.nexus_subdir = nexus_subdir
        self.mrbayes_subdir = mrbayes_subdir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Placeholders for datasets
        self.dataset_train: Optional[TreeDataset] = None
        self.dataset_val: Optional[TreeDataset] = None
        self.dataset_test: Optional[TreeDataset] = None
        self.dataset_predict: Optional[TreeDataset] = None

    def prepare_data(self) -> None:
        """Download or generate data if needed.
        TODO: implement data acquisition (only called on 1 process).
        """
        # TODO: e.g., download archives, verify checksums
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets for different stages.
        TODO: instantiate TreeDataset objects and perform splits.
        """
        nexus_dir = os.path.join(self.data_root, self.nexus_subdir)
        runs_dir = os.path.join(self.data_root, self.mrbayes_subdir)

        # TODO: perform real split logic
        if stage in ("fit", None):
            # TODO: build train & val datasets
            self.dataset_train = TreeDataset(nexus_dir, runs_dir, transform=None, cache=False)
            self.dataset_val = TreeDataset(nexus_dir, runs_dir, transform=None, cache=False)

        if stage in ("test", None):
            # TODO: build test dataset
            self.dataset_test = TreeDataset(nexus_dir, runs_dir, transform=None, cache=False)

        if stage in ("predict", None):
            # TODO: build predict/inference dataset
            self.dataset_predict = TreeDataset(nexus_dir, runs_dir, transform=None, cache=False)

    def train_dataloader(self) -> DataLoader:
        # TODO: customize collate_fn if needed
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    # --- Optional utility hooks / helpers ---
    def teardown(self, stage: Optional[str] = None) -> None:
        """Release resources after training/testing.
        TODO: clear caches or close file handles.
        """
        pass

    def size(self) -> Tuple[int, int, int]:
        """Return sizes of (train, val, test) datasets.
        TODO: implement with actual lengths.
        """
        train = len(self.dataset_train) if self.dataset_train else 0
        val = len(self.dataset_val) if self.dataset_val else 0
        test = len(self.dataset_test) if self.dataset_test else 0
        return train, val, test

    def describe(self) -> Dict[str, Any]:
        """Return a summary dictionary for logging/debugging.
        TODO: enrich with domain-specific metadata.
        """
        return {
            "root": self.data_root,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "sizes": self.size(),
        }


def test():
    dm = TreeDataset(nexus_root="/Users/yashaektefaie/Desktop/PhylaFlow/example_data/nexus/",
                     mrbayes_root="/Users/yashaektefaie/Desktop/PhylaFlow/example_data/runs/")


test()


