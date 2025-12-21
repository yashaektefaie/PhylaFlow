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
import re
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from utils.bhv_utils import return_sampled_tree_velocity
import random
from model.treeTokenizer import TreeFeatureTokenizer
from utils.random_tree import Tree
from ete3 import Tree as EteTree


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
        filter_ids: Optional[List[str]] = None,
    ) -> None:
        self.nexus_root = nexus_root
        self.mrbayes_root = mrbayes_root
        self.filter_ids = filter_ids


        # Internal containers
        self._ids: List[str] = []  # populated by build_index()
        self._index: List[Dict[str, Any]] = []  # list of sample metadata dicts
        self._id_to_idx: Dict[str, int] = {}
        self._trees_cache: Dict[str, Any] = {}  # TODO: store parsed tree objects
        self._seqs_cache: Dict[str, Any] = {}  # TODO: store parsed sequences

        # Build index immediately; optionally preload
        self.build_index()


    def __len__(self) -> int:  # Required for torch Dataset
        return len(self._ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:  # Required for torch Dataset
        meta = self._index[index]

        seqs, taxa_order = self.parse_nexus(meta['nexus_path'])
        real_tree = random.sample(self.load_posterior_trees_from_tfiles(meta["tree_paths"]), 1)[0]
        random_tree = self.sample_random_tree(real_tree)
        timepoint = random.uniform(0, 1)
        newick, velocity = return_sampled_tree_velocity(random_tree, real_tree, timepoint)

        sample = {
            "id": meta["id"],
            "nexus_path": meta["nexus_path"],
            "tree_paths": meta["tree_paths"],  # list of .t files, may be 1
            # Placeholders for parsed content:
            "sequences": seqs,
            "taxa_order": taxa_order, 
            "newick_tree": newick,
            "velocity": velocity,
            "timepoint": timepoint,
        }

        return sample
    
    def sample_random_tree(self, real_tree):
        """
        real_tree: Newick string or an ETE Tree.
        Returns: Newick string for a random tree with the same leaf names.
        """
        # Parse to ETE
        if isinstance(real_tree, str):
            t = EteTree(real_tree, format=1)
        else:
            t = real_tree

        # Collect leaf names; order however you like (here: sorted for determinism)
        leaves = t.get_leaves()
        leaves_sorted = sorted(leaves, key=lambda x: x.name)
        n_leaves = len(leaves_sorted)

        # Build a random unrooted binary tree on {1,...,n_leaves}
        rt = Tree(num_leaves=n_leaves, random=True)

        # Produce Newick with the same taxa names but random topology/lengths
        random_newick = str(rt)
        return random_newick

    def extract_newick_from_line(self, line: str) -> str:
        """
        Given a line from a .t/.trees file, extract the Newick string.
        Handles BEAST-style 'tree STATE_... = [&R] (..);' or raw '(..);'.
        Returns '' if no Newick found.
        """
        line = line.strip()
        if not line or line.startswith("#"):
            return ""
        
        # Find first '(' and last ')' or ';'
        start = line.find("(")
        if start == -1:
            return ""
        
        # Newick typically ends at ';', but sometimes there's stuff after.
        # We'll go to the last ';' if it exists, else end of line.
        end = line.rfind(";")
        if end == -1:
            end = len(line)
        else:
            end = end + 1  # include ';'
        
        newick = line[start:end].strip()
        return newick if newick else ""


    def load_posterior_trees_from_tfiles(self,
        tree_files: List[str],
        burn_in_fraction: float = 0.25,
    ) -> List[str]:
        """
        Given a list of .t/.trees files, extract posterior Newick trees
        applying a per-file burn-in and thinning.

        Args
        ----
        tree_files : list of paths to .t files
        burn_in_fraction : fraction of samples per file to discard as burn-in

        Returns
        -------
        trees : list of Newick strings (posterior samples)
        """
        
        all_trees = []

        for path in tree_files:
            file_trees = []

            with open(path, "r") as f:
                for line in f:
                    newick = self.extract_newick_from_line(line)
                    if newick:
                        file_trees.append(newick)

            if not file_trees:
                continue

            # Apply burn-in per file
            burn = int(len(file_trees) * burn_in_fraction)
            kept = file_trees[burn:]

            all_trees.extend(kept)

        return all_trees

    def parse_nexus(self, path: str) -> tuple[Dict[str, str], List[str]]:
        """Parse sequences from a NEXUS alignment file.

        Returns a dict mapping taxon/sequence ID to its sequence string.
        This lightweight parser targets common cases:
        - MATRIX block under BEGIN DATA/CHARACTERS
        - Interleaved or non-interleaved; sequence chunks are concatenated
        - Comments in square brackets are stripped

        Note: For complex/edge-case NEXUS dialects, consider using Biopython.
        """
        taxa_order = []
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Remove NEXUS comments [ ... ] (non-nested) across lines
        text = re.sub(r"\[.*?\]", "", text, flags=re.DOTALL)

        lines = [ln.strip() for ln in text.splitlines()]
        seqs: Dict[str, str] = {}
        in_matrix = False

        for raw in lines:
            line = raw
            if not in_matrix:
                # Look for the 'MATRIX' keyword (case-insensitive)
                idx = line.lower().find("matrix")
                if idx == -1:
                    continue

                # Switch to matrix mode; process any remainder on the same line
                in_matrix = True
                remainder = line[idx + len("matrix"):].strip()
                if remainder:
                    # Process potential inline first entry after MATRIX
                    term = False
                    if ";" in remainder:
                        remainder, _sep, _after = remainder.partition(";")
                        term = True
                    tokens = remainder.split()
                    if len(tokens) >= 2:
                        name = tokens[0]
                        if name not in taxa_order:
                            taxa_order.append(name)
                        seq = "".join(tokens[1:])
                        seqs[name] = seqs.get(name, "") + seq
                    if term:
                        break
                continue

            # In MATRIX: accumulate lines until a ';'
            if not line:
                continue

            terminated = False
            if ";" in line:
                line, _sep, _after = line.partition(";")
                terminated = True

            line = line.strip()
            if not line:
                if terminated:
                    break
                continue

            tokens = line.split()
            if len(tokens) >= 2:
                name = tokens[0]
                if name not in taxa_order:
                    taxa_order.append(name)
                seq = "".join(tokens[1:])
                seqs[name] = seqs.get(name, "") + seq
            # Lines with fewer than 2 tokens are ignored

            if terminated:
                break

        unaligned_seqs = {}
        for i in seqs:
            unaligned_seqs[i] = seqs[i].replace('-', '')

        return unaligned_seqs, taxa_order

    def build_index(self) -> None:
        """Scan nexus_root and mrbayes_root to build ID->paths mapping.

        Strategy:
        - Accept .nex or .nexus as nexus files.
        - ID := basename without extension.
        - For each ID, look for mrbayes_root/ID directory and collect .t files.
        - Include all .t files.
        """
        nexus_exts = {".nex", ".nexus"}
        if not os.path.isdir(self.nexus_root):
            raise Exception(f"Nexus root is not a directory: {self.nexus_root}")

        ids: List[str] = []
        id_to_nexus: Dict[str, str] = {}

        for name in os.listdir(self.nexus_root):
            base, ext = os.path.splitext(name)
            if self.filter_ids is not None and base not in self.filter_ids:
                continue
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


class PhylaDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for managing TreeDataset splits.

    Responsibilities:
    - prepare_data(): download / generate raw data (non-distributed)
    - setup(stage): create Train/Val/Test datasets (distributed safe)
    - train_dataloader()/val_dataloader()/test_dataloader()/predict_dataloader()
    """

    def __init__(
        self,
        config,
        train_ids: List[str],
        test_ids: List[str],
    ) -> None:
        super().__init__()
        self.nexus_dir = config['data']['nexus_root']
        self.mrbayes_dir = config['data']['mrbayes_root']
        self.batch_size = config['data']['batch_size']
        self.num_workers = config['data']['num_workers']
        self.pin_memory = config['data']['pin_memory']

        self.train_ids = train_ids
        self.test_ids = test_ids

        self.dataset_train = TreeDataset(self.nexus_dir, self.mrbayes_dir, filter_ids=self.train_ids)
        self.dataset_val = TreeDataset(self.nexus_dir, self.mrbayes_dir, filter_ids=self.test_ids)
        self.tree_tokenizer = TreeFeatureTokenizer(config['model']['num_node_types'], config['model']['num_edge_types'], config['model']['hidden_dim'],)


    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        """Custom collate function if needed."""
        trees_to_tokenize = [item['newick_tree'] for item in batch]
        tokenized_trees = self.tree_tokenizer(trees_to_tokenize)
        to_run = {
            "tokenized_trees": tokenized_trees,
            "batched_velocity": [item['velocity'] for item in batch],
            "batched_time": torch.tensor([item['timepoint'] for item in batch], dtype=torch.float32),
            #"phyla_embeddings": torch.tensor([item['phyla_embedding'] for item in batch], dtype=torch.float32),
            "phyla_embeddings": None,
        }
        return to_run


def test():
    dm = TreeDataset(nexus_root="/Users/yashaektefaie/Desktop/PhylaFlow/example_data/nexus/",
                     mrbayes_root="/Users/yashaektefaie/Desktop/PhylaFlow/example_data/runs/")
    res_one = dm[0]
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    test()

