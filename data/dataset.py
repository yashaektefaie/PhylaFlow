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
from utils.bhv_utils import (
    return_sampled_tree_orthant_velocity,
    return_sampled_tree_boundary_decisions,
)
import random
from model.treeTokenizer import TreeFeatureTokenizer
from utils.random_tree import Tree
from ete3 import Tree as EteTree


class SizeDetector:
    def __init__(self, max_aa=None):
        self.max_aa = max_aa

    def update_max_aa(self, new_max_aa):
        self.max_aa = new_max_aa


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
        validation=False,
    ) -> None:
        self.nexus_root = nexus_root
        self.mrbayes_root = mrbayes_root
        self.filter_ids = filter_ids
        self.validation = validation
        self.size_detector = SizeDetector()
        # State tracker for adaptive batching (index, subtree_size, num_subtrees)
        # Default initialization
        self.chosen_tree = (0, 100, 1)
        self.name_to_seq = {}

        # Internal containers
        self._ids: List[str] = []  # populated by build_index()
        self._index: List[Dict[str, Any]] = []  # list of sample metadata dicts
        self._id_to_idx: Dict[str, int] = {}

        # Build index immediately; optionally preload
        self.build_index()

    def __len__(self) -> int:  # Required for torch Dataset
        return len(self._ids)

    def return_number_leaves(self, index: int) -> int:
        """Return number of leaves in the alignment for the given index."""
        meta = self._index[index]
        seqs, _ = self.parse_nexus(meta['nexus_path'])
        return len(seqs)
    
    def return_posterior_trees(self, index: int) -> List[str]:
        """Return list of posterior Newick trees for the given index.

        Applies burn-in and thinning as per load_posterior_trees_from_tfiles.
        """
        meta = self._index[index]
        tree_paths = meta["tree_paths"]
        trees = self.load_posterior_trees_from_tfiles(tree_paths)
        return trees
    
    def return_nexus_filepath(self, index: int) -> str:
        """Return the Nexus file path for the given index."""
        meta = self._index[index]
        return meta['nexus_path']
    
    def return_nexus_number_to_name(self, index: int) -> Dict[int, str]:
        """Return mapping from taxon number to name for the given index."""
        meta = self._index[index]
        _, taxa_order = self.parse_nexus(meta['nexus_path'])
        num_to_name = {i: name for i, name in enumerate(taxa_order)}
        return num_to_name

    def __getitem__(
        self, index: int, preset_subtree_size: Optional[int] = None
    ) -> Dict[str, Any]:  # Required for torch Dataset
        meta = self._index[index]
        if self.validation:
            return {
                "id": meta["id"],
                "posterior_trees": self.return_posterior_trees(index),
            }

        seqs, taxa_order = self.parse_nexus(meta["nexus_path"])

        # Update name_to_seq cache (dumb update for now)
        self.name_to_seq = seqs

        # Attempt to parse translation block from the first tree file
        translate_map = {}
        if meta["tree_paths"]:
            translate_map = self.parse_translate_block(meta["tree_paths"][0])

        trees = self.load_posterior_trees_from_tfiles(meta["tree_paths"])
        if not trees:
            # Fallback: try to reload or skip. For now, raise informative error or return another item
            print(
                f"Dataset Warning: No trees found in {meta['tree_paths']}. Skipping/Replacing with index 0."
            )
            return self.__getitem__(0, preset_subtree_size)

        real_tree_newick = random.sample(trees, 1)[0]

        # Pruning logic for adaptive batching
        t = EteTree(real_tree_newick, format=1)
        leaves = t.get_leaves()

        if preset_subtree_size is not None and len(leaves) > preset_subtree_size:
            kept_leaves = random.sample(leaves, preset_subtree_size)
            t.prune(kept_leaves, preserve_branch_length=True)
            # real_tree_newick = t.write(format=1) # Don't write yet, wait for re-indexing
            # Update leaves for size tracking
            leaves = t.get_leaves()

        current_size = len(leaves)
        self.chosen_tree = (index, current_size, 1)  # (index, size, num_subtrees)

        # Normalize tree indices to 0..N-1 and subset sequences
        # Sort leaves for deterministic indexing
        leaves.sort(key=lambda x: x.name)

        new_seqs = {}
        original_names_map = {}

        for i, leaf in enumerate(leaves):
            original_node_name = leaf.name
            # Resolve taxon name: check translate map, else use node name
            taxon_name = translate_map.get(original_node_name, original_node_name)

            # Map new index (0..N-1) to sequence
            new_idx_str = str(i)
            # Store sequences using the new index as key
            new_seqs[new_idx_str] = seqs.get(taxon_name, "")

            # Rename leaf in the tree
            leaf.name = new_idx_str

            # Record mapping if needed
            original_names_map[new_idx_str] = taxon_name

        # Serialize the normalized tree
        real_tree_newick = t.write(format=1)

        # Re-parse purely to ensure we are passing consistent objects
        # (Though prune modifies in-place, let's keep it safe)
        t_pruned = EteTree(real_tree_newick, format=1)
        random_tree = self.sample_random_tree(t_pruned)
        timepoint = random.uniform(0, 1)

        # Both trees now use "0".."N-1" names, so bhv utils will work happily
        newick, velocity = return_sampled_tree_orthant_velocity(
            random_tree, real_tree_newick, timepoint
        )
        final_labels = return_sampled_tree_boundary_decisions(
            random_tree, real_tree_newick
        )
        chosen_autoregressive_event = random.choice(final_labels)
        num_to_name = self.return_nexus_number_to_name(index)
        sample = {
            "id": meta["id"],
            "nexus_path": meta["nexus_path"],
            "tree_paths": meta["tree_paths"],  # list of .t files, may be 1
            # Placeholders for parsed content:
            "sequences": new_seqs,
            "taxa_order": list(new_seqs.keys()),  # e.g. ["0", "1", ...]
            "newick_tree": newick,
            "velocity": velocity,
            "timepoint": timepoint,
            "autoregressive_newick": chosen_autoregressive_event['newick'],
            "autoregressive_labels": chosen_autoregressive_event['labels'],
            "num_to_name": num_to_name,
        }

        return sample

    def parse_translate_block(self, path: str) -> Dict[str, str]:
        """Extract 'translate' block from a Nexus/MrBayes file to map IDs to Taxon names."""
        mapping = {}
        in_translate = False
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Check for start of translate block
                    if not in_translate:
                        if line.lower().startswith("translate"):
                            in_translate = True
                            # Remove 'translate' keyword to process rest of line
                            line = line[9:].strip()
                            if not line:
                                continue

                    if in_translate:
                        # Parsing entries like: 1 Marmota_marmota, 2 Jaculus, ...
                        # Ends with ;
                        term = False
                        if ";" in line:
                            term = True
                            line = line.replace(";", "")

                        # Split by comma
                        tokens = line.split(",")
                        for token in tokens:
                            token = token.strip()
                            if not token:
                                continue
                            parts = token.split()
                            if len(parts) >= 2:
                                # mapping ID -> Name
                                mapping[parts[0]] = parts[1]

                        if term:
                            break
        except Exception:
            # If parsing fails or file not found, return empty dict
            pass
        return mapping

    def return_max_length(self, name_to_seq):
        if not name_to_seq:
            return 0
        return max(len(s) for s in name_to_seq.values())

    def sample_random_tree(self, real_tree, subtree_size: Optional[int] = None):
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
        # Random tree creates leaves 0..n_leaves-1
        rt = Tree(num_leaves=n_leaves, random=True)

        # Map 0..n_leaves-1 back to the sorted real leaf names
        for i, real_leaf in enumerate(leaves_sorted):
            rt.id_to_name[i] = real_leaf.name

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

    def load_posterior_trees_from_tfiles(
        self,
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
                remainder = line[idx + len("matrix") :].strip()
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
            unaligned_seqs[i] = seqs[i].replace("-", "")

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
        self.nexus_dir = config["data"]["nexus_root"]
        self.mrbayes_dir = config["data"]["mrbayes_root"]
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.pin_memory = config["data"]["pin_memory"]

        self.train_ids = train_ids
        self.test_ids = test_ids

        self.dataset_train = TreeDataset(
            self.nexus_dir, self.mrbayes_dir, filter_ids=self.train_ids
        )
        self.dataset_val = TreeDataset(
            self.nexus_dir, self.mrbayes_dir, filter_ids=self.test_ids, validation=True
        )
        self.tree_tokenizer = TreeFeatureTokenizer(
            config["model"]["num_node_types"],
            config["model"]["num_edge_types"],
            config["model"]["hidden_dim"],
        )
        self.msa_distance = True

    @property
    def chosen_tree(self):
        return self.dataset_train.chosen_tree

    @chosen_tree.setter
    def chosen_tree(self, value):
        self.dataset_train.chosen_tree = value

    @property
    def size_detector(self):
        return self.dataset_train.size_detector

    @property
    def name_to_seq(self):
        return self.dataset_train.name_to_seq

    def return_max_length(self, name_to_seq):
        return self.dataset_train.return_max_length(name_to_seq)

    def __getitem__(self, *args, **kwargs):
        return self.dataset_train.__getitem__(*args, **kwargs)

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

    def collate_fn(self, batch, preset_subtree_num=None):
        """Custom collate function if needed."""
        if [len(item) for item in batch][0] == 2:  # validation mode
            ids = [item["id"] for item in batch]
            posterior_trees = [item["posterior_trees"] for item in batch]
            mappings = [item['num_to_name'] for item in batch]
            phyla_embeddings = None

            return {
                "ids": ids,
                "posterior_trees": posterior_trees,
                "phyla_embeddings": phyla_embeddings,,
                "mappings": mappings,
                "nexus_filepaths": [item['nexus_path'] for item in batch],
                "tree_paths": [item['tree_paths'] for item in batch],

            }

        # preset_subtree_num is accepted but currently unused in logic below
        # Just ensuring signature matches call site

        trees_to_tokenize = [item["newick_tree"] for item in batch]
        # Tokenizer runs in worker if num_workers > 0, so must disable gradients
        # to avoid pickling errors (grad_fn cannot be pickled).
        with torch.no_grad():
            tokenized_trees = self.tree_tokenizer(trees_to_tokenize)
        num_leaves = [len(batch[i]["sequences"]) for i in range(len(batch))]

        autoregressive_trees_to_tokenize = [
            item["autoregressive_newick"] for item in batch
        ]
        autoregressive_tokenized_trees = self.tree_tokenizer(
            autoregressive_trees_to_tokenize
        )
        mappings = [item['num_to_name'] for item in batch]

        to_run = {
            "tokenized_trees": tokenized_trees,
            "tokenized_autoregressive_trees": autoregressive_tokenized_trees,
            "nexus_filepaths": [item['nexus_path'] for item in batch],
            "tree_paths": [item['tree_paths'] for item in batch],
            "original_trees": [item["newick_tree"] for item in batch],
            "batched_velocity": [item["velocity"] for item in batch],
            "batched_autoregressive_labels": [
                item["autoregressive_labels"] for item in batch
            ],
            "batched_time": torch.tensor(
                [item["timepoint"] for item in batch], dtype=torch.float32
            ),
            # "phyla_embeddings": torch.tensor([item['phyla_embedding'] for item in batch], dtype=torch.float32),
            "phyla_embeddings": None,
            "num_leaves": num_leaves,
            "mappings": mappings,
        }
        return to_run


def test():
    dm = TreeDataset(
        nexus_root="/Users/yashaektefaie/Desktop/PhylaFlow/example_data/nexus/",
        mrbayes_root="/Users/yashaektefaie/Desktop/PhylaFlow/example_data/runs/",
    )
    res_one = dm[0]
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    test()
