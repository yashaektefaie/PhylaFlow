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
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from utils.bhv_utils import (
    BHVEncoder,
    return_sampled_tree_orthant_velocity,
    return_sampled_tree_boundary_decisions,
    return_tree_boundary_merge_paths,
)
import random
from model.treeTokenizer import TreeFeatureTokenizer
from utils.random_tree import Tree
from ete3 import Tree as EteTree
from utils.utils import remove_bit


class SizeDetector:
    def __init__(self, max_aa=None):
        self.max_aa = max_aa

    def update_max_aa(self, new_max_aa):
        self.max_aa = new_max_aa


def resolve_training_target_tree_for_prefix(
    start_tree_newick: str,
    target_tree_newick: str,
    prefix_k: int,
) -> str:
    if int(prefix_k) < 0:
        return target_tree_newick

    boundary_paths = return_tree_boundary_merge_paths(
        start_tree_newick,
        target_tree_newick,
        legacy_training_semantics=False,
    )
    if not boundary_paths:
        return target_tree_newick

    prefix_idx = min(int(prefix_k), len(boundary_paths) - 1)
    return boundary_paths[prefix_idx]["end_newick"]


def resolve_training_target_tree_for_event_prefix(
    start_tree_newick: str,
    target_tree_newick: str,
    event_prefix_count: int,
) -> str:
    event_prefix_count = int(event_prefix_count)
    if event_prefix_count < 0:
        return target_tree_newick
    if event_prefix_count == 0:
        return start_tree_newick

    boundary_paths = return_tree_boundary_merge_paths(
        start_tree_newick,
        target_tree_newick,
        legacy_training_semantics=False,
    )
    if not boundary_paths:
        return target_tree_newick

    remaining_events = event_prefix_count
    current_tree = start_tree_newick
    for path in boundary_paths:
        events = path["events"]
        if remaining_events == 0:
            return current_tree
        if remaining_events < len(events):
            return path["events"][remaining_events]["newick"]
        if remaining_events == len(events):
            return path["end_newick"]
        remaining_events -= len(events)
        current_tree = path["end_newick"]

    return target_tree_newick


def _remap_tree_leaf_names_to_match_reference(
    tree_newick: str,
    reference_tree_newick: str,
) -> str:
    tree = EteTree(tree_newick, format=1)
    reference_tree = EteTree(reference_tree_newick, format=1)

    tree_leaves = sorted((leaf.name for leaf in tree.get_leaves()), key=lambda x: int(x))
    reference_leaves = sorted(
        (leaf.name for leaf in reference_tree.get_leaves()),
        key=lambda x: int(x),
    )

    if tree_leaves == reference_leaves:
        return tree_newick

    if len(tree_leaves) != len(reference_leaves):
        raise ValueError(
            "Cannot remap tree leaf names: leaf counts differ between tree and reference."
        )

    remap = {src: dst for src, dst in zip(tree_leaves, reference_leaves)}
    for leaf in tree.get_leaves():
        if leaf.name not in remap:
            raise ValueError(f"Leaf {leaf.name} missing from remap dictionary.")
        leaf.name = remap[leaf.name]

    return tree.write(format=1)


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
        sanity_check: bool = False,
        random_sanity_check: bool = False,
        overfit_velocity_zero: bool = False,
        overfit_velocity_event_states: bool = False,
        overfit_velocity_orthant_start_states: bool = False,
        overfit_velocity_explicit_boundary_end_states: bool = False,
        overfit_velocity_fixed_timepoints: Optional[List[float]] = None,
        overfit_boundary_prefix_k: int = -1,
        overfit_start_boundary_prefix_k: int = -1,
        overfit_event_prefix_count: int = -1,
        overfit_event_horizon: int = 1,
        overfit_fixed_pair: bool = False,
        overfit_fixed_pair_start_tree_newick: Optional[str] = None,
        overfit_fixed_pair_start_tree_json_path: Optional[str] = None,
        overfit_fixed_pair_start_tree_json_paths: Optional[List[str]] = None,
        overfit_fixed_pair_target_tree_newick: Optional[str] = None,
        overfit_fixed_pair_target_tree_json_path: Optional[str] = None,
        overfit_fixed_pair_target_tree_json_paths: Optional[List[str]] = None,
        overfit_split_multi_subset_events: bool = False,
        overfit_oracle_prefix_start_prob: float = 0.0,
        overfit_oracle_prefix_max_fraction: float = 0.5,
    ) -> None:
        self.nexus_root = nexus_root
        self.mrbayes_root = mrbayes_root
        self.filter_ids = filter_ids
        self.validation = validation
        self.overfit_velocity_zero = overfit_velocity_zero
        self.overfit_velocity_event_states = bool(overfit_velocity_event_states)
        self.overfit_velocity_orthant_start_states = bool(
            overfit_velocity_orthant_start_states
        )
        self.overfit_velocity_explicit_boundary_end_states = bool(
            overfit_velocity_explicit_boundary_end_states
        )
        self.overfit_velocity_fixed_timepoints = (
            [float(t) for t in overfit_velocity_fixed_timepoints]
            if overfit_velocity_fixed_timepoints
            else None
        )
        self.overfit_boundary_prefix_k = int(overfit_boundary_prefix_k)
        self.overfit_start_boundary_prefix_k = int(overfit_start_boundary_prefix_k)
        self.overfit_event_prefix_count = int(overfit_event_prefix_count)
        self.overfit_event_horizon = max(1, int(overfit_event_horizon))
        self.overfit_fixed_pair = bool(overfit_fixed_pair)
        self.overfit_oracle_prefix_start_prob = float(
            overfit_oracle_prefix_start_prob
        )
        self.overfit_oracle_prefix_max_fraction = float(
            overfit_oracle_prefix_max_fraction
        )
        override_start_tree = None
        override_start_tree_bank: List[str] = []
        if overfit_fixed_pair_start_tree_newick:
            override_start_tree = str(overfit_fixed_pair_start_tree_newick)
        elif overfit_fixed_pair_start_tree_json_path:
            override_payload = json.loads(
                Path(overfit_fixed_pair_start_tree_json_path).read_text()
            )
            override_start_tree = str(
                override_payload.get("final_tree")
                or override_payload.get("start_tree")
            )
        if overfit_fixed_pair_start_tree_json_paths:
            for raw_path in overfit_fixed_pair_start_tree_json_paths:
                override_payload = json.loads(Path(raw_path).read_text())
                override_tree = str(
                    override_payload.get("final_tree")
                    or override_payload.get("start_tree")
                )
                if override_tree:
                    override_start_tree_bank.append(override_tree)
        if override_start_tree is not None and override_start_tree not in override_start_tree_bank:
            override_start_tree_bank.append(str(override_start_tree))
        self.overfit_fixed_pair_start_tree_newick = override_start_tree
        self.overfit_fixed_pair_start_tree_newick_bank = list(override_start_tree_bank)
        override_target_tree = None
        override_target_tree_bank: List[str] = []
        if overfit_fixed_pair_target_tree_newick:
            override_target_tree = str(overfit_fixed_pair_target_tree_newick)
        elif overfit_fixed_pair_target_tree_json_path:
            override_payload = json.loads(
                Path(overfit_fixed_pair_target_tree_json_path).read_text()
            )
            override_target_tree = str(
                override_payload.get("target_tree")
                or override_payload.get("final_tree")
                or override_payload.get("start_tree")
            )
        if overfit_fixed_pair_target_tree_json_paths:
            for raw_path in overfit_fixed_pair_target_tree_json_paths:
                override_payload = json.loads(Path(raw_path).read_text())
                override_tree = str(
                    override_payload.get("target_tree")
                    or override_payload.get("final_tree")
                    or override_payload.get("start_tree")
                )
                if override_tree:
                    override_target_tree_bank.append(override_tree)
        if (
            override_target_tree is not None
            and override_target_tree not in override_target_tree_bank
        ):
            override_target_tree_bank.append(str(override_target_tree))
        self.overfit_fixed_pair_target_tree_newick = override_target_tree
        self.overfit_fixed_pair_target_tree_newick_bank = list(override_target_tree_bank)
        self.overfit_split_multi_subset_events = bool(
            overfit_split_multi_subset_events
        )
        self.size_detector = SizeDetector()
        # State tracker for adaptive batching (index, subtree_size, num_subtrees)
        # Default initialization
        self.chosen_tree = (0, 100, 1)
        self.name_to_seq = {}

        # Internal containers
        self._ids: List[str] = []  # populated by build_index()
        self._index: List[Dict[str, Any]] = []  # list of sample metadata dicts
        self._id_to_idx: Dict[str, int] = {}
        self._cached_overfit_pairs: Dict[int, Dict[str, Any]] = {}
        self._cached_overfit_pair_banks: Dict[int, List[Dict[str, Any]]] = {}
        self.random_tree = None
        self.sanity_check = sanity_check
        self.random_sanity_check = random_sanity_check

        if self.sanity_check and self.random_sanity_check:
            raise Exception("Cannot have both sanity_check and random_sanity_check enabled!")

        # Build index immediately; optionally preload
        self.build_index()

    def _normalize_start_tree_bank(self, start_tree_bank: List[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for raw_tree in start_tree_bank:
            tree = str(raw_tree).strip()
            if not tree or tree in seen:
                continue
            seen.add(tree)
            normalized.append(tree)
        return normalized

    def set_overfit_fixed_pair_start_tree_bank(
        self,
        start_tree_bank: List[str],
    ) -> List[str]:
        normalized = self._normalize_start_tree_bank(start_tree_bank)
        self.overfit_fixed_pair_start_tree_newick_bank = list(normalized)
        self.overfit_fixed_pair_start_tree_newick = (
            normalized[0] if normalized else None
        )
        self._cached_overfit_pairs.clear()
        self._cached_overfit_pair_banks.clear()
        return list(self.overfit_fixed_pair_start_tree_newick_bank)

    def _normalize_target_tree_bank(self, target_tree_bank: List[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for raw_tree in target_tree_bank:
            tree = str(raw_tree).strip()
            if not tree or tree in seen:
                continue
            seen.add(tree)
            normalized.append(tree)
        return normalized

    def set_overfit_fixed_pair_target_tree_bank(
        self,
        target_tree_bank: List[str],
    ) -> List[str]:
        normalized = self._normalize_target_tree_bank(target_tree_bank)
        self.overfit_fixed_pair_target_tree_newick_bank = list(normalized)
        self.overfit_fixed_pair_target_tree_newick = (
            normalized[0] if normalized else None
        )
        self._cached_overfit_pairs.clear()
        self._cached_overfit_pair_banks.clear()
        return list(self.overfit_fixed_pair_target_tree_newick_bank)

    def _oracle_prefix_candidates(
        self,
        start_tree_newick: str,
        target_tree_newick: str,
    ) -> List[str]:
        boundary_paths = return_tree_boundary_merge_paths(
            start_tree_newick,
            target_tree_newick,
            legacy_training_semantics=False,
        )
        candidates = [str(path["end_newick"]) for path in boundary_paths[:-1]]
        if not candidates:
            return []
        max_fraction = float(self.overfit_oracle_prefix_max_fraction)
        if 0.0 < max_fraction < 1.0:
            keep = max(1, int(math.ceil(len(candidates) * max_fraction)))
            candidates = candidates[:keep]
        return candidates

    def set_overfit_fixed_pair_best_start_tree(
        self,
        start_tree_newick: str,
        *,
        max_bank_size: int = 2,
        keep_first: bool = True,
    ) -> List[str]:
        candidate = str(start_tree_newick).strip()
        if not candidate:
            return list(self.overfit_fixed_pair_start_tree_newick_bank)

        current_bank = list(self.overfit_fixed_pair_start_tree_newick_bank)
        if not current_bank:
            current_bank = [candidate]
        elif keep_first:
            anchor = current_bank[0]
            new_bank = [anchor]
            if candidate != anchor:
                new_bank.append(candidate)
            if int(max_bank_size) > 0:
                new_bank = new_bank[: max(1, int(max_bank_size))]
            return self.set_overfit_fixed_pair_start_tree_bank(new_bank)
        else:
            current_bank.append(candidate)

        if int(max_bank_size) > 0 and len(current_bank) > int(max_bank_size):
            current_bank = current_bank[-int(max_bank_size) :]
        return self.set_overfit_fixed_pair_start_tree_bank(current_bank)

    def resolve_training_target_tree(
        self,
        start_tree_newick: str,
        target_tree_newick: str,
        base_start_tree_newick: Optional[str] = None,
    ) -> str:
        resolved_target_tree = target_tree_newick
        if (
            self.overfit_start_boundary_prefix_k >= 0
            and self.overfit_boundary_prefix_k >= 0
            and (self.sanity_check or self.random_sanity_check)
        ):
            if base_start_tree_newick is None:
                original_prefix = self.overfit_start_boundary_prefix_k
                self.overfit_start_boundary_prefix_k = -1
                try:
                    base_start_tree = self.sample_random_tree(target_tree_newick)
                finally:
                    self.overfit_start_boundary_prefix_k = original_prefix
            else:
                base_start_tree = base_start_tree_newick
            resolved_target_tree = resolve_training_target_tree_for_prefix(
                base_start_tree,
                target_tree_newick,
                self.overfit_boundary_prefix_k,
            )
        else:
            resolved_target_tree = resolve_training_target_tree_for_prefix(
                start_tree_newick,
                target_tree_newick,
                self.overfit_boundary_prefix_k,
            )

        resolved_target_tree = resolve_training_target_tree_for_event_prefix(
            start_tree_newick,
            resolved_target_tree,
            self.overfit_event_prefix_count,
        )
        return _remap_tree_leaf_names_to_match_reference(
            resolved_target_tree,
            start_tree_newick,
        )

    def sample_random_tree_with_base(
        self,
        real_tree,
        subtree_size: Optional[int] = None,
    ) -> Tuple[str, str]:
        if self.overfit_start_boundary_prefix_k < 0:
            start_tree = self.sample_random_tree(real_tree, subtree_size=subtree_size)
            return start_tree, start_tree

        original_prefix = self.overfit_start_boundary_prefix_k
        self.overfit_start_boundary_prefix_k = -1
        try:
            base_random_tree = self.sample_random_tree(real_tree, subtree_size=subtree_size)
        finally:
            self.overfit_start_boundary_prefix_k = original_prefix

        target_tree_newick = real_tree if isinstance(real_tree, str) else real_tree.write(format=1)
        start_tree = resolve_training_target_tree_for_prefix(
            base_random_tree,
            target_tree_newick,
            original_prefix,
        )
        return base_random_tree, start_tree

    def __len__(self) -> int:  # Required for torch Dataset
        return len(self._ids)

    def return_number_leaves(self, index: int) -> int:
        """Return number of leaves in the alignment for the given index."""
        if type(index) == str:
            index = self._id_to_idx[index]
        meta = self._index[index]
        seqs, _ = self.parse_nexus(meta["nexus_path"])
        return len(seqs)

    def return_posterior_trees(self, index: int) -> List[str]:
        """Return list of posterior Newick trees for the given index.

        Applies burn-in and thinning as per load_posterior_trees_from_tfiles.
        """
        if type(index) == str:
            index = self._id_to_idx[index] 
        meta = self._index[index]
        tree_paths = meta["tree_paths"]
        trees = self.load_posterior_trees_from_tfiles(tree_paths)
        if self.sanity_check or self.random_sanity_check:
            return ['((52:6.821929e-03,((((2:4.398080e-02,(((((145:8.657433e-03,91:4.622826e-03):2.222114e-02,((93:1.284674e-02,132:1.985680e-02):8.439914e-03,(((89:1.020633e-02,88:4.611548e-03):8.036501e-03,90:1.429933e-02):1.439583e-02,92:9.908956e-03):1.750425e-03):5.724766e-03):7.037225e-03,87:1.626403e-02):4.747258e-03,(((7:9.739291e-03,5:5.587849e-03):6.613494e-03,(150:1.664500e-02,152:1.724125e-02):1.823357e-02):1.534167e-03,(11:9.469409e-03,(61:5.838223e-04,8:8.467112e-03):4.839481e-03):1.478233e-02):5.742910e-03):1.842902e-02,9:3.628046e-02):4.929418e-03):4.712541e-03,82:4.185746e-02):3.380475e-03,(((((124:1.558528e-02,125:1.705412e-02):5.917463e-03,((102:2.129084e-02,155:9.638513e-03):4.271744e-03,(149:2.436750e-02,(106:2.722806e-02,(130:3.687226e-02,114:4.149299e-02):1.000061e-02):4.023099e-03):2.147868e-03):1.233641e-02):3.719823e-03,((((54:1.214695e-02,19:2.168458e-02):1.996561e-02,((((38:1.057892e-03,((36:7.753757e-04,(46:2.622027e-03,24:1.787443e-03):2.029430e-04):1.481560e-03,(58:1.714262e-04,37:2.458943e-04):2.601513e-03):1.274226e-04):2.389942e-03,56:1.949454e-03):1.489446e-02,(48:1.137571e-02,108:1.964277e-02):1.324782e-03):7.644887e-03,(((39:1.804400e-03,50:3.879771e-04):3.067958e-03,59:3.259752e-03):6.995533e-03,(((6:1.574761e-04,42:2.710545e-05):4.343268e-04,(33:2.616433e-03,41:6.108494e-06):1.184825e-03):1.058122e-02,(((14:6.407019e-05,35:1.283827e-03):3.152093e-03,40:3.500127e-03):1.097830e-02,((((107:1.333270e-03,84:3.209979e-04):2.963766e-03,43:1.910537e-03):1.115617e-03,31:2.194031e-03):1.856883e-03,((47:2.408071e-03,(109:3.641932e-04,49:1.155934e-04):3.415181e-03):2.683443e-03,34:2.281842e-03):4.028540e-03):3.179844e-03):2.139192e-03):1.070869e-03):2.343303e-03):8.589032e-03):3.753159e-03,(((((((136:1.054858e-03,45:1.777442e-04):9.470442e-04,18:8.069737e-04):2.680686e-03,51:1.513089e-03):1.472587e-02,(20:1.791231e-02,((17:6.050183e-03,(((53:2.324964e-03,((100:9.971849e-04,85:1.361781e-03):1.047096e-03,131:4.906427e-03):7.243432e-04):1.869450e-03,62:6.183967e-03):2.898555e-03,(55:6.269062e-03,(13:5.770393e-03,97:4.514650e-03):3.446166e-03):3.282607e-03):1.379633e-03):2.240766e-02,((120:1.168439e-03,25:4.536840e-03):2.917149e-03,(((((63:3.576971e-04,((104:6.782281e-04,105:8.341392e-05):2.542285e-03,(103:2.267958e-03,148:4.839063e-05):2.790594e-04):1.147993e-03):3.031773e-04,29:1.385090e-03):2.901980e-04,(28:1.255159e-03,30:8.757002e-04):1.490579e-03):1.139399e-03,26:2.656695e-03):5.694807e-04,116:5.345046e-03):3.441066e-03):2.961141e-03):5.676048e-03):6.805834e-03):5.342581e-03,(((22:5.519097e-04,21:5.260546e-04):1.063093e-03,23:1.409785e-03):2.054762e-02,99:1.438413e-02):1.228856e-02):1.107615e-03,(((95:9.283287e-03,98:1.739668e-02):6.505733e-03,((16:1.866647e-03,(153:9.563353e-04,15:6.850920e-04):5.016234e-04):6.647760e-03,60:6.967831e-03):1.297224e-02):6.103056e-04,((111:1.652701e-02,(113:1.019696e-02,(118:1.794353e-02,112:9.963961e-03):5.803295e-03):3.456559e-03):1.124700e-02,(126:1.994845e-02,((86:1.119997e-02,(((135:4.125296e-03,123:2.609975e-03):1.338629e-03,122:5.599224e-04):1.247347e-04,121:1.956632e-03):4.255348e-03):6.514329e-03,44:8.702270e-03):1.406738e-03):3.081689e-03):4.390131e-03):6.593693e-03):2.821015e-04,(101:4.858902e-03,12:4.198135e-03):1.335322e-02):4.585962e-04):2.756843e-03,((144:4.292494e-02,((143:8.977315e-03,142:8.391560e-03):7.271121e-02,(140:1.547496e-02,(137:2.375343e-02,(141:1.801854e-02,(139:4.845625e-03,138:7.509111e-03):5.295656e-03):2.358940e-03):1.229887e-02):2.599447e-02):4.443259e-02):7.088933e-03,134:5.769189e-02):6.826836e-03):5.876646e-04):4.031933e-03,(32:3.250200e-02,((117:3.522599e-02,(151:7.473737e-03,110:9.434156e-03):5.396982e-03):7.039427e-03,(27:2.809174e-02,154:2.327384e-02):4.710086e-03):2.332033e-03):3.477086e-03):1.630971e-03,((((127:5.961962e-03,57:2.631306e-03):6.478147e-04,((96:1.409200e-03,115:2.740476e-03):7.137445e-03,133:7.424993e-03):1.159072e-03):6.523926e-03,(4:5.405740e-03,(((((81:6.815847e-04,68:1.155674e-03):2.735598e-03,69:7.025004e-04):8.998414e-04,80:2.236265e-03):1.839654e-03,(((3:4.236887e-03,79:1.677530e-03):3.770879e-04,78:1.062032e-03):2.312128e-03,((77:1.334890e-03,(66:2.301659e-04,((70:8.259407e-05,(75:2.060793e-03,65:3.982049e-03):3.647037e-04):4.418750e-04,(71:1.999872e-03,(72:7.517143e-04,73:6.105537e-04):8.489613e-05):5.596686e-04):9.888103e-04):1.665829e-03):7.810491e-04,(64:1.400547e-03,76:2.738529e-03):7.529917e-04):1.655366e-03):3.548366e-03):4.212272e-04,(67:1.216874e-03,74:1.827134e-03):2.430697e-03):1.433684e-03):2.358579e-03):1.337100e-02,((128:1.130754e-02,129:1.857543e-02):2.664069e-02,10:2.449606e-02):1.431688e-02):1.932062e-03):3.746715e-03):1.785518e-03,((83:4.168728e-02,119:4.097966e-02):7.403229e-03,(146:1.888170e-02,147:1.810523e-02):1.032872e-02):2.647861e-02):2.939351e-02):4.262485e-03,94:2.756089e-04,1:6.820178e-04);']
        return trees

    def return_nexus_filepath(self, index: int) -> str:
        """Return the Nexus file path for the given index."""
        if type(index) == str:
            index = self._id_to_idx[index]
        meta = self._index[index]
        return meta["nexus_path"]

    def return_nexus_number_to_name(self, index: int) -> Dict[int, str]:
        """Return mapping from taxon number to name for the given index."""
        if type(index) == str:
            index = self._id_to_idx[index]
        meta = self._index[index]
        _, taxa_order = self.parse_nexus(meta["nexus_path"])
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
                "nexus_path": meta["nexus_path"],
                "tree_paths": meta["tree_paths"],
                "num_to_name": self.return_nexus_number_to_name(index),
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

        #######VERY IMPORTANT HERE FOR DEBUG PURPOSES WE WILL ALWAYS SAMPLE THE FIRST TREE########
        real_tree_newick = random.sample(trees, 1)[0]
        # real_tree_newick = trees[0]
        #########################################################################################

        t = EteTree(real_tree_newick, format=1)
        leaves = t.get_leaves()

        # Pruning logic for adaptive batching

        if preset_subtree_size is not None and len(leaves) > preset_subtree_size:
            kept_leaves = random.sample(leaves, preset_subtree_size)
            t.prune(kept_leaves, preserve_branch_length=True)
            # real_tree_newick = t.write(format=1) # Don't write yet, wait for re-indexing
            # Update leaves for size tracking
            leaves = t.get_leaves()

        real_tree_original_label_newick = t.write(format=1)

        current_size = len(leaves)
        self.chosen_tree = (index, current_size, 1)  # (index, size, num_subtrees)

        # Normalize tree indices to 0..N-1 and subset sequences
        # Sort leaves for deterministic indexing
        leaves.sort(key=lambda x: x.name)

        new_seqs = {}
        original_names_map = {}
        seq_ordering_map = {}

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

            seq_ordering_map[original_node_name] = new_idx_str

        # Serialize the normalized tree
        real_tree_newick = t.write(format=1)

        # Re-parse purely to ensure we are passing consistent objects
        # (Though prune modifies in-place, let's keep it safe)
        t_pruned = EteTree(real_tree_newick, format=1)
        def _remap_random_tree_to_dataset_indexing(random_tree_newick: str) -> str:
            t_random = EteTree(random_tree_newick, format=1)

            # Now remap the random tree to make the indices match up with the real tree.
            if self.sanity_check:
                for leaf in t_random.get_leaves():
                    name = leaf.name
                    if name in seq_ordering_map:
                        leaf.name = seq_ordering_map[name]
                    else:
                        raise Exception(
                            "Leaf name in random tree not found in original names map!"
                        )
            else:
                for leaf in t_random.get_leaves():
                    name = str(int(leaf.name))
                    if name in seq_ordering_map:
                        leaf.name = seq_ordering_map[name]
                    else:
                        raise Exception(
                            "Leaf name in random tree not found in original names map!"
                        )
            return t_random.write(format=1)

        sample_source_tree = t_pruned
        if self.overfit_start_boundary_prefix_k >= 0 and (
            self.sanity_check or self.random_sanity_check
        ):
            # Keep start/target prefix resolution in the original leaf-name space,
            # then remap both together into dataset indexing.
            sample_source_tree = real_tree_original_label_newick

        def _build_pair(
            forced_start_tree_newick: Optional[str] = None,
            forced_target_tree_newick: Optional[str] = None,
        ) -> Dict[str, Any]:
            chosen_start_tree_newick = (
                str(forced_start_tree_newick)
                if forced_start_tree_newick is not None
                else None
            )
            if chosen_start_tree_newick is None and self.overfit_fixed_pair_start_tree_newick:
                chosen_start_tree_newick = str(self.overfit_fixed_pair_start_tree_newick)

            if chosen_start_tree_newick is not None:
                random_tree = chosen_start_tree_newick
                base_random_tree = random_tree
            else:
                base_random_tree_raw, random_tree_raw = self.sample_random_tree_with_base(
                    sample_source_tree
                )
                base_random_tree = _remap_random_tree_to_dataset_indexing(
                    base_random_tree_raw
                )
                random_tree = _remap_random_tree_to_dataset_indexing(random_tree_raw)

            target_tree_newick = (
                str(forced_target_tree_newick)
                if forced_target_tree_newick is not None
                else real_tree_newick
            )
            # Both trees now use "0".."N-1" names, so bhv utils will work happily
            effective_target_tree = self.resolve_training_target_tree(
                random_tree,
                target_tree_newick,
                base_start_tree_newick=base_random_tree,
            )
            boundary_paths = return_tree_boundary_merge_paths(
                random_tree,
                effective_target_tree,
                legacy_training_semantics=False,
            )
            final_labels = return_sampled_tree_boundary_decisions(
                random_tree,
                effective_target_tree,
                split_multi_label_events=self.overfit_split_multi_subset_events,
                legacy_training_semantics=False,
            )

            # If final_labels is empty, resample random tree until we get valid labels
            while not final_labels:
                if chosen_start_tree_newick is not None:
                    raise ValueError(
                        "Configured overfit_fixed_pair_start_tree_newick did not "
                        "yield any valid boundary decisions."
                    )
                base_random_tree_raw, random_tree_raw = self.sample_random_tree_with_base(
                    sample_source_tree
                )
                base_random_tree = _remap_random_tree_to_dataset_indexing(
                    base_random_tree_raw
                )
                random_tree = _remap_random_tree_to_dataset_indexing(random_tree_raw)
                target_tree_newick = (
                    str(forced_target_tree_newick)
                    if forced_target_tree_newick is not None
                    else real_tree_newick
                )
                effective_target_tree = self.resolve_training_target_tree(
                    random_tree,
                    target_tree_newick,
                    base_start_tree_newick=base_random_tree,
                )
                boundary_paths = return_tree_boundary_merge_paths(
                    random_tree,
                    effective_target_tree,
                    legacy_training_semantics=False,
                )
                final_labels = return_sampled_tree_boundary_decisions(
                    random_tree,
                    effective_target_tree,
                    split_multi_label_events=self.overfit_split_multi_subset_events,
                    legacy_training_semantics=False,
                )

            return {
                "base_random_tree": base_random_tree,
                "random_tree": random_tree,
                "effective_target_tree": effective_target_tree,
                "boundary_paths": boundary_paths,
                "final_labels": final_labels,
            }

        if self.overfit_fixed_pair:
            if (
                not self.validation
                and self.overfit_oracle_prefix_start_prob > 0.0
                and random.random() < self.overfit_oracle_prefix_start_prob
            ):
                start_bank = list(self.overfit_fixed_pair_start_tree_newick_bank or [])
                target_bank = list(
                    self.overfit_fixed_pair_target_tree_newick_bank or []
                )
                if start_bank and target_bank:
                    chosen_start_tree = random.choice(start_bank)
                    chosen_target_tree = random.choice(target_bank)
                    oracle_prefix_candidates = self._oracle_prefix_candidates(
                        chosen_start_tree,
                        chosen_target_tree,
                    )
                    if oracle_prefix_candidates:
                        oracle_start_tree = random.choice(oracle_prefix_candidates)
                        pair = _build_pair(
                            forced_start_tree_newick=oracle_start_tree,
                            forced_target_tree_newick=chosen_target_tree,
                        )
                        pair["oracle_prefix_start_tree"] = str(oracle_start_tree)
                        pair["oracle_prefix_base_start_tree"] = str(
                            chosen_start_tree
                        )
                        pair["oracle_prefix_target_tree"] = str(chosen_target_tree)
                    else:
                        pair = None
                else:
                    pair = None
            else:
                pair = None

            if pair is None and len(self.overfit_fixed_pair_target_tree_newick_bank) > 1:
                forced_start_tree_newick = (
                    random.choice(self.overfit_fixed_pair_start_tree_newick_bank)
                    if self.overfit_fixed_pair_start_tree_newick_bank
                    else None
                )
                forced_target_tree_newick = random.choice(
                    self.overfit_fixed_pair_target_tree_newick_bank
                )
                pair = _build_pair(
                    forced_start_tree_newick=forced_start_tree_newick,
                    forced_target_tree_newick=forced_target_tree_newick,
                )
            elif pair is None and len(self.overfit_fixed_pair_start_tree_newick_bank) > 1:
                pair_bank = self._cached_overfit_pair_banks.get(index)
                if pair_bank is None:
                    random_state = random.getstate()
                    try:
                        random.seed(13)
                        pair_bank = [
                            _build_pair(forced_start_tree_newick=start_tree_newick)
                            for start_tree_newick in self.overfit_fixed_pair_start_tree_newick_bank
                        ]
                    finally:
                        random.setstate(random_state)
                    self._cached_overfit_pair_banks[index] = pair_bank
                pair = random.choice(pair_bank)
            elif pair is None:
                pair = self._cached_overfit_pairs.get(index)
                if pair is None:
                    random_state = random.getstate()
                    try:
                        random.seed(13)
                        pair = _build_pair()
                    finally:
                        random.setstate(random_state)
                    self._cached_overfit_pairs[index] = pair
        else:
            pair = _build_pair()

        base_random_tree = pair["base_random_tree"]
        random_tree = pair["random_tree"]
        effective_target_tree = pair["effective_target_tree"]
        boundary_paths = pair["boundary_paths"]
        final_labels = pair["final_labels"]

        horizon = min(self.overfit_event_horizon, len(final_labels))
        max_start_index = max(0, len(final_labels) - horizon)
        random_index = random.randint(0, max_start_index)

        def _build_step_sample(event_index: int) -> Dict[str, Any]:
            chosen_autoregressive_event = final_labels[event_index]
            autoregressive_time = (
                0.0
                if len(final_labels) <= 1
                else event_index / float(len(final_labels) - 1)
            )
            velocity_next_boundary_tree = None
            if self.overfit_velocity_explicit_boundary_end_states:
                explicit_velocity_trees = [random_tree]
                explicit_velocity_trees.extend(
                    path["end_newick"] for path in boundary_paths[:-1]
                )
                if self.overfit_velocity_fixed_timepoints:
                    explicit_velocity_timepoints = list(
                        self.overfit_velocity_fixed_timepoints
                    )
                    if len(explicit_velocity_timepoints) == 1:
                        explicit_velocity_timepoints = explicit_velocity_timepoints * len(
                            explicit_velocity_trees
                        )
                else:
                    explicit_velocity_timepoints = [0.0]
                    explicit_velocity_timepoints.extend(
                        float(path["global_time"]) for path in boundary_paths[:-1]
                    )
                if len(explicit_velocity_trees) != len(explicit_velocity_timepoints):
                    raise ValueError(
                        "Explicit boundary-end velocity supervision requires one "
                        "global timepoint per orthant-start state. "
                        f"Got {len(explicit_velocity_timepoints)} "
                        f"timepoints for {len(explicit_velocity_trees)} states."
                    )

                explicit_velocity_options = list(
                    zip(
                        explicit_velocity_trees,
                        [path["start_newick"] for path in boundary_paths],
                        explicit_velocity_timepoints,
                    )
                )
                (
                    velocity_source_tree,
                    velocity_next_boundary_tree,
                    model_timepoint,
                ) = random.choice(
                    explicit_velocity_options
                )
                newick, velocity = return_sampled_tree_orthant_velocity(
                    velocity_source_tree,
                    effective_target_tree,
                    0.0,
                    legacy_training_semantics=False,
                )
                timepoint = float(model_timepoint)
            elif self.overfit_velocity_orthant_start_states:
                orthant_start_trees = [random_tree]
                orthant_start_trees.extend(
                    path["end_newick"] for path in boundary_paths[:-1]
                )
                next_boundary_trees = [
                    path.get("start_newick", path["end_newick"])
                    for path in boundary_paths
                ]
                if all("start_newick" in path for path in boundary_paths):
                    (
                        velocity_source_tree,
                        velocity_next_boundary_tree,
                    ) = random.choice(
                        list(zip(orthant_start_trees, next_boundary_trees))
                    )
                else:
                    velocity_source_tree = random.choice(orthant_start_trees)
                    source_index = orthant_start_trees.index(velocity_source_tree)
                    velocity_next_boundary_tree = next_boundary_trees[source_index]
                timepoint = 0.0
                newick, velocity = return_sampled_tree_orthant_velocity(
                    velocity_source_tree,
                    effective_target_tree,
                    timepoint,
                    legacy_training_semantics=False,
                )
            elif self.overfit_velocity_event_states:
                velocity_source_tree = chosen_autoregressive_event["newick"]
                timepoint = 0.0
                newick, velocity = return_sampled_tree_orthant_velocity(
                    velocity_source_tree,
                    effective_target_tree,
                    timepoint,
                    legacy_training_semantics=False,
                )
            else:
                velocity_source_tree = random_tree
                if self.overfit_velocity_fixed_timepoints is not None:
                    timepoint = float(random.choice(self.overfit_velocity_fixed_timepoints))
                elif self.overfit_velocity_zero:
                    timepoint = 0.0
                else:
                    timepoint = random.uniform(0, 1)
                newick, velocity = return_sampled_tree_orthant_velocity(
                    velocity_source_tree,
                    effective_target_tree,
                    timepoint,
                    legacy_training_semantics=False,
                )

            return {
                "id": meta["id"],
                "nexus_path": meta["nexus_path"],
                "tree_paths": meta["tree_paths"],
                "sequences": new_seqs,
                "taxa_order": list(new_seqs.keys()),
                "start_tree": random_tree,
                "newick_tree": newick,
                "target_tree": effective_target_tree,
                "fixed_pair_num_events": int(len(final_labels)),
                "velocity": velocity,
                "velocity_next_boundary_tree": velocity_next_boundary_tree,
                "timepoint": timepoint,
                "autoregressive_newick": chosen_autoregressive_event["newick"],
                "autoregressive_labels": chosen_autoregressive_event["labels"],
                "autoregressive_stop_after_merge": bool(
                    chosen_autoregressive_event.get("stop_after_merge", False)
                ),
                "autoregressive_event_index": int(event_index),
                "autoregressive_newick_time": autoregressive_time,
                "num_to_name": original_names_map,
                "seq_ordering_map": seq_ordering_map,
            }

        step_samples = [
            _build_step_sample(event_index)
            for event_index in range(random_index, random_index + horizon)
        ]

        num_to_name = self.return_nexus_number_to_name(index)
        sample = dict(step_samples[0])
        sample["num_to_name"] = original_names_map
        if "oracle_prefix_start_tree" in pair:
            sample["oracle_prefix_start_tree"] = pair["oracle_prefix_start_tree"]
            sample["oracle_prefix_base_start_tree"] = pair[
                "oracle_prefix_base_start_tree"
            ]
            sample["oracle_prefix_target_tree"] = pair["oracle_prefix_target_tree"]
        sample["seq_ordering_map"] = seq_ordering_map
        if len(step_samples) > 1:
            sample["multi_step_samples"] = step_samples

        return sample

    def get_overfit_fixed_pair(self, index: int) -> Optional[Dict[str, Any]]:
        if not self.overfit_fixed_pair:
            return None
        if len(self.overfit_fixed_pair_target_tree_newick_bank) > 1:
            return None
        if len(self.overfit_fixed_pair_start_tree_newick_bank) > 1:
            if index not in self._cached_overfit_pair_banks:
                _ = self[index]
            pair_bank = self._cached_overfit_pair_banks.get(index)
            if pair_bank:
                return pair_bank[0]
            return None
        if index not in self._cached_overfit_pairs:
            _ = self[index]
        return self._cached_overfit_pairs.get(index)

    def sample_overfit_fixed_pair_bank_pair(self) -> Optional[Dict[str, Any]]:
        if not self.overfit_fixed_pair:
            return None
        start_bank = list(self.overfit_fixed_pair_start_tree_newick_bank or [])
        target_bank = list(self.overfit_fixed_pair_target_tree_newick_bank or [])
        if not start_bank or not target_bank:
            return None

        chosen_start_tree = random.choice(start_bank)
        chosen_target_tree = (
            random.choice(target_bank) if len(target_bank) > 1 else target_bank[0]
        )
        base_random_tree = str(chosen_start_tree)
        random_tree = str(chosen_start_tree)
        effective_target_tree = self.resolve_training_target_tree(
            random_tree,
            str(chosen_target_tree),
            base_start_tree_newick=base_random_tree,
        )
        boundary_paths = return_tree_boundary_merge_paths(
            random_tree,
            effective_target_tree,
            legacy_training_semantics=False,
        )
        final_labels = return_sampled_tree_boundary_decisions(
            random_tree,
            effective_target_tree,
            split_multi_label_events=self.overfit_split_multi_subset_events,
            legacy_training_semantics=False,
        )
        return {
            "base_random_tree": base_random_tree,
            "random_tree": random_tree,
            "effective_target_tree": effective_target_tree,
            "boundary_paths": boundary_paths,
            "final_labels": final_labels,
        }

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
        if self.overfit_start_boundary_prefix_k >= 0:
            original_prefix = self.overfit_start_boundary_prefix_k
            self.overfit_start_boundary_prefix_k = -1
            try:
                base_random_tree = self.sample_random_tree(real_tree, subtree_size=subtree_size)
            finally:
                self.overfit_start_boundary_prefix_k = original_prefix
            target_tree_newick = real_tree if isinstance(real_tree, str) else real_tree.write(format=1)
            return resolve_training_target_tree_for_prefix(
                base_random_tree,
                target_tree_newick,
                original_prefix,
            )

        ###DEBUG PURPOSES ONLY RETURN THE SAME TREE###
        # if self.random_tree is not None:
        #     return self.random_tree

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
        # if self.random_tree is None:
        #     self.random_tree = random_newick
        if self.sanity_check:
            return '((52:6.821929e-03,((((2:4.398080e-02,(((((145:8.657433e-03,91:4.622826e-03):2.222114e-02,((93:1.284674e-02,132:1.985680e-02):8.439914e-03,(((89:1.020633e-02,88:4.611548e-03):8.036501e-03,90:1.429933e-02):1.439583e-02,92:9.908956e-03):1.750425e-03):5.724766e-03):7.037225e-03,87:1.626403e-02):4.747258e-03,(((7:9.739291e-03,5:5.587849e-03):6.613494e-03,(150:1.664500e-02,152:1.724125e-02):1.823357e-02):1.534167e-03,(11:9.469409e-03,(61:5.838223e-04,8:8.467112e-03):4.839481e-03):1.478233e-02):5.742910e-03):1.842902e-02,9:3.628046e-02):4.929418e-03):4.712541e-03,82:4.185746e-02):3.380475e-03,(((((124:1.558528e-02,125:1.705412e-02):5.917463e-03,((102:2.129084e-02,155:9.638513e-03):4.271744e-03,(149:2.436750e-02,(106:2.722806e-02,(130:3.687226e-02,114:4.149299e-02):1.000061e-02):4.023099e-03):2.147868e-03):1.233641e-02):3.719823e-03,((((54:1.214695e-02,19:2.168458e-02):1.996561e-02,((((38:1.057892e-03,((36:7.753757e-04,(46:2.622027e-03,24:1.787443e-03):2.029430e-04):1.481560e-03,(58:1.714262e-04,37:2.458943e-04):2.601513e-03):1.274226e-04):2.389942e-03,56:1.949454e-03):1.489446e-02,(48:1.137571e-02,108:1.964277e-02):1.324782e-03):7.644887e-03,(((39:1.804400e-03,50:3.879771e-04):3.067958e-03,59:3.259752e-03):6.995533e-03,(((6:1.574761e-04,42:2.710545e-05):4.343268e-04,(33:2.616433e-03,41:6.108494e-06):1.184825e-03):1.058122e-02,(((14:6.407019e-05,35:1.283827e-03):3.152093e-03,40:3.500127e-03):1.097830e-02,((((107:1.333270e-03,84:3.209979e-04):2.963766e-03,43:1.910537e-03):1.115617e-03,31:2.194031e-03):1.856883e-03,((47:2.408071e-03,(109:3.641932e-04,49:1.155934e-04):3.415181e-03):2.683443e-03,34:2.281842e-03):4.028540e-03):3.179844e-03):2.139192e-03):1.070869e-03):2.343303e-03):8.589032e-03):3.753159e-03,(((((((136:1.054858e-03,51:1.777442e-04):9.470442e-04,18:8.069737e-04):2.680686e-03,45:1.513089e-03):1.472587e-02,(20:1.791231e-02,((17:6.050183e-03,(((53:2.324964e-03,((100:9.971849e-04,85:1.361781e-03):1.047096e-03,131:4.906427e-03):7.243432e-04):1.869450e-03,62:6.183967e-03):2.898555e-03,(55:6.269062e-03,(13:5.770393e-03,97:4.514650e-03):3.446166e-03):3.282607e-03):1.379633e-03):2.240766e-02,((120:1.168439e-03,25:4.536840e-03):2.917149e-03,(((((63:3.576971e-04,((104:6.782281e-04,105:8.341392e-05):2.542285e-03,(103:2.267958e-03,148:4.839063e-05):2.790594e-04):1.147993e-03):3.031773e-04,29:1.385090e-03):2.901980e-04,(28:1.255159e-03,30:8.757002e-04):1.490579e-03):1.139399e-03,26:2.656695e-03):5.694807e-04,116:5.345046e-03):3.441066e-03):2.961141e-03):5.676048e-03):6.805834e-03):5.342581e-03,(((22:5.519097e-04,21:5.260546e-04):1.063093e-03,23:1.409785e-03):2.054762e-02,99:1.438413e-02):1.228856e-02):1.107615e-03,(((95:9.283287e-03,98:1.739668e-02):6.505733e-03,((16:1.866647e-03,(153:9.563353e-04,15:6.850920e-04):5.016234e-04):6.647760e-03,60:6.967831e-03):1.297224e-02):6.103056e-04,((111:1.652701e-02,(113:1.019696e-02,(118:1.794353e-02,112:9.963961e-03):5.803295e-03):3.456559e-03):1.124700e-02,(126:1.994845e-02,((86:1.119997e-02,(((135:4.125296e-03,123:2.609975e-03):1.338629e-03,122:5.599224e-04):1.247347e-04,121:1.956632e-03):4.255348e-03):6.514329e-03,44:8.702270e-03):1.406738e-03):3.081689e-03):4.390131e-03):6.593693e-03):2.821015e-04,(101:4.858902e-03,12:4.198135e-03):1.335322e-02):4.585962e-04):2.756843e-03,((144:4.292494e-02,((143:8.977315e-03,142:8.391560e-03):7.271121e-02,(140:1.547496e-02,(137:2.375343e-02,(141:1.801854e-02,(139:4.845625e-03,138:7.509111e-03):5.295656e-03):2.358940e-03):1.229887e-02):2.599447e-02):4.443259e-02):7.088933e-03,134:5.769189e-02):6.826836e-03):5.876646e-04):4.031933e-03,(32:3.250200e-02,((117:3.522599e-02,(151:7.473737e-03,110:9.434156e-03):5.396982e-03):7.039427e-03,(27:2.809174e-02,154:2.327384e-02):4.710086e-03):2.332033e-03):3.477086e-03):1.630971e-03,((((127:5.961962e-03,57:2.631306e-03):6.478147e-04,((96:1.409200e-03,115:2.740476e-03):7.137445e-03,133:7.424993e-03):1.159072e-03):6.523926e-03,(4:5.405740e-03,(((((81:6.815847e-04,68:1.155674e-03):2.735598e-03,69:7.025004e-04):8.998414e-04,80:2.236265e-03):1.839654e-03,(((3:4.236887e-03,79:1.677530e-03):3.770879e-04,78:1.062032e-03):2.312128e-03,((77:1.334890e-03,(66:2.301659e-04,((70:8.259407e-05,(75:2.060793e-03,65:3.982049e-03):3.647037e-04):4.418750e-04,(71:1.999872e-03,(72:7.517143e-04,73:6.105537e-04):8.489613e-05):5.596686e-04):9.888103e-04):1.665829e-03):7.810491e-04,(64:1.400547e-03,76:2.738529e-03):7.529917e-04):1.655366e-03):3.548366e-03):4.212272e-04,(67:1.216874e-03,74:1.827134e-03):2.430697e-03):1.433684e-03):2.358579e-03):1.337100e-02,((128:1.130754e-02,129:1.857543e-02):2.664069e-02,10:2.449606e-02):1.431688e-02):1.932062e-03):3.746715e-03):1.785518e-03,((83:4.168728e-02,119:4.097966e-02):7.403229e-03,(146:1.888170e-02,147:1.810523e-02):1.032872e-02):2.647861e-02):2.939351e-02):4.262485e-03,94:2.756089e-04,1:6.820178e-04);'
        elif self.random_sanity_check:
            # Return a fixed random tree for sanity checking
            #return '((((((((((((((((((((114:0.10647175508658419,138:0.16166919943341312):0.16066029529927675,((((116:0.5627405078277302,128:0.9121985789594937):0.24817340926137418,(31:0.7138600194795188,70:0.27733161619156094):0.4220283090253836):0.13590539030009635,45:0.9266067111065456):0.41469258200132686,97:0.9567996962072277):0.3973856401724488):0.12827166224771633,93:0.3554077644937247):0.4654676518823466,82:0.6121947293792284):0.15119162175581607,136:0.5780135217434872):0.681561776165486,((((121:0.8205332597755289,26:0.4976519606747656):0.7351007020839916,50:0.16247867919653797):0.26513923220851,(((((21:0.6708560025782748,61:0.985812264384931):0.1457190042962776,123:0.887037358663522):0.16191218265401736,148:0.41410795146806356):0.2177904539147108,((144:0.9361164985349627,69:0.718171432223074):0.2006864925304015,27:0.6048161782972717):0.6809507099538845):0.5115021153134068,57:0.622530760364837):0.5163652615608707):0.22616406738529704,(((32:0.4183107711531565,56:0.3350936767511892):0.9331874403126009,35:0.7745298500291854):0.3130635990064228,67:0.5679283137099269):0.7002400695719887):0.1545514066693123):0.7553922295970644,83:0.8503526523736518):0.7311503693259214,41:0.24327805868265814):0.15263496280656946,133:0.29593080857275744):0.7621634609101391,(77:0.3935646143716599,9:0.5600565971871284):0.7002036136951567):0.24542377864649756,25:0.9818181584459523):0.8756372817731165,139:0.16116477703818152):0.5985293701820318,113:0.118930834006353):0.6295961029693159,154:0.33801932933841494):0.8941447676266638,(((118:0.3186975181561521,38:0.2614245970763863):0.9173431415967147,20:0.5548681522881885):0.32600912686290473,23:0.13967808561914502):0.6185904847488098):0.3231150522994072,((((((((((((140:0.17378506494538792,142:0.7027533772928463):0.6210271517564896,(((103:0.20068095278880554,22:0.8464132804349973):0.27243106492748864,122:0.1983421290203236):0.58265717851931,84:0.3734874543415184):0.13576749606015798):0.5749064086764286,(109:0.9679266154067097,63:0.9095542596200796):0.645934525734366):0.3167844332025098,((24:0.21773769463153203,59:0.6369141294889702):0.4002008847445794,105:0.758716835360165):0.1708813434622295):0.6106212493200189,19:0.4085922590139328):0.6390501260142061,((((((((102:0.9301906725564473,53:0.7115663065242372):0.6930844745631968,(((12:0.4699920484447643,129:0.1019398405162838):0.778208953617708,126:0.18393649856545236):0.3127415414875745,((78:0.5989712299001143,79:0.27265608369113276):0.6984364907914944,94:0.6974684526744989):0.7274875727749591):0.41886780993059847):0.35062953756809234,((117:0.7510173092928738,66:0.4924517639094751):0.4586193807083052,49:0.6940298644737661):0.5853105382552791):0.5368730492354417,48:0.30063119248522807):0.9581497728881525,1:0.7477126019973318):0.24539413217783806,(((55:0.8893345704180042,73:0.6970487286389898):0.1248404816076723,15:0.3722654286810009):0.5888961660354282,145:0.10838391119582176):0.942160858700341):0.3822443287927564,134:0.15486181432886426):0.14512810353755823,33:0.6038951672294827):0.5397451372260531):0.926644144956132,11:0.9360221437019591):0.24234197891020276,(((((((((104:0.3155070793042529,81:0.25662460888894956):0.3535806443121049,17:0.7681750155828988):0.23990311879749143,18:0.28956668932624874):0.7574547856739193,((131:0.6956841807556707,72:0.6608398689517683):0.416031516174477,51:0.7959939736963249):0.8050253756358585):0.986789241049431,((((((((37:0.47106496931841757,43:0.5236210189773913):0.25934713328834313,150:0.576046623342043):0.985467979226334,((101:0.43652246733203315,137:0.5570456472940767):0.8662082908097093,62:0.20408120562639742):0.826389948938628):0.5332225761739355,127:0.46577265215722485):0.8781853546271939,151:0.7010768833413252):0.9274484177488677,135:0.5291096653980414):0.8575024133552204,(((((141:0.33468912604326956,6:0.41111840051182025):0.8077853603833137,85:0.28916441639437773):0.47054184989639836,4:0.8041478465328685):0.6087042314596806,((10:0.9746992889831029,92:0.7623653013461995):0.6929656704985796,95:0.657334724826138):0.3847328613790544):0.6916358480252702,(2:0.8676215338527076,96:0.1840484800680064):0.7187474159211975):0.27901999158825974):0.5568965937798708,(147:0.21921091332982057,68:0.8258597827300935):0.5552962246438888):0.8260973243174158):0.42668269046284024,((((((((((119:0.184893216842694,52:0.5584651079752336):0.33905306127821844,143:0.38276547651601844):0.18811785538144032,(125:0.11328207084608549,47:0.8841907322767074):0.6869304045986698):0.5865291442902746,76:0.5564102536317422):0.7633339241412703,7:0.946236808357798):0.46635396533734297,107:0.6937757404726307):0.5256756486501344,64:0.27107089705206644):0.14875406850895348,120:0.8774530735326429):0.6881099394392124,65:0.139947880341908):0.24861426307671491,108:0.9521638117344918):0.5794355525765641):0.2131844603683972,(86:0.6459954134317863,91:0.9638488033147325):0.8071829447298221):0.27090808054087345,132:0.5380042704888965):0.7045182636744548,((111:0.9096526761786509,80:0.773321104143001):0.819898326916207,36:0.24625159039039476):0.43607845227636444):0.10292078449155637):0.7498828551789197,((((14:0.6127078691243664,90:0.8143618591954018):0.9474365145061804,13:0.8066164211147094):0.4633979961864497,30:0.9749766385155344):0.4513795818290318,((((153:0.8903142684762329,44:0.2796155470231767):0.9385804354924697,(146:0.5838709636914646,98:0.3113817386740156):0.19322785172639034):0.6590433299437836,(((115:0.15298639018869106,58:0.9851955155947784):0.13443147761793645,89:0.7080645098710899):0.3980348813929664,54:0.8592950469448727):0.6066942810750514):0.18998312403212964,75:0.9100421515873822):0.9338662662034):0.7370568518936087):0.9801860443096008,106:0.8344209280780797):0.7857659791468582,(100:0.6627701742097758,112:0.5063375272520168):0.5569135071438928):0.1957697405179345,5:0.3507427549207035):0.15762330669371044):0.779738676972225,((((29:0.13764597235197207,88:0.49760741515415763):0.13485214689920783,42:0.4442875503372409):0.5972725870704979,((((149:0.4943661157069389,152:0.59914205424071):0.31985597485731826,110:0.4590930699976863):0.35648865209499103,74:0.212162784097931):0.4881676596624882,(71:0.9286937705962361,8:0.4412307494102876):0.9539870373741829):0.48680804159269464):0.4322943860609726,39:0.9320385657173911):0.7416540755372096):0.7208808688896593,((((((40:0.32335051700566064,99:0.37937751786542373):0.7615517849450885,130:0.7269323532401463):0.7029164163593771,87:0.39002430676288435):0.9397792094685887,3:0.42818636379453046):0.9505458451419752,((34:0.9764533203735601,46:0.8384827428744058):0.887241451186982,28:0.13242532425474757):0.7219176613802888):0.23414885693066756,60:0.4927544954835472):0.35672430605222827):0.16784639953686042,(124:0.8703702032215355,16:0.9007414363051383):0.8272200202585324):0.0,0:0.0);'
            #return '((((((74:0.00158,67:0.00147):0.00219,(((80:0.00175,(69:0.00153,(81:0.00128,68:0.00047):0.00108):0.00021):0.00156,4:0.00497):0.00013,((133:0.00707,((115:0.00142,96:0.00162):0.00389,(127:0.00419,57:0.00234):0.00024):0.00043):0.00610,(((129:0.01483,128:0.01258):0.02349,10:0.02220):0.01067,(((110:0.01986,32:0.03018):0.00085,27:0.02439):0.00161,((84:0.02267,(90:0.05224,65:0.01999):0.06741):0.08712,((((98:0.01559,95:0.01357):0.00483,((((113:0.00996,(118:0.01509,112:0.01146):0.00201):0.00243,111:0.01759):0.00797,(126:0.01728,(((153:0.03625,135:0.03207):0.02023,((123:0.00285,(122:0.00146,121:0.00028):0.00194):0.00538,86:0.01007):0.00596):0.00314,44:0.01368):0.00247):0.00218):0.00103,(60:0.00611,(16:0.00049,15:0.00169):0.00607):0.01037):0.00035):0.00489,(((125:0.01786,124:0.01565):0.00892,(((130:0.03630,114:0.03724):0.00223,106:0.02954):0.00318,102:0.02412):0.00628):0.00240,((54:0.01483,19:0.02781):0.01776,((((45:0.00394,(136:0.00477,(51:0.00003,18:0.00041):0.00002):0.00346):0.01614,(20:0.01759,((((116:0.00391,(((105:0.00001,104:0.00042):0.00068,103:0.00237):0.00066,((63:0.00081,29:0.00050):0.00051,(30:0.00104,28:0.00070):0.00036):0.00043):0.00088):0.00037,26:0.00309):0.00137,(120:0.00521,25:0.00175):0.00206):0.00460,(17:0.00870,((62:0.00415,((131:0.00377,(100:0.00199,85:0.00976):0.00058):0.00016,53:0.00071):0.00238):0.00213,(55:0.00505,(97:0.00254,13:0.00443):0.00431):0.00283):0.00208):0.01527):0.00236):0.00180):0.00285,((99:0.01663,(23:0.00062,(22:0.00022,21:0.00022):0.00068):0.01808):0.00968,(101:0.00441,12:0.00560):0.01501):0.00020):0.00037,((((108:0.01911,48:0.01179):0.00251,(46:0.00155,((58:0.00043,37:0.00001):0.00143,((56:0.00191,38:0.00157):0.00046,(36:0.00140,24:0.00121):0.00020):0.00042):0.00063):0.00811):0.00699,((59:0.00365,(50:0.00067,39:0.00107):0.00288):0.00897,(((((109:0.00043,49:0.00000):0.00067,47:0.00107):0.00202,34:0.00102):0.00266,((107:0.00304,43:0.00088):0.00136,31:0.00212):0.00103):0.00509,(40:0.00414,(35:0.00000,14:0.00000):0.00326):0.00796):0.00133):0.00212):0.00123,((41:0.00009,33:0.00165):0.01313,(148:0.01747,(42:0.00000,6:0.00000):0.02431):0.02581):0.00435):0.00457):0.00092):0.00074):0.00062):0.00244,((52:0.00623,(94:0.00106,1:0.03289):0.00421):0.02268,(((((143:0.01269,142:0.00907):0.06123,(140:0.02122,((141:0.01612,(139:0.00865,138:0.00789):0.00673):0.00139,137:0.02363):0.00457):0.02164):0.02569,(134:0.10397,(144:0.07853,(151:0.09246,(154:0.27905,117:0.06125):0.07269):0.04451):0.01766):0.00510):0.00403,((147:0.01720,146:0.01718):0.01042,(119:0.03662,83:0.02953):0.00351):0.01413):0.00066,(82:0.04391,((9:0.03393,((((145:0.01759,91:0.00547):0.02071,((132:0.01806,93:0.01632):0.00670,(92:0.01241,(89:0.00723,88:0.00408):0.01522):0.00272):0.00104):0.00989,87:0.02134):0.00196,((11:0.00845,(61:0.00226,8:0.00427):0.00417):0.01284,(150:0.03854,(7:0.00957,5:0.00871):0.00715):0.00266):0.00414):0.01177):0.00574,(((155:0.32836,152:0.04457):0.02833,149:0.02867):0.07311,2:0.09508):0.02500):0.00275):0.00127):0.00272):0.00096):0.00172):0.00170):0.00016):0.01078):0.00373):0.00032):0.00194,((79:0.00087,78:0.00044):0.00036,3:0.00225):0.00104):0.00028,(77:0.00227,(76:0.00200,64:0.00148):0.00034):0.00008):0.00025,(70:0.00002,66:0.00085):0.00013):0.00009,(73:0.00075,(72:0.00087,71:0.00044):0.00012):0.00022,75:0.00196):0.00000;'
            return '((((((74:0.00158,67:0.00147):0.00219,(((80:0.00175,(69:0.00153,(81:0.00128,68:0.00047):0.00108):0.00021):0.00156,4:0.00497):0.00013,((133:0.00707,((115:0.00142,96:0.00162):0.00389,(127:0.00419,57:0.00234):0.00024):0.00043):0.00610,(((129:0.01483,128:0.01258):0.02349,10:0.02220):0.01067,(((110:0.01986,32:0.03018):0.00085,27:0.02439):0.00161,((84:0.02267,(90:0.05224,65:0.01999):0.06741):0.08712,((((98:0.01559,95:0.01357):0.00483,((((113:0.00996,(118:0.01509,112:0.01146):0.00201):0.00243,111:0.01759):0.00797,(126:0.01728,(((153:0.03625,135:0.03207):0.02023,((123:0.00285,(122:0.00146,121:0.00028):0.00194):0.00538,86:0.01007):0.00596):0.00314,44:0.01368):0.00247):0.00218):0.00103,(60:0.00611,(16:0.00049,15:0.00169):0.00607):0.01037):0.00035):0.00489,(((125:0.01786,124:0.01565):0.00892,(((130:0.03630,114:0.03724):0.00223,106:0.02954):0.00318,102:0.02412):0.00628):0.00240,((54:0.01483,19:0.02781):0.01776,((((45:0.00394,(136:0.00477,(51:0.00003,18:0.00041):0.00002):0.00346):0.01614,(20:0.01759,((((116:0.00391,(((105:0.00001,104:0.00042):0.00068,103:0.00237):0.00066,((63:0.00081,29:0.00050):0.00051,(30:0.00104,28:0.00070):0.00036):0.00043):0.00088):0.00037,26:0.00309):0.00137,(120:0.00521,25:0.00175):0.00206):0.00460,(17:0.00870,((62:0.00415,((131:0.00377,(100:0.00199,85:0.00976):0.00058):0.00016,53:0.00071):0.00238):0.00213,(55:0.00505,(97:0.00254,13:0.00443):0.00431):0.00283):0.00208):0.01527):0.00236):0.00180):0.00285,((99:0.01663,(23:0.00062,(22:0.00022,21:0.00022):0.00068):0.01808):0.00968,(101:0.00441,12:0.00560):0.01501):0.00020):0.00037,((((108:0.01911,48:0.01179):0.00251,(46:0.00155,((58:0.00043,37:0.00001):0.00143,((56:0.00191,38:0.00157):0.00046,(36:0.00140,24:0.00121):0.00020):0.00042):0.00063):0.00811):0.00699,((59:0.00365,(50:0.00067,39:0.00107):0.00288):0.00897,(((((109:0.00043,49:0.00000):0.00067,47:0.00107):0.00202,34:0.00102):0.00266,((107:0.00304,43:0.00088):0.00136,31:0.00212):0.00103):0.00509,(40:0.00414,(35:0.00000,14:0.00000):0.00326):0.00796):0.00133):0.00212):0.00123,((41:0.00009,33:0.00165):0.01313,(148:0.01747,(42:0.00000,6:0.00000):0.02431):0.02581):0.00435):0.00457):0.00092):0.00074):0.00062):0.00244,((52:0.00623,(94:0.00106,1:0.03289):0.00421):0.02268,(((((143:0.01269,142:0.00907):0.06123,(140:0.02122,((141:0.01612,(139:0.00865,138:0.00789):0.00673):0.00139,137:0.02363):0.00457):0.02164):0.02569,(134:0.10397,(144:0.07853,(151:0.09246,(154:0.27905,117:0.06125):0.07269):0.04451):0.01766):0.00510):0.00403,((147:0.01720,146:0.01718):0.01042,(119:0.03662,83:0.02953):0.00351):0.01413):0.00066,(82:0.04391,((9:0.03393,((((145:0.01759,91:0.00547):0.02071,((132:0.01806,93:0.01632):0.00670,(92:0.01241,(89:0.00723,88:0.00408):0.01522):0.00272):0.00104):0.00989,87:0.02134):0.00196,((11:0.00845,(61:0.00226,8:0.00427):0.00417):0.01284,(150:0.03854,(7:0.00957,5:0.00871):0.00715):0.00266):0.00414):0.01177):0.00574,(((155:0.32836,152:0.04457):0.02833,149:0.02867):0.07311,2:0.09508):0.02500):0.00275):0.00127):0.00272):0.00096):0.00172):0.00170):0.00016):0.01078):0.00373):0.00032):0.00194,((79:0.00087,78:0.00044):0.00036,3:0.00225):0.00104):0.00028,(77:0.00227,(76:0.00200,64:0.00148):0.00034):0.00008):0.00025,(70:0.00002,66:0.00085):0.00013):0.00009,(73:0.00075,(72:0.00087,71:0.00044):0.00012):0.00022,75:0.00196):0.00000;'
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
        if self.sanity_check or self.random_sanity_check:
            return ['((52:6.821929e-03,((((2:4.398080e-02,(((((145:8.657433e-03,91:4.622826e-03):2.222114e-02,((93:1.284674e-02,132:1.985680e-02):8.439914e-03,(((89:1.020633e-02,88:4.611548e-03):8.036501e-03,90:1.429933e-02):1.439583e-02,92:9.908956e-03):1.750425e-03):5.724766e-03):7.037225e-03,87:1.626403e-02):4.747258e-03,(((7:9.739291e-03,5:5.587849e-03):6.613494e-03,(150:1.664500e-02,152:1.724125e-02):1.823357e-02):1.534167e-03,(11:9.469409e-03,(61:5.838223e-04,8:8.467112e-03):4.839481e-03):1.478233e-02):5.742910e-03):1.842902e-02,9:3.628046e-02):4.929418e-03):4.712541e-03,82:4.185746e-02):3.380475e-03,(((((124:1.558528e-02,125:1.705412e-02):5.917463e-03,((102:2.129084e-02,155:9.638513e-03):4.271744e-03,(149:2.436750e-02,(106:2.722806e-02,(130:3.687226e-02,114:4.149299e-02):1.000061e-02):4.023099e-03):2.147868e-03):1.233641e-02):3.719823e-03,((((54:1.214695e-02,19:2.168458e-02):1.996561e-02,((((38:1.057892e-03,((36:7.753757e-04,(46:2.622027e-03,24:1.787443e-03):2.029430e-04):1.481560e-03,(58:1.714262e-04,37:2.458943e-04):2.601513e-03):1.274226e-04):2.389942e-03,56:1.949454e-03):1.489446e-02,(48:1.137571e-02,108:1.964277e-02):1.324782e-03):7.644887e-03,(((39:1.804400e-03,50:3.879771e-04):3.067958e-03,59:3.259752e-03):6.995533e-03,(((6:1.574761e-04,42:2.710545e-05):4.343268e-04,(33:2.616433e-03,41:6.108494e-06):1.184825e-03):1.058122e-02,(((14:6.407019e-05,35:1.283827e-03):3.152093e-03,40:3.500127e-03):1.097830e-02,((((107:1.333270e-03,84:3.209979e-04):2.963766e-03,43:1.910537e-03):1.115617e-03,31:2.194031e-03):1.856883e-03,((47:2.408071e-03,(109:3.641932e-04,49:1.155934e-04):3.415181e-03):2.683443e-03,34:2.281842e-03):4.028540e-03):3.179844e-03):2.139192e-03):1.070869e-03):2.343303e-03):8.589032e-03):3.753159e-03,(((((((136:1.054858e-03,45:1.777442e-04):9.470442e-04,18:8.069737e-04):2.680686e-03,51:1.513089e-03):1.472587e-02,(20:1.791231e-02,((17:6.050183e-03,(((53:2.324964e-03,((100:9.971849e-04,85:1.361781e-03):1.047096e-03,131:4.906427e-03):7.243432e-04):1.869450e-03,62:6.183967e-03):2.898555e-03,(55:6.269062e-03,(13:5.770393e-03,97:4.514650e-03):3.446166e-03):3.282607e-03):1.379633e-03):2.240766e-02,((120:1.168439e-03,25:4.536840e-03):2.917149e-03,(((((63:3.576971e-04,((104:6.782281e-04,105:8.341392e-05):2.542285e-03,(103:2.267958e-03,148:4.839063e-05):2.790594e-04):1.147993e-03):3.031773e-04,29:1.385090e-03):2.901980e-04,(28:1.255159e-03,30:8.757002e-04):1.490579e-03):1.139399e-03,26:2.656695e-03):5.694807e-04,116:5.345046e-03):3.441066e-03):2.961141e-03):5.676048e-03):6.805834e-03):5.342581e-03,(((22:5.519097e-04,21:5.260546e-04):1.063093e-03,23:1.409785e-03):2.054762e-02,99:1.438413e-02):1.228856e-02):1.107615e-03,(((95:9.283287e-03,98:1.739668e-02):6.505733e-03,((16:1.866647e-03,(153:9.563353e-04,15:6.850920e-04):5.016234e-04):6.647760e-03,60:6.967831e-03):1.297224e-02):6.103056e-04,((111:1.652701e-02,(113:1.019696e-02,(118:1.794353e-02,112:9.963961e-03):5.803295e-03):3.456559e-03):1.124700e-02,(126:1.994845e-02,((86:1.119997e-02,(((135:4.125296e-03,123:2.609975e-03):1.338629e-03,122:5.599224e-04):1.247347e-04,121:1.956632e-03):4.255348e-03):6.514329e-03,44:8.702270e-03):1.406738e-03):3.081689e-03):4.390131e-03):6.593693e-03):2.821015e-04,(101:4.858902e-03,12:4.198135e-03):1.335322e-02):4.585962e-04):2.756843e-03,((144:4.292494e-02,((143:8.977315e-03,142:8.391560e-03):7.271121e-02,(140:1.547496e-02,(137:2.375343e-02,(141:1.801854e-02,(139:4.845625e-03,138:7.509111e-03):5.295656e-03):2.358940e-03):1.229887e-02):2.599447e-02):4.443259e-02):7.088933e-03,134:5.769189e-02):6.826836e-03):5.876646e-04):4.031933e-03,(32:3.250200e-02,((117:3.522599e-02,(151:7.473737e-03,110:9.434156e-03):5.396982e-03):7.039427e-03,(27:2.809174e-02,154:2.327384e-02):4.710086e-03):2.332033e-03):3.477086e-03):1.630971e-03,((((127:5.961962e-03,57:2.631306e-03):6.478147e-04,((96:1.409200e-03,115:2.740476e-03):7.137445e-03,133:7.424993e-03):1.159072e-03):6.523926e-03,(4:5.405740e-03,(((((81:6.815847e-04,68:1.155674e-03):2.735598e-03,69:7.025004e-04):8.998414e-04,80:2.236265e-03):1.839654e-03,(((3:4.236887e-03,79:1.677530e-03):3.770879e-04,78:1.062032e-03):2.312128e-03,((77:1.334890e-03,(66:2.301659e-04,((70:8.259407e-05,(75:2.060793e-03,65:3.982049e-03):3.647037e-04):4.418750e-04,(71:1.999872e-03,(72:7.517143e-04,73:6.105537e-04):8.489613e-05):5.596686e-04):9.888103e-04):1.665829e-03):7.810491e-04,(64:1.400547e-03,76:2.738529e-03):7.529917e-04):1.655366e-03):3.548366e-03):4.212272e-04,(67:1.216874e-03,74:1.827134e-03):2.430697e-03):1.433684e-03):2.358579e-03):1.337100e-02,((128:1.130754e-02,129:1.857543e-02):2.664069e-02,10:2.449606e-02):1.431688e-02):1.932062e-03):3.746715e-03):1.785518e-03,((83:4.168728e-02,119:4.097966e-02):7.403229e-03,(146:1.888170e-02,147:1.810523e-02):1.032872e-02):2.647861e-02):2.939351e-02):4.262485e-03,94:2.756089e-04,1:6.820178e-04);']
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
        self.loader_seed = config["data"].get(
            "loader_seed",
            config.get("trainer", {}).get("seed"),
        )

        self.train_ids = train_ids
        self.test_ids = test_ids

        self.dataset_train = TreeDataset(
            self.nexus_dir, self.mrbayes_dir, filter_ids=self.train_ids, sanity_check=config["data"].get("sanity_check", False), random_sanity_check=config["data"].get("random_sanity_check", False),
            overfit_velocity_zero=config["data"].get("overfit_velocity_zero", False),
            overfit_velocity_event_states=config["data"].get("overfit_velocity_event_states", False),
            overfit_velocity_orthant_start_states=config["data"].get(
                "overfit_velocity_orthant_start_states", False
            ),
            overfit_velocity_explicit_boundary_end_states=config["data"].get(
                "overfit_velocity_explicit_boundary_end_states", False
            ),
            overfit_velocity_fixed_timepoints=config["data"].get(
                "overfit_velocity_fixed_timepoints"
            ),
            overfit_boundary_prefix_k=config["data"].get("overfit_boundary_prefix_k", -1),
            overfit_start_boundary_prefix_k=config["data"].get("overfit_start_boundary_prefix_k", -1),
            overfit_event_prefix_count=config["data"].get("overfit_event_prefix_count", -1),
            overfit_event_horizon=config["data"].get("overfit_event_horizon", 1),
            overfit_fixed_pair=config["data"].get("overfit_fixed_pair", False),
            overfit_fixed_pair_start_tree_newick=config["data"].get(
                "overfit_fixed_pair_start_tree_newick"
            ),
            overfit_fixed_pair_start_tree_json_path=config["data"].get(
                "overfit_fixed_pair_start_tree_json_path"
            ),
            overfit_fixed_pair_start_tree_json_paths=config["data"].get(
                "overfit_fixed_pair_start_tree_json_paths"
            ),
            overfit_fixed_pair_target_tree_newick=config["data"].get(
                "overfit_fixed_pair_target_tree_newick"
            ),
            overfit_fixed_pair_target_tree_json_path=config["data"].get(
                "overfit_fixed_pair_target_tree_json_path"
            ),
            overfit_fixed_pair_target_tree_json_paths=config["data"].get(
                "overfit_fixed_pair_target_tree_json_paths"
            ),
            overfit_split_multi_subset_events=config["data"].get(
                "overfit_split_multi_subset_events", False
            ),
            overfit_oracle_prefix_start_prob=config["data"].get(
                "overfit_oracle_prefix_start_prob",
                config["data"].get(
                    "analysis_oracle_prefix_start_prob",
                    config.get("trainer", {}).get(
                        "analysis_oracle_prefix_start_prob", 0.0
                    ),
                ),
            ),
            overfit_oracle_prefix_max_fraction=config["data"].get(
                "overfit_oracle_prefix_max_fraction",
                config["data"].get(
                    "analysis_oracle_prefix_max_fraction",
                    config.get("trainer", {}).get(
                        "analysis_oracle_prefix_max_fraction", 0.5
                    ),
                ),
            ),
        )
        self.dataset_val = TreeDataset(
            self.nexus_dir, self.mrbayes_dir, filter_ids=self.test_ids, validation=True, sanity_check=config["data"].get("sanity_check", False), random_sanity_check=config["data"].get("random_sanity_check", False),
            overfit_velocity_zero=config["data"].get("overfit_velocity_zero", False),
            overfit_velocity_event_states=config["data"].get("overfit_velocity_event_states", False),
            overfit_velocity_orthant_start_states=config["data"].get(
                "overfit_velocity_orthant_start_states", False
            ),
            overfit_velocity_explicit_boundary_end_states=config["data"].get(
                "overfit_velocity_explicit_boundary_end_states", False
            ),
            overfit_velocity_fixed_timepoints=config["data"].get(
                "overfit_velocity_fixed_timepoints"
            ),
            overfit_boundary_prefix_k=config["data"].get("overfit_boundary_prefix_k", -1),
            overfit_start_boundary_prefix_k=config["data"].get("overfit_start_boundary_prefix_k", -1),
            overfit_event_prefix_count=config["data"].get("overfit_event_prefix_count", -1),
            overfit_event_horizon=config["data"].get("overfit_event_horizon", 1),
            overfit_fixed_pair=config["data"].get("overfit_fixed_pair", False),
            overfit_fixed_pair_start_tree_newick=config["data"].get(
                "overfit_fixed_pair_start_tree_newick"
            ),
            overfit_fixed_pair_start_tree_json_path=config["data"].get(
                "overfit_fixed_pair_start_tree_json_path"
            ),
            overfit_fixed_pair_start_tree_json_paths=config["data"].get(
                "overfit_fixed_pair_start_tree_json_paths"
            ),
            overfit_fixed_pair_target_tree_newick=config["data"].get(
                "overfit_fixed_pair_target_tree_newick"
            ),
            overfit_fixed_pair_target_tree_json_path=config["data"].get(
                "overfit_fixed_pair_target_tree_json_path"
            ),
            overfit_fixed_pair_target_tree_json_paths=config["data"].get(
                "overfit_fixed_pair_target_tree_json_paths"
            ),
            overfit_split_multi_subset_events=config["data"].get(
                "overfit_split_multi_subset_events", False
            ),
            overfit_oracle_prefix_start_prob=config["data"].get(
                "overfit_oracle_prefix_start_prob",
                config["data"].get(
                    "analysis_oracle_prefix_start_prob",
                    config.get("trainer", {}).get(
                        "analysis_oracle_prefix_start_prob", 0.0
                    ),
                ),
            ),
            overfit_oracle_prefix_max_fraction=config["data"].get(
                "overfit_oracle_prefix_max_fraction",
                config["data"].get(
                    "analysis_oracle_prefix_max_fraction",
                    config.get("trainer", {}).get(
                        "analysis_oracle_prefix_max_fraction", 0.5
                    ),
                ),
            ),
        )
        self.tree_tokenizer = TreeFeatureTokenizer(
            config["model"]["num_node_types"],
            config["model"]["num_edge_types"],
            config["model"]["hidden_dim"],
        )
        self.use_historical_collate = bool(
            config["data"].get("use_historical_collate", False)
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
        generator = None
        if self.loader_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.loader_seed))
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            generator=generator,
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
        if self.use_historical_collate:
            flat_batch = []
            for item in batch:
                if item is None:
                    continue
                multi_step_samples = item.get("multi_step_samples")
                if multi_step_samples:
                    flat_batch.extend(multi_step_samples)
                else:
                    flat_batch.append(item)
            batch = flat_batch

            if "posterior_trees" in batch[0]:
                ids = [item["id"] for item in batch]
                posterior_trees = [item["posterior_trees"] for item in batch]
                mappings = [item["num_to_name"] for item in batch]
                return {
                    "ids": ids,
                    "posterior_trees": posterior_trees,
                    "phyla_embeddings": None,
                    "mappings": mappings,
                    "nexus_filepaths": [item["nexus_path"] for item in batch],
                    "tree_paths": [item["tree_paths"] for item in batch],
                }

            trees_to_tokenize = [item["newick_tree"] for item in batch]
            tokenized_trees = self.tree_tokenizer(trees_to_tokenize)
            num_leaves = [len(batch[i]["sequences"]) for i in range(len(batch))]
            autoregressive_trees_to_tokenize = [
                item["autoregressive_newick"] for item in batch
            ]
            autoregressive_tokenized_trees = self.tree_tokenizer(
                autoregressive_trees_to_tokenize
            )
            mappings = [item["num_to_name"] for item in batch]
            ids = [item["id"] for item in batch]
            batched_autoregressive_time = torch.tensor(
                [item["autoregressive_newick_time"] for item in batch],
                dtype=torch.float32,
            )

            return {
                "tokenized_trees": tokenized_trees,
                "tokenized_autoregressive_trees": autoregressive_tokenized_trees,
                "newick_autoregressive_trees": autoregressive_trees_to_tokenize,
                "nexus_filepaths": [item["nexus_path"] for item in batch],
                "tree_paths": [item["tree_paths"] for item in batch],
                "original_trees": [item["newick_tree"] for item in batch],
                "target_trees": [item["target_tree"] for item in batch],
                "batched_velocity": [item["velocity"] for item in batch],
                "velocity_next_boundary_trees": [
                    item.get("velocity_next_boundary_tree") for item in batch
                ],
                "batched_autoregressive_time": batched_autoregressive_time,
                "batched_autoregressive_labels": [
                    item["autoregressive_labels"] for item in batch
                ],
                "batched_autoregressive_stop_after_merge": torch.tensor(
                    [
                        1.0
                        if item.get("autoregressive_stop_after_merge", False)
                        else 0.0
                        for item in batch
                    ],
                    dtype=torch.float32,
                ),
                "batched_time": torch.tensor(
                    [item["timepoint"] for item in batch], dtype=torch.float32
                ),
                "phyla_embeddings": None,
                "num_leaves": num_leaves,
                "ids": ids,
                "mappings": mappings,
            }

        flat_batch = []
        for item in batch:
            if item is None:
                continue
            multi_step_samples = item.get("multi_step_samples")
            if multi_step_samples:
                flat_batch.extend(multi_step_samples)
            else:
                flat_batch.append(item)
        batch = flat_batch

        if "posterior_trees" in batch[0]:
            ids = [item["id"] for item in batch]
            posterior_trees = [item["posterior_trees"] for item in batch]
            mappings = [item["num_to_name"] for item in batch]
            phyla_embeddings = None

            return {
                "ids": ids,
                "posterior_trees": posterior_trees,
                "phyla_embeddings": phyla_embeddings,
                "mappings": mappings,
                "nexus_filepaths": [item["nexus_path"] for item in batch],
                "tree_paths": [item["tree_paths"] for item in batch],
            }

        # preset_subtree_num is accepted but currently unused in logic below
        # Just ensuring signature matches call site

        trees_to_tokenize = [item["newick_tree"] for item in batch]
        structural_trees = [
            self.tree_tokenizer._newick_to_structural(tree)
            for tree in trees_to_tokenize
        ]
        # Tokenizer runs in worker if num_workers > 0, so must disable gradients
        # to avoid pickling errors (grad_fn cannot be pickled).
        
        try:
            with torch.no_grad():
                tokenized_trees = self.tree_tokenizer(structural_trees)
        except Exception as e:
            print(f"Error in tree tokenization: {e}")
            return None 

        def _aligned_true_edge_lengths(tree_newick, token_masks):
            tree_obj = Tree(tree_newick)
            split_masks, split_lengths = BHVEncoder().return_BHV_encoding(tree_obj)
            true_length_map = {
                int(mask): float(length)
                for mask, length in zip(split_masks, split_lengths)
                if length is not None and float(length) > 1e-8
            }
            biological_bits = max(tree_obj.n_leaves - 1, 0)
            full_model_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
            aligned_lengths = []
            for raw_mask in token_masks:
                raw_mask = int(raw_mask)
                if raw_mask == 0:
                    aligned_lengths.append(0.0)
                    continue

                edge_length = 0.0
                if raw_mask in true_length_map:
                    edge_length = float(true_length_map[raw_mask])
                elif full_model_mask and (full_model_mask ^ raw_mask) in true_length_map:
                    edge_length = float(true_length_map[int(full_model_mask ^ raw_mask)])
                aligned_lengths.append(edge_length)
            return torch.as_tensor(aligned_lengths, dtype=torch.float32)

        tokenized_tree_edge_lengths = [
            _aligned_true_edge_lengths(tree_newick, tokenized_trees[-1][idx])
            for idx, tree_newick in enumerate(trees_to_tokenize)
        ]

        velocity_next_boundary_active_masks = []
        for batch_idx, item in enumerate(batch):
            next_boundary_tree = item.get("velocity_next_boundary_tree")
            if not next_boundary_tree:
                velocity_next_boundary_active_masks.append(None)
                continue

            current_tree_obj = Tree(item["newick_tree"])
            boundary_tree_obj = Tree(next_boundary_tree)
            boundary_masks, boundary_lengths = BHVEncoder().return_BHV_encoding(
                boundary_tree_obj
            )
            boundary_length_map = {
                int(mask): float(length)
                for mask, length in zip(boundary_masks, boundary_lengths)
                if length is not None and float(length) > 1e-8
            }
            biological_bits = max(current_tree_obj.n_leaves - 1, 0)
            full_model_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
            current_masks = [int(mask) for mask in tokenized_trees[-1][batch_idx]]
            active_masks = set()
            for raw_mask in current_masks:
                raw_mask = int(raw_mask)
                if raw_mask == 0:
                    continue
                if raw_mask in boundary_length_map or (
                    full_model_mask and (full_model_mask ^ raw_mask) in boundary_length_map
                ):
                    active_masks.add(raw_mask)
            velocity_next_boundary_active_masks.append(active_masks)

        num_leaves = [len(batch[i]["sequences"]) for i in range(len(batch))]

        autoregressive_trees_to_tokenize = [
            item["autoregressive_newick"] for item in batch
        ]

        try:
            autoregressive_tokenized_trees = self.tree_tokenizer(
                autoregressive_trees_to_tokenize
            )
        except Exception as e:
            print(f"Error in autoregressive tree tokenization: {e}")
            return None
            
        mappings = [item['num_to_name'] for item in batch]
        ids = [item["id"] for item in batch]

        batched_autoregressive_time = torch.tensor(
            [item["autoregressive_newick_time"] for item in batch], dtype=torch.float32
        )

        to_run = {
            "tokenized_trees": tokenized_trees,
            "tokenized_autoregressive_trees": autoregressive_tokenized_trees,
            "newick_autoregressive_trees": autoregressive_trees_to_tokenize,
            "nexus_filepaths": [item["nexus_path"] for item in batch],
            "tree_paths": [item["tree_paths"] for item in batch],
            "original_trees": [item["newick_tree"] for item in batch],
            "target_trees": [item["target_tree"] for item in batch],
            "batched_velocity": [item["velocity"] for item in batch],
            "tokenized_tree_edge_lengths": tokenized_tree_edge_lengths,
            "velocity_next_boundary_trees": [
                item.get("velocity_next_boundary_tree") for item in batch
            ],
            "velocity_next_boundary_active_masks": velocity_next_boundary_active_masks,
            "batched_autoregressive_time": batched_autoregressive_time,
            "batched_autoregressive_labels": [
                item["autoregressive_labels"] for item in batch
            ],
            "batched_autoregressive_stop_after_merge": torch.tensor(
                [
                    1.0 if item.get("autoregressive_stop_after_merge", False) else 0.0
                    for item in batch
                ],
                dtype=torch.float32,
            ),
            "batched_time": torch.tensor(
                [item["timepoint"] for item in batch], dtype=torch.float32
            ),
            # "phyla_embeddings": torch.tensor([item['phyla_embedding'] for item in batch], dtype=torch.float32),
            "phyla_embeddings": None,
            "num_leaves": num_leaves,
            "ids": ids,
            "mappings": mappings,
            "sequence_ordering_maps": [item["seq_ordering_map"] for item in batch],
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
