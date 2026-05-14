import argparse
import copy
import json
import math
import os
import random
import re
import shlex
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml
from ete3 import Tree as EteTree

from model.model import return_model
from run.TrainingModule import (
    _branch_relax_raw_length_map,
    _branch_relax_entries_for_tree,
    _build_branch_relax_samples_for_module,
    _move_tokenized_batch_to_device,
    _tree_to_model_split_lengths,
)
from utils.bhv_movie import build_tree_from_splits
from analysis.full_sanity_fixedpair_20260401.multi_ds_branchwarm_cumulative_mh_experiment import (
    GenericJCLikelihood,
)


DEFAULT_BASE_CONFIG = (
    "/home/yektefai/PhylaFlow/configs/"
    "local_ds1_frozenprobe64_fh16_aradd_scale128x4_lr2e3_20260428.yaml"
)
DEFAULT_START_TREES = (
    "/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/"
    "ds1_arcase_best_topokl_step031000_checkpoint_samples_20260427/"
    "ds1_arcase_best_topokl_step031000_trees/"
    "ds1_arcase_best_topokl_step031000_sampled_start_trees.txt"
)
DEFAULT_TARGET_TREES = (
    "/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/"
    "ds1_arcase_topokl031000_branchwarm_splitguided_mh_20260427/"
    "branch_warmed_start_trees.txt"
)
DEFAULT_OUT_DIR = (
    "/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/"
    "standalone_branch_relaxer_ds1_20260429"
)
DEFAULT_REAL_DATA_ROOT = Path(
    os.environ.get("PHYLAFLOW_DATASETS_ROOT", "/ewsc/yektefai/phylaflow_datasets")
)
DEFAULT_REAL_NEXUS_ROOT = DEFAULT_REAL_DATA_ROOT / "nexus"
DEFAULT_REAL_RUNS_ROOT = DEFAULT_REAL_DATA_ROOT / "runs"
DEFAULT_REAL_STREAM_ROOT = DEFAULT_REAL_DATA_ROOT / "real_unique_topology_stream"


class BranchDeltaHead(nn.Module):
    def __init__(self, edge_dim, *, hidden_dim=128, case_dim=0, num_cases=0):
        super().__init__()
        self.case_dim = int(case_dim)
        self.case_embedding = None
        if self.case_dim > 0:
            self.case_embedding = nn.Embedding(max(1, int(num_cases)), self.case_dim)
        input_dim = int(edge_dim) + 3 + self.case_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, edge_features, numeric_features, case_indices=None):
        parts = [edge_features, numeric_features]
        if self.case_embedding is not None:
            if case_indices is None:
                case_indices = torch.zeros(
                    edge_features.shape[0],
                    dtype=torch.long,
                    device=edge_features.device,
                )
            case_indices = torch.clamp(
                case_indices.to(device=edge_features.device, dtype=torch.long),
                min=0,
                max=self.case_embedding.num_embeddings - 1,
            )
            parts.append(self.case_embedding(case_indices))
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


class StandaloneRelaxer(nn.Module):
    def __init__(self, model, head):
        super().__init__()
        self.model = model
        self.head = head

    def forward_batch(self, tokenized, newicks, samples, device, phyla_bank=None):
        tokenized = _move_tokenized_batch_to_device(tokenized, device)
        phyla_embeddings = None
        if phyla_bank:
            phyla_embeddings = [
                phyla_bank[str(sample["dataset_id"]).upper()]
                for sample in samples
            ]
        edge_outputs = self.model(
            tokenized,
            torch.full((len(newicks),), 4.0, dtype=torch.float32, device=device),
            phyla_embeddings=phyla_embeddings,
            return_leafs_only=False,
            return_edges_only=True,
            return_edge_features=True,
        )
        _edge_values, _edge_pad_mask, edge_features = edge_outputs
        edge_split_masks = tokenized[-1]

        preds = []
        labels = []
        for batch_idx, sample in enumerate(samples):
            entries, _lengths, _n_leaves, _mapping = _branch_relax_entries_for_tree(
                self,
                sample["newick_tree"],
                edge_split_masks[batch_idx],
                labels=sample["labels"],
            )
            if not entries:
                continue
            feature_block = torch.stack(
                [edge_features[batch_idx, entry["edge_index"]] for entry in entries],
                dim=0,
            )
            numeric = torch.tensor(
                [entry["numeric"] for entry in entries],
                dtype=torch.float32,
                device=device,
            )
            case_indices = torch.full(
                (len(entries),),
                int(sample["case_index"]),
                dtype=torch.long,
                device=device,
            )
            preds.append(self.head(feature_block, numeric, case_indices))
            labels.append(
                torch.tensor(
                    [float(entry["label"]) for entry in entries],
                    dtype=torch.float32,
                    device=device,
                )
            )
        if not preds:
            return None, None
        return torch.cat(preds), torch.cat(labels)


def _small_model_config(base_config_path, args):
    with open(base_config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = copy.deepcopy(cfg)
    model_cfg = cfg["model"]
    model_cfg["embed_dim"] = int(args.embed_dim)
    model_cfg["hidden_dim"] = int(args.embed_dim)
    model_cfg["n_layers"] = int(args.n_layers)
    model_cfg["n_heads"] = int(args.n_heads)
    model_cfg["dropout"] = float(args.dropout)
    model_cfg["attention_dropout"] = float(args.dropout)
    model_cfg["activation_dropout"] = float(args.dropout)
    model_cfg["drop_path_rate"] = 0.0
    model_cfg["phyla_dim"] = int(args.phyla_dim)
    model_cfg["phyla_use_leaf_tokens"] = bool(args.phyla_use_leaf_tokens)
    model_cfg["phyla_use_split_tokens"] = bool(args.phyla_use_split_tokens)
    model_cfg["use_performer"] = False
    model_cfg["autoregressive_use_case_conditioning"] = False
    model_cfg["autoregressive_use_start_topology_conditioning"] = False
    model_cfg["autoregressive_frozen_start_case_embedding_path"] = None
    model_cfg["first_hit_head_mode"] = "base"
    model_cfg["first_hit_head_num_cases"] = None
    model_cfg["first_hit_frozen_start_case_embedding_path"] = None
    return cfg


def _parse_dataset_spec(raw):
    parts = str(raw).split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "dataset specs must be DATASET_ID:START_TREES:TARGET_TREES; "
            f"got {raw!r}"
        )
    dataset_id, start_trees, target_trees = parts
    return str(dataset_id).upper(), start_trees, target_trees


def _parse_eval_spec(raw):
    parts = str(raw).split(":", 3)
    if len(parts) != 4:
        raise ValueError(
            "eval specs must be NAME:DATASET_ID:START_TREES:TARGET_TREES; "
            f"got {raw!r}"
        )
    eval_name, dataset_id, start_trees, target_trees = parts
    return str(eval_name), str(dataset_id).upper(), start_trees, target_trees


def _read_dataset_spec_file(path):
    specs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) < 3:
                    raise ValueError(
                        f"{path}:{line_number} needs DATASET_ID<TAB>START<TAB>TARGET"
                    )
                specs.append(f"{parts[0]}:{parts[1]}:{parts[2]}")
            else:
                # Accept the same DATASET_ID:START:TARGET shape used by --dataset-spec.
                _parse_dataset_spec(line)
                specs.append(line)
    return specs


def _read_eval_spec_file(path):
    specs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) < 4:
                    raise ValueError(
                        f"{path}:{line_number} needs NAME<TAB>DATASET_ID<TAB>START<TAB>TARGET"
                    )
                specs.append(f"{parts[0]}:{parts[1]}:{parts[2]}:{parts[3]}")
            else:
                _parse_eval_spec(line)
                specs.append(line)
    return specs


def _ensure_newick_semicolon(newick):
    stripped = str(newick).strip()
    return stripped if stripped.endswith(";") else stripped + ";"


def _localize_index_path(raw_path, stream_root):
    raw = str(raw_path)
    candidate = Path(raw)
    if candidate.exists():
        return candidate
    if stream_root:
        stream_root = Path(stream_root)
        marker = "/real_unique_topology_stream/"
        if marker in raw:
            suffix = raw.split(marker, 1)[1]
            candidate = stream_root / suffix
            if candidate.exists():
                return candidate
        match = re.search(r"(worker_\d{3}/.+)$", raw)
        if match:
            candidate = stream_root / match.group(1)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(raw_path)


def _read_tree_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    for key in ("tree", "start_tree", "target_tree", "final_tree"):
        if payload.get(key):
            return _ensure_newick_semicolon(payload[key])
    raise ValueError(f"No tree key found in {path}")


def _read_sample_index_files(paths, *, stream_root):
    rows = []
    for index_path in paths:
        with Path(index_path).open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                row = json.loads(line)
                dataset_id = str(row.get("dataset_id") or "").upper()
                start_path = (
                    row.get("start_path")
                    or row.get("start_json")
                    or row.get("local_start_path")
                )
                target_path = (
                    row.get("target_path")
                    or row.get("target_json")
                    or row.get("local_target_path")
                )
                if not dataset_id or not start_path or not target_path:
                    raise ValueError(
                        f"{index_path}:{line_number} needs dataset_id, start_path, target_path"
                    )
                rows.append(
                    {
                        **row,
                        "dataset_id": dataset_id,
                        "case_index": int(row.get("case_index", len(rows))),
                        "start_path": str(_localize_index_path(start_path, stream_root)),
                        "target_path": str(_localize_index_path(target_path, stream_root)),
                        "index_path": str(index_path),
                        "index_line_number": int(line_number),
                    }
                )
    if not rows:
        raise ValueError("No rows loaded from --sample-index-file")
    return rows


def _build_branch_relax_sample_from_trees(
    module_like,
    *,
    start_tree,
    target_tree,
    dataset_id,
    case_index,
    num_leaves_hint=None,
):
    start_tree = _ensure_newick_semicolon(start_tree)
    target_tree = _ensure_newick_semicolon(target_tree)
    start_lengths, n_leaves, _mapping = _tree_to_model_split_lengths(
        module_like,
        start_tree,
    )
    target_lengths, _target_n_leaves, _target_mapping = _branch_relax_raw_length_map(
        target_tree
    )
    biological_bits = max(int(n_leaves) - 1, 0)
    full_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
    labels = {}
    for mask, start_length in sorted(start_lengths.items()):
        if full_mask and int(mask) == int(full_mask):
            continue
        target_length = target_lengths.get(int(mask))
        if target_length is None and full_mask:
            target_length = target_lengths.get(full_mask ^ int(mask))
        if target_length is None:
            continue
        labels[int(mask)] = float(target_length) - float(start_length)
    if not labels:
        return None
    return {
        "case_index": int(case_index),
        "dataset_id": str(dataset_id).upper(),
        "newick_tree": start_tree,
        "target_tree": target_tree,
        "num_leaves": int(num_leaves_hint or n_leaves),
        "labels": labels,
    }


class IndexedSampleBank:
    def __init__(self, rows, module_like):
        self.rows = list(rows)
        self.module_like = module_like
        self.rows_by_dataset = defaultdict(list)
        for row in self.rows:
            self.rows_by_dataset[str(row["dataset_id"]).upper()].append(row)
        self.dataset_ids = sorted(self.rows_by_dataset)

    def __len__(self):
        return len(self.rows)

    def _build_sample(self, row):
        start_tree = _read_tree_json(row["start_path"])
        target_tree = _read_tree_json(row["target_path"])
        return _build_branch_relax_sample_from_trees(
            self.module_like,
            start_tree=start_tree,
            target_tree=target_tree,
            dataset_id=row["dataset_id"],
            case_index=int(row.get("case_index", 0)),
            num_leaves_hint=row.get("num_leaves"),
        )

    def random_batch(self, rng, batch_size, *, balanced):
        batch = []
        target_size = min(int(batch_size), len(self.rows))
        max_attempts = max(100, target_size * 20)
        attempts = 0
        while len(batch) < target_size and attempts < max_attempts:
            attempts += 1
            if balanced and self.dataset_ids:
                dataset_id = rng.choice(self.dataset_ids)
                row = rng.choice(self.rows_by_dataset[dataset_id])
            else:
                row = rng.choice(self.rows)
            sample = self._build_sample(row)
            if sample is not None:
                batch.append(sample)
        if not batch:
            raise ValueError("Failed to materialize any indexed branch-relax samples")
        return batch

    def materialize_eval_samples(self, rng, eval_count):
        if eval_count is not None and int(eval_count) > 0 and len(self.rows) > int(eval_count):
            rows = rng.sample(self.rows, int(eval_count))
        else:
            rows = list(self.rows)
        samples = []
        for row in rows:
            sample = self._build_sample(row)
            if sample is not None:
                samples.append(sample)
        if not samples:
            raise ValueError("Failed to materialize any indexed eval samples")
        return samples


def _parse_real_nexus_matrix(path):
    sequences = {}
    in_matrix = False
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("["):
                continue
            lower = line.lower()
            if not in_matrix:
                if lower.startswith("matrix"):
                    in_matrix = True
                    line = line[len("matrix") :].strip()
                    if not line:
                        continue
                else:
                    continue
            if ";" in line:
                line = line.split(";", 1)[0].strip()
                done = True
            else:
                done = False
            if line:
                parts = shlex.split(line, comments=False, posix=True)
                if len(parts) >= 2:
                    sequences[str(parts[0])] = "".join(str(part) for part in parts[1:])
            if done:
                break
    if not sequences:
        raise ValueError(f"Failed to parse MATRIX block from {path}")
    return sequences


def _parse_real_translate_block(path):
    mapping = {}
    in_translate = False
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("["):
                continue
            if not in_translate:
                if line.lower().startswith("translate"):
                    in_translate = True
                    line = line[len("translate") :].strip()
                    if not line:
                        continue
                else:
                    continue
            done = line.endswith(";")
            line = line.rstrip(";").rstrip(",").strip()
            match = re.match(r"(\d+)\s+(.+)$", line)
            if match:
                taxon = match.group(2).strip().strip(",").strip("'\"")
                mapping[int(match.group(1))] = taxon
            if done:
                break
    if not mapping:
        raise ValueError(f"Failed to parse translate block from {path}")
    return mapping


def _encode_real_sequence(sequence):
    masks = {
        "A": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "C": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        "G": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
        "T": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "U": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "?": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        "-": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        "N": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    return np.asarray(
        [masks.get(str(base).upper(), masks["?"]) for base in str(sequence)],
        dtype=np.float64,
    )


def _real_translate_path(runs_root, dataset_id):
    run_dir = Path(runs_root) / str(dataset_id)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    candidates = sorted(run_dir.glob("*run1.t")) + sorted(run_dir.glob("*.t"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No .t file found under {run_dir}")


class RealNexusJCLikelihood:
    def __init__(
        self,
        *,
        dataset_id,
        nexus_root,
        runs_root,
        branch_length_floor=1e-8,
    ):
        dataset_id = str(dataset_id).upper()
        nexus_path = Path(nexus_root) / f"{dataset_id}.nex"
        if not nexus_path.exists():
            alternate = Path(nexus_root) / f"{dataset_id}.nexus"
            nexus_path = alternate if alternate.exists() else nexus_path
        if not nexus_path.exists():
            raise FileNotFoundError(nexus_path)
        translate = _parse_real_translate_block(
            _real_translate_path(runs_root, dataset_id)
        )
        sequences = _parse_real_nexus_matrix(nexus_path)
        ordered_taxa = [translate[index] for index in sorted(translate)]
        missing = [name for name in ordered_taxa if name not in sequences]
        if missing:
            raise ValueError(f"{nexus_path} is missing taxa: {missing[:10]}")
        lengths = {len(sequences[name]) for name in ordered_taxa}
        if len(lengths) != 1:
            raise ValueError(f"Sequences are not aligned to one length: {sorted(lengths)}")

        self.n_sites = int(next(iter(lengths)))
        self.branch_length_floor = float(branch_length_floor)
        self.transition_cache = {}
        self.leaf_vectors = {
            str(index): _encode_real_sequence(sequences[translate[index]])
            for index in sorted(translate)
        }

    def _leaf_vector(self, name):
        key = str(name)
        cached = self.leaf_vectors.get(key)
        if cached is not None:
            return cached
        try:
            one_based = str(int(key) + 1)
        except Exception:
            one_based = ""
        cached = self.leaf_vectors.get(one_based)
        if cached is None:
            raise ValueError(f"Unknown leaf label {name!r}; expected numeric labels")
        return cached

    def _transition(self, branch_length):
        length = max(float(branch_length), self.branch_length_floor)
        cached = self.transition_cache.get(length)
        if cached is not None:
            return cached
        decay = math.exp(-4.0 * length / 3.0)
        same = 0.25 + 0.75 * decay
        different = 0.25 - 0.25 * decay
        matrix = np.full((4, 4), different, dtype=np.float64)
        np.fill_diagonal(matrix, same)
        self.transition_cache[length] = matrix
        return matrix

    def log_likelihood(self, newick):
        tree_text = str(newick).strip()
        if not tree_text.endswith(";"):
            tree_text += ";"
        tree = EteTree(tree_text, format=1, quoted_node_names=True)
        values = {}
        zeros = np.zeros(self.n_sites, dtype=np.float64)
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                values[id(node)] = (self._leaf_vector(str(node.name)), zeros)
                continue
            clv = np.ones((self.n_sites, 4), dtype=np.float64)
            log_scale = np.zeros(self.n_sites, dtype=np.float64)
            for child in node.children:
                child_clv, child_log_scale = values[id(child)]
                transition = self._transition(float(child.dist))
                contribution = child_clv @ transition.T
                clv *= contribution
                log_scale += child_log_scale
            site_scale = np.max(clv, axis=1)
            site_scale = np.maximum(site_scale, np.finfo(np.float64).tiny)
            clv = clv / site_scale[:, None]
            log_scale += np.log(site_scale)
            values[id(node)] = (clv, log_scale)
        root_clv, root_log_scale = values[id(tree)]
        site_likelihood = 0.25 * np.sum(root_clv, axis=1)
        site_likelihood = np.maximum(site_likelihood, np.finfo(np.float64).tiny)
        return float(np.sum(np.log(site_likelihood) + root_log_scale))


def _make_likelihood_scorer(dataset_id, args):
    dataset_id = str(dataset_id).upper()
    if args.likelihood_source in {"auto", "ds"}:
        try:
            return GenericJCLikelihood(dataset_id=dataset_id)
        except FileNotFoundError:
            if args.likelihood_source == "ds":
                raise
    if args.likelihood_source in {"auto", "real"}:
        return RealNexusJCLikelihood(
            dataset_id=dataset_id,
            nexus_root=args.real_nexus_root,
            runs_root=args.real_runs_root,
        )
    raise ValueError(f"Unknown likelihood source: {args.likelihood_source}")


class LazyLikelihoodScorers:
    def __init__(self, dataset_ids, args):
        self.dataset_ids = {str(dataset_id).upper() for dataset_id in dataset_ids}
        self.args = args
        self._cache = {}

    def __getitem__(self, dataset_id):
        dataset_id = str(dataset_id).upper()
        if self.dataset_ids and dataset_id not in self.dataset_ids:
            raise KeyError(dataset_id)
        scorer = self._cache.get(dataset_id)
        if scorer is None:
            scorer = _make_likelihood_scorer(dataset_id, self.args)
            self._cache[dataset_id] = scorer
        return scorer


def _load_phyla_embeddings(path, device):
    payload = torch.load(path, map_location="cpu")
    embeddings = payload.get("embeddings")
    if embeddings is None:
        embeddings = payload.get("phyla_embeddings")
    if embeddings is None:
        raise ValueError(f"No embeddings tensor found in {path}")
    embeddings = embeddings.float()
    if embeddings.dim() == 3:
        if embeddings.shape[0] != 1:
            raise ValueError(f"Expected batch dimension 1 in {path}, got {tuple(embeddings.shape)}")
        embeddings = embeddings.squeeze(0)
    if embeddings.dim() != 2:
        raise ValueError(f"Expected embeddings shape (taxa, dim) in {path}, got {tuple(embeddings.shape)}")
    return embeddings.to(device)


def _candidate_phyla_embedding_paths(embedding_dirs, dataset_id):
    dataset_id = str(dataset_id).upper()
    filenames = [
        f"{dataset_id}_phyla_beta_embeddings.pt",
        f"{dataset_id}_phyla_beta_sitechunk_w256_s256_embeddings.pt",
    ]
    for embedding_dir in embedding_dirs:
        for filename in filenames:
            yield Path(embedding_dir) / filename


def _load_phyla_embedding_bank(datasets, embedding_dir, device):
    if not datasets:
        return {}
    embedding_dirs = [
        Path(part)
        for part in str(embedding_dir).split(os.pathsep)
        if str(part).strip()
    ]
    if not embedding_dirs:
        embedding_dirs = [Path(embedding_dir)]
    bank = {}
    for dataset_id in sorted({str(item).upper() for item in datasets}):
        path = None
        for candidate in _candidate_phyla_embedding_paths(embedding_dirs, dataset_id):
            if candidate.exists():
                path = candidate
                break
        if path is None:
            checked = [
                str(candidate)
                for candidate in _candidate_phyla_embedding_paths(
                    embedding_dirs,
                    dataset_id,
                )
            ]
            raise FileNotFoundError(
                f"No phyla embedding found for {dataset_id}; checked {checked}"
            )
        bank[dataset_id] = _load_phyla_embeddings(path, device)
    return bank


def _add_sample_metadata(samples, *, dataset_id, case_offset):
    updated = []
    for local_index, sample in enumerate(samples):
        item = dict(sample)
        item["dataset_id"] = str(dataset_id).upper()
        item["case_index"] = int(case_offset) + int(local_index)
        updated.append(item)
    return updated


def _build_samples_from_specs(module_like, specs, default_dataset_id):
    all_samples = []
    if specs:
        for raw_spec in specs:
            dataset_id, start_trees, target_trees = _parse_dataset_spec(raw_spec)
            samples = _build_branch_relax_samples_for_module(
                module_like,
                start_trees,
                target_trees,
            )
            all_samples.extend(
                _add_sample_metadata(
                    samples,
                    dataset_id=dataset_id,
                    case_offset=len(all_samples),
                )
            )
    else:
        samples = _build_branch_relax_samples_for_module(
            module_like,
            default_dataset_id[1],
            default_dataset_id[2],
        )
        all_samples.extend(
            _add_sample_metadata(
                samples,
                dataset_id=default_dataset_id[0],
                case_offset=0,
            )
        )
    if not all_samples:
        raise ValueError("No branch relax samples were built")
    return all_samples


def _apply_relaxer(relaxer, sample, device, *, scale=1.0, edge_floor=1e-8, phyla_bank=None):
    relaxer.eval()
    tokenized = relaxer.model.tokenizer([sample["newick_tree"]])
    tokenized = _move_tokenized_batch_to_device(tokenized, device)
    phyla_embeddings = None
    if phyla_bank:
        phyla_embeddings = phyla_bank[str(sample["dataset_id"]).upper()]
    with torch.inference_mode():
        edge_outputs = relaxer.model(
            tokenized,
            torch.tensor([4.0], dtype=torch.float32, device=device),
            phyla_embeddings=phyla_embeddings,
            return_leafs_only=False,
            return_edges_only=True,
            return_edge_features=True,
        )
        _edge_values, _edge_pad_mask, edge_features = edge_outputs
        edge_split_masks = tokenized[-1]
        entries, lengths, n_leaves, mapping = _branch_relax_entries_for_tree(
            relaxer,
            sample["newick_tree"],
            edge_split_masks[0],
            labels=None,
        )
        if not entries:
            return sample["newick_tree"], {"applied": False}
        features = torch.stack(
            [edge_features[0, entry["edge_index"]] for entry in entries],
            dim=0,
        )
        numeric = torch.tensor(
            [entry["numeric"] for entry in entries],
            dtype=torch.float32,
            device=device,
        )
        case_indices = torch.full(
            (len(entries),),
            int(sample["case_index"]),
            dtype=torch.long,
            device=device,
        )
        deltas = relaxer.head(features, numeric, case_indices).detach().cpu().numpy()

    next_lengths = {int(mask): float(length) for mask, length in lengths.items()}
    max_abs_delta = 0.0
    for entry, delta in zip(entries, deltas):
        next_len = max(
            float(entry["length"]) + float(scale) * float(delta),
            float(edge_floor),
        )
        next_lengths[int(entry.get("source_mask", entry["mask"]))] = next_len
        max_abs_delta = max(max_abs_delta, abs(next_len - float(entry["length"])))
    td_next = {
        int(mask): float(length)
        for mask, length in next_lengths.items()
        if float(length) > float(edge_floor)
    }
    newick = build_tree_from_splits(
        list(td_next.keys()),
        td_next,
        int(n_leaves),
        root_leaf=int(n_leaves) - 1,
        mapping=mapping,
    )[1]
    return newick, {"applied": True, "max_abs_delta": float(max_abs_delta)}


def _evaluate(relaxer, samples, scorers, device, *, eval_count, rng, scale, phyla_bank=None):
    if hasattr(samples, "materialize_eval_samples"):
        selected = samples.materialize_eval_samples(rng, eval_count)
    elif eval_count is not None and int(eval_count) > 0 and len(samples) > int(eval_count):
        selected = rng.sample(samples, int(eval_count))
    else:
        selected = list(samples)
    before = []
    after = []
    target = []
    applied = 0
    for sample in selected:
        scorer = scorers[str(sample["dataset_id"]).upper()]
        before.append(scorer.log_likelihood(sample["newick_tree"]))
        relaxed_tree, info = _apply_relaxer(
            relaxer,
            sample,
            device,
            scale=scale,
            phyla_bank=phyla_bank,
        )
        after.append(scorer.log_likelihood(relaxed_tree))
        target.append(scorer.log_likelihood(sample["target_tree"]))
        applied += int(bool(info.get("applied")))
    return {
        "eval_count": len(selected),
        "applied_count": int(applied),
        "before_log_likelihood_mean": float(np.mean(before)),
        "after_log_likelihood_mean": float(np.mean(after)),
        "target_log_likelihood_mean": float(np.mean(target)),
        "after_minus_before_mean": float(np.mean(np.asarray(after) - np.asarray(before))),
        "target_minus_before_mean": float(np.mean(np.asarray(target) - np.asarray(before))),
    }


def _prefix_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _write_jsonl(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--start-trees", default=DEFAULT_START_TREES)
    parser.add_argument("--target-trees", default=DEFAULT_TARGET_TREES)
    parser.add_argument("--dataset-spec", action="append", default=[])
    parser.add_argument("--dataset-spec-file", action="append", default=[])
    parser.add_argument("--sample-index-file", action="append", default=[])
    parser.add_argument("--index-stream-root", default=str(DEFAULT_REAL_STREAM_ROOT))
    parser.add_argument("--eval-start-trees", default=None)
    parser.add_argument("--eval-target-trees", default=None)
    parser.add_argument("--eval-name", default="zero_eval")
    parser.add_argument("--eval-spec", action="append", default=[])
    parser.add_argument("--eval-spec-file", action="append", default=[])
    parser.add_argument("--extra-eval-name", action="append", default=[])
    parser.add_argument("--extra-eval-start-trees", action="append", default=[])
    parser.add_argument("--extra-eval-target-trees", action="append", default=[])
    parser.add_argument("--score-prefix", default=None)
    parser.add_argument("--score-prefixes", default=None)
    parser.add_argument("--load-checkpoint", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--balanced-dataset-sampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-train-eval", action="store_true")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-id", default="DS1")
    parser.add_argument(
        "--phyla-embedding-dir",
        default=(
            "/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/"
            "ds_phyla_embeddings_20260428"
        ),
    )
    parser.add_argument("--real-nexus-root", default=str(DEFAULT_REAL_NEXUS_ROOT))
    parser.add_argument("--real-runs-root", default=str(DEFAULT_REAL_RUNS_ROOT))
    parser.add_argument(
        "--likelihood-source",
        choices=["auto", "ds", "real"],
        default="auto",
    )
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--final-eval-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--case-dim", type=int, default=0)
    parser.add_argument("--phyla-dim", type=int, default=256)
    parser.add_argument("--phyla-use-leaf-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phyla-use-split-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--relax-scale", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = str(out_dir / "metrics.jsonl")
    ckpt_path = str(out_dir / "best.pt")

    cfg = _small_model_config(args.base_config, args)
    model = return_model(cfg).to(device)
    module_like = SimpleNamespace(model=model)
    dataset_specs = list(args.dataset_spec)
    for spec_file in args.dataset_spec_file:
        dataset_specs.extend(_read_dataset_spec_file(spec_file))
    if args.sample_index_file:
        if dataset_specs:
            raise ValueError(
                "Use either --sample-index-file or --dataset-spec/--dataset-spec-file "
                "for training samples in this script"
            )
        samples = IndexedSampleBank(
            _read_sample_index_files(
                args.sample_index_file,
                stream_root=args.index_stream_root,
            ),
            module_like,
        )
    else:
        samples = _build_samples_from_specs(
            module_like,
            dataset_specs,
            (str(args.dataset_id).upper(), args.start_trees, args.target_trees),
        )
    eval_sets = []
    eval_specs = list(args.eval_spec)
    for spec_file in args.eval_spec_file:
        eval_specs.extend(_read_eval_spec_file(spec_file))
    for raw_spec in eval_specs:
        eval_name, dataset_id, eval_start_trees, eval_target_trees = _parse_eval_spec(raw_spec)
        eval_sets.append(
            (
                str(eval_name),
                _add_sample_metadata(
                    _build_branch_relax_samples_for_module(
                        module_like,
                        eval_start_trees,
                        eval_target_trees,
                    ),
                    dataset_id=dataset_id,
                    case_offset=len(samples) + sum(len(items) for _name, items in eval_sets),
                ),
            )
        )
    if args.eval_start_trees or args.eval_target_trees:
        if not (args.eval_start_trees and args.eval_target_trees):
            raise ValueError("--eval-start-trees and --eval-target-trees must be provided together")
        eval_sets.append(
            (
                str(args.eval_name),
                _add_sample_metadata(
                    _build_branch_relax_samples_for_module(
                        module_like,
                        args.eval_start_trees,
                        args.eval_target_trees,
                    ),
                    dataset_id=str(args.dataset_id).upper(),
                    case_offset=len(samples) + sum(len(items) for _name, items in eval_sets),
                ),
            )
        )
    if not (
        len(args.extra_eval_name)
        == len(args.extra_eval_start_trees)
        == len(args.extra_eval_target_trees)
    ):
        raise ValueError(
            "--extra-eval-name, --extra-eval-start-trees, and "
            "--extra-eval-target-trees must be repeated the same number of times"
        )
    for eval_name, eval_start_trees, eval_target_trees in zip(
        args.extra_eval_name,
        args.extra_eval_start_trees,
        args.extra_eval_target_trees,
    ):
        eval_sets.append(
            (
                str(eval_name),
                _add_sample_metadata(
                    _build_branch_relax_samples_for_module(
                        module_like,
                        eval_start_trees,
                        eval_target_trees,
                    ),
                    dataset_id=str(args.dataset_id).upper(),
                    case_offset=len(samples) + sum(len(items) for _name, items in eval_sets),
                ),
            )
        )
    head = BranchDeltaHead(
        int(model.embed_dim),
        hidden_dim=int(args.head_hidden_dim),
        case_dim=int(args.case_dim),
        num_cases=len(samples),
    ).to(device)
    relaxer = StandaloneRelaxer(model, head).to(device)
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        head.load_state_dict(checkpoint["head"])

    optimizer = torch.optim.AdamW(
        relaxer.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    if hasattr(samples, "dataset_ids"):
        dataset_ids = set(samples.dataset_ids)
    else:
        dataset_ids = {
            str(sample["dataset_id"]).upper()
            for sample in samples
        }
    for _eval_name, eval_samples in eval_sets:
        dataset_ids.update(str(sample["dataset_id"]).upper() for sample in eval_samples)
    scorers = LazyLikelihoodScorers(dataset_ids, args)
    phyla_bank = _load_phyla_embedding_bank(
        dataset_ids,
        args.phyla_embedding_dir,
        device,
    )
    rng = random.Random(args.seed + 17)
    best_after = -float("inf")

    print(
        json.dumps(
            {
                "event": "start",
                "num_samples": len(samples),
                "datasets": sorted(dataset_ids),
                "eval_sets": {
                    str(name): len(eval_samples)
                    for name, eval_samples in eval_sets
                },
                "sample_index_files": list(args.sample_index_file),
                "index_stream_root": str(args.index_stream_root),
                "device": str(device),
                "out_dir": str(out_dir),
                "case_dim": int(args.case_dim),
                "embed_dim": int(args.embed_dim),
                "n_layers": int(args.n_layers),
                "n_heads": int(args.n_heads),
                "phyla_dim": int(args.phyla_dim),
                "phyla_embedding_dir": str(args.phyla_embedding_dir),
                "real_nexus_root": str(args.real_nexus_root),
                "real_runs_root": str(args.real_runs_root),
                "likelihood_source": str(args.likelihood_source),
                "final_eval_count": (
                    None if args.final_eval_count is None else int(args.final_eval_count)
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.eval_only:
        eval_only_metrics = {"event": "eval_only", "timestamp": time.time()}
        if not bool(args.skip_train_eval):
            eval_only_metrics.update(
                _prefix_metrics(
                    "train_eval",
                    _evaluate(
                        relaxer,
                        samples,
                        scorers,
                        device,
                        eval_count=int(args.eval_count),
                        rng=rng,
                        scale=float(args.relax_scale),
                        phyla_bank=phyla_bank,
                    ),
                )
            )
        if eval_sets:
            for eval_name, eval_samples in eval_sets:
                eval_only_metrics.update(
                    _prefix_metrics(
                        str(eval_name),
                        _evaluate(
                            relaxer,
                            eval_samples,
                            scorers,
                            device,
                            eval_count=int(args.eval_count),
                            rng=rng,
                            scale=float(args.relax_scale),
                            phyla_bank=phyla_bank,
                        ),
                    )
                )
        else:
            eval_only_metrics.update(
                _evaluate(
                    relaxer,
                    samples,
                    scorers,
                    device,
                    eval_count=int(args.eval_count),
                    rng=rng,
                    scale=float(args.relax_scale),
                    phyla_bank=phyla_bank,
                )
            )
        _write_jsonl(metrics_path, eval_only_metrics)
        print(json.dumps(eval_only_metrics, sort_keys=True), flush=True)
        return

    samples_by_dataset = defaultdict(list)
    if hasattr(samples, "random_batch"):
        train_dataset_ids = list(samples.dataset_ids)
    else:
        for sample in samples:
            samples_by_dataset[str(sample["dataset_id"]).upper()].append(sample)
        train_dataset_ids = sorted(samples_by_dataset)

    for step in range(1, int(args.max_steps) + 1):
        relaxer.train()
        if hasattr(samples, "random_batch"):
            batch = samples.random_batch(
                rng,
                int(args.batch_size),
                balanced=bool(args.balanced_dataset_sampling),
            )
        elif bool(args.balanced_dataset_sampling) and train_dataset_ids:
            batch = []
            for _ in range(min(int(args.batch_size), len(samples))):
                dataset_id = rng.choice(train_dataset_ids)
                batch.append(rng.choice(samples_by_dataset[dataset_id]))
        else:
            batch = rng.sample(samples, min(int(args.batch_size), len(samples)))
        tokenized = model.tokenizer([sample["newick_tree"] for sample in batch])
        pred, target = relaxer.forward_batch(
            tokenized,
            [sample["newick_tree"] for sample in batch],
            batch,
            device,
            phyla_bank=phyla_bank,
        )
        if pred is None:
            continue
        loss = torch.mean((pred - target) ** 2)
        mae = torch.mean(torch.abs(pred - target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = None
        if float(args.grad_clip) > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                relaxer.parameters(),
                float(args.grad_clip),
            )
        optimizer.step()

        if step == 1 or step % int(args.eval_every) == 0:
            metrics = {
                "step": int(step),
                "timestamp": time.time(),
                "train_mse": float(loss.detach().cpu()),
                "train_mae": float(mae.detach().cpu()),
            }
            if grad_norm is not None:
                metrics["grad_norm_before_clip"] = float(grad_norm)
            metrics.update(
                _prefix_metrics(
                    "train_eval",
                    _evaluate(
                        relaxer,
                        samples,
                        scorers,
                        device,
                        eval_count=int(args.eval_count),
                        rng=rng,
                        scale=float(args.relax_scale),
                        phyla_bank=phyla_bank,
                    ),
                )
            )
            if eval_sets:
                for eval_name, eval_samples in eval_sets:
                    metrics.update(
                        _prefix_metrics(
                            str(eval_name),
                            _evaluate(
                                relaxer,
                                eval_samples,
                                scorers,
                                device,
                                eval_count=int(args.eval_count),
                                rng=rng,
                                scale=float(args.relax_scale),
                                phyla_bank=phyla_bank,
                            ),
                        )
                    )
            else:
                metrics.update(
                    _evaluate(
                        relaxer,
                        samples,
                        scorers,
                        device,
                        eval_count=int(args.eval_count),
                        rng=rng,
                        scale=float(args.relax_scale),
                        phyla_bank=phyla_bank,
                    )
                )
            _write_jsonl(metrics_path, metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if args.score_prefixes:
                score_keys = [
                    f"{prefix.strip()}_after_log_likelihood_mean"
                    for prefix in str(args.score_prefixes).split(",")
                    if prefix.strip()
                ]
                score_value = float(np.mean([metrics[key] for key in score_keys]))
            elif args.score_prefix:
                score_key = f"{args.score_prefix}_after_log_likelihood_mean"
                score_value = metrics[score_key]
            elif eval_sets:
                score_key = f"{eval_sets[0][0]}_after_log_likelihood_mean"
                score_value = metrics[score_key]
            else:
                score_key = "after_log_likelihood_mean"
                score_value = metrics[score_key]
            if score_value > best_after:
                best_after = score_value
                torch.save(
                    {
                        "model": model.state_dict(),
                        "head": head.state_dict(),
                        "args": vars(args),
                        "metrics": metrics,
                    },
                    ckpt_path,
                )

    if args.final_eval_count is None:
        train_final_eval_count = int(args.eval_count) if hasattr(samples, "materialize_eval_samples") else 0
        extra_final_eval_count = 0
    else:
        train_final_eval_count = int(args.final_eval_count)
        extra_final_eval_count = int(args.final_eval_count)

    final_metrics = {}
    final_metrics.update(
        _prefix_metrics(
            "train_eval",
            _evaluate(
                relaxer,
                samples,
                scorers,
                device,
                eval_count=train_final_eval_count,
                rng=rng,
                scale=float(args.relax_scale),
                phyla_bank=phyla_bank,
            ),
        )
    )
    if eval_sets:
        for eval_name, eval_samples in eval_sets:
            final_metrics.update(
                _prefix_metrics(
                    str(eval_name),
                    _evaluate(
                        relaxer,
                        eval_samples,
                        scorers,
                        device,
                        eval_count=extra_final_eval_count,
                        rng=rng,
                        scale=float(args.relax_scale),
                        phyla_bank=phyla_bank,
                    )
                )
            )
    else:
        final_metrics.update(
            _evaluate(
                relaxer,
                samples,
                scorers,
                device,
                eval_count=train_final_eval_count,
                rng=rng,
                scale=float(args.relax_scale),
                phyla_bank=phyla_bank,
            )
        )
    final_metrics.update({"step": int(args.max_steps), "event": "final", "timestamp": time.time()})
    _write_jsonl(metrics_path, final_metrics)
    print(json.dumps(final_metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
