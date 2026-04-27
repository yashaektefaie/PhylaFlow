# PhylaFlow
Code for PhylaFlow

## DS1-DS8 Fixed-Bank Experiments

The DS fixed-bank recipes in `configs/` use the local DS artifact bundle at
`/n/holylfs06/LABS/mzitnik_lab/Users/yektefaie/DS_data/30272299` and write
checkpoints, metrics, and W&B files to
`/n/netscratch/mzitnik_lab/Lab/yektefaie/phylaflow`.  Use positional config
launches, for example:

```bash
python -m run.run configs/ds1_caseadapt_arcase64_anchors_20260427.yaml
```

For Slurm on the Kempner GPU queues, the shared runner is:

```bash
sbatch slurm/run_ds_24h.sbatch configs/ds1_caseadapt_arcase64_anchors_20260427.yaml
```

`submit_ds_24h.sh` and `submit_ds1_ds8_24h.sh` are thin wrappers for the
baseline DS1-DS8 current-recipe configs.

### Checkpoint Compatibility

The DS1 best checkpoint exported to
`gs://phyla/30272299/DS1/best_04_27/best_topological_kl_step031000.ckpt`
uses the AR case-conditioning code path.  Agents loading that checkpoint should
use this repository version or newer, and should instantiate it with the matching
config:

```text
configs/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptboth_arref1_arcase_6000_currentrecipe_20260426.yaml
```

The important model/trainer knobs for that checkpoint are:

```yaml
model:
  first_hit_head_mode: case_adapted_mlp
  first_hit_head_num_cases: 234
  autoregressive_head_mode: structured_subset
  autoregressive_use_case_conditioning: true
  autoregressive_num_cases: 234
  autoregressive_case_dim: 16
  autoregressive_group_refinement_layers: 1
trainer:
  velocity_terminal_head_weight: 0.0
```

### Experiment Branches

The main DS1 experimental configs added here are:

- `configs/ds1_caseadapt_arcase64_anchors_20260427.yaml`: small `64x2` model,
  first-hit case adaptation, AR case conditioning with `autoregressive_case_dim:
  64`, full-path anchors, terminal-head loss off.
- `configs/ds1_caseadapt_arref1_arcase64_anchors_20260427.yaml`: same as above
  plus one autoregressive refinement layer.
- `configs/ds1_arcase64_scale96x3_lr2e3_20260427.yaml` and
  `configs/ds1_arcase64_scale128x4_lr2e3_20260427.yaml`: scaled trunk variants
  of the AR-case64 branch with `lr: 0.002`.
- Matching `lr1e3` scaling configs are included as negative controls; early
  runs showed `lr: 0.001` was too low for these larger models.
- The `noanchors` configs remove
  `data.overfit_full_path_control_extra_velocity_samples_json_path` to test the
  value of the extra full-path anchor samples.

Additional first-hit architecture probes are also available:

- `first_hit_head_enable_refinement: true` lets
  `case_adapted_mlp` use first-hit edge refinement blocks without switching to
  the legacy `edge_refined_mlp` mode.
- `first_hit_head_mode: start_topology_raw_pool_concat_mlp` appends raw
  start-topology `sum/mean/max` pooled split features to each first-hit edge
  input.
- `first_hit_head_mode: start_tree_graph_token_mlp` appends a graph-token
  encoding of the start tree to each first-hit edge input.  Set
  `first_hit_start_tree_graph_detach: true` to stop gradients through that start
  tree encoding.

The most useful DS1 result so far was AR case conditioning.  The `arcase64`
branch improved the small-model DS1 topological tree KL, while the larger
`128x4` branch with `lr: 0.002` gave the first credible scaling signal.  The
first-hit structural replacements were mostly weaker than case adaptation.
