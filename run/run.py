from model.model import return_model
from data.dataset import PhylaDataModule
import yaml
import sys
from utils.utils import get_possible_ids
from run.TrainingModule import TrainingModule
from utils.random_tree import Tree
import random
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import multiprocessing
import os
import torch
from datetime import datetime

import logging

# Global variables to hold the model in worker processes
worker_model = None


def _set_global_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_torch_runtime():
    if not torch.cuda.is_available():
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True


def _init_wandb_run(config, default_project):
    trainer_cfg = config.get("trainer", {})
    wandb_kwargs = {
        "project": trainer_cfg.get("wandb_project", default_project),
        "name": trainer_cfg.get("wandb_name"),
        "group": trainer_cfg.get("wandb_group"),
        "job_type": trainer_cfg.get("wandb_job_type"),
        "notes": trainer_cfg.get("wandb_notes"),
        "tags": trainer_cfg.get("wandb_tags"),
        "config": {
            "seed": trainer_cfg.get("seed"),
            "epochs": trainer_cfg.get("epochs"),
            "training_step_autoregressive_weight": trainer_cfg.get(
                "training_step_autoregressive_weight"
            ),
            "training_step_velocity_weight": trainer_cfg.get(
                "training_step_velocity_weight"
            ),
            "training_step_autoregressive_grad_ratio": trainer_cfg.get(
                "training_step_autoregressive_grad_ratio"
            ),
            "autoregressive_use_time": trainer_cfg.get("autoregressive_use_time"),
            "autoregressive_target_mode": trainer_cfg.get(
                "autoregressive_target_mode"
            ),
            "autoregressive_rollin_prob": trainer_cfg.get(
                "autoregressive_rollin_prob"
            ),
            "autoregressive_dagger_prob": trainer_cfg.get(
                "autoregressive_dagger_prob"
            ),
            "autoregressive_dagger_max_steps": trainer_cfg.get(
                "autoregressive_dagger_max_steps"
            ),
            "autoregressive_structure_perturb_prob": trainer_cfg.get(
                "autoregressive_structure_perturb_prob"
            ),
            "autoregressive_structure_perturb_mode": trainer_cfg.get(
                "autoregressive_structure_perturb_mode"
            ),
            "velocity_length_jitter_prob": trainer_cfg.get(
                "velocity_length_jitter_prob"
            ),
            "velocity_length_jitter_scale": trainer_cfg.get(
                "velocity_length_jitter_scale"
            ),
            "training_sampling_start": trainer_cfg.get("training_sampling_start"),
            "training_sampling_frequency": trainer_cfg.get(
                "training_sampling_frequency"
            ),
        },
    }
    wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
    return wandb.init(**wandb_kwargs)


def init_worker(config_file, device_id):
    """
    Initializer for worker processes. Loads the model once.
    """
    global worker_model

    # Silence detailed logs in workers
    logging.getLogger("run.TrainingModule").setLevel(logging.WARNING)
    logging.getLogger("phyla").setLevel(logging.WARNING)

    # Load config
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    _configure_torch_runtime()
    _set_global_seed(config["trainer"].get("seed"))

    # Initialize Dataset (needed for embeddings calculation in sample)
    ids = get_possible_ids(config["data"]["nexus_root"])
    ran = random.Random(42)
    ran.shuffle(ids)
    train_ids = ids[: int(0.8 * len(ids))]
    test_ids = ids[int(0.8 * len(ids)) :]

    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)

    # Initialize Model
    phyla_flow = return_model(config)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    phyla_flow.to(device)

    model = TrainingModule(
        model=phyla_flow,
        lr=config["trainer"]["lr"],
        record=config["trainer"]["record"],
        epochs=config["trainer"]["epochs"],
        dataset=dataset,
        lr_scheduler="default",
        num_annealing_steps=10000,
        num_warmup_steps=1000,
        deepspeed=False,
        logger=None,
        phyla_checkpoint_path=config["trainer"].get("phyla_checkpoint_path"),
        velocity_loss_mode=config["trainer"].get("velocity_loss_mode", "weighted"),
        velocity_loss_plain_weight=config["trainer"].get(
            "velocity_loss_plain_weight", 0.5
        ),
        velocity_sign_eps=config["trainer"].get("velocity_sign_eps", 1e-3),
        training_step_velocity_weight=config["trainer"].get(
            "training_step_velocity_weight", 1.0
        ),
        training_step_autoregressive_weight=config["trainer"].get(
            "training_step_autoregressive_weight", 1.0
        ),
        training_step_autoregressive_grad_ratio=config["trainer"].get(
            "training_step_autoregressive_grad_ratio"
        ),
        autoregressive_use_time=config["trainer"].get(
            "autoregressive_use_time", False
        ),
        autoregressive_target_mode=config["trainer"].get(
            "autoregressive_target_mode", "scheduled"
        ),
        autoregressive_rollin_prob=config["trainer"].get(
            "autoregressive_rollin_prob", 0.0
        ),
        autoregressive_dagger_prob=config["trainer"].get(
            "autoregressive_dagger_prob", 0.0
        ),
        autoregressive_dagger_max_steps=config["trainer"].get(
            "autoregressive_dagger_max_steps", 4
        ),
        autoregressive_structure_perturb_prob=config["trainer"].get(
            "autoregressive_structure_perturb_prob", 0.0
        ),
        autoregressive_structure_perturb_mode=config["trainer"].get(
            "autoregressive_structure_perturb_mode", "random_wrong_pair"
        ),
        velocity_length_jitter_prob=config["trainer"].get(
            "velocity_length_jitter_prob", 0.0
        ),
        velocity_length_jitter_scale=config["trainer"].get(
            "velocity_length_jitter_scale", 0.0
        ),
        velocity_dt_candidate_weight=config["trainer"].get(
            "velocity_dt_candidate_weight", 0.0
        ),
        velocity_dt_hit_weight=config["trainer"].get("velocity_dt_hit_weight", 0.0),
        velocity_dt_eps=config["trainer"].get("velocity_dt_eps", 1e-6),
        velocity_event_weight=config["trainer"].get("velocity_event_weight", 0.5),
        velocity_event_temp=config["trainer"].get("velocity_event_temp", 0.5),
        velocity_event_rate_beta=config["trainer"].get("velocity_event_rate_beta", 5.0),
        velocity_event_normalize_by_log_candidates=config["trainer"].get(
            "velocity_event_normalize_by_log_candidates", True
        ),
        training_sampling_mode=config["trainer"].get(
            "training_sampling_mode", "batch_compare"
        ),
        training_sampling_dt_base=config["trainer"].get(
            "training_sampling_dt_base", 0.02
        ),
        sample_metrics_trace_path=config["trainer"].get("sample_metrics_trace_path"),
    )
    model.to(device)
    model.eval()

    worker_model = model


def sample_worker_task(tree_list):
    """
    Task function called by pool.map. Uses the global worker_model.
    """
    global worker_model
    # dt_base set to 0.01 to force approx 100 steps (T=1.0)
    return worker_model.sample(tree_list, None, num_samples=1, dt_base=0.01)


def run_test():
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    _configure_torch_runtime()
    _set_global_seed(config["trainer"].get("seed"))

    ids = get_possible_ids(config["data"]["nexus_root"])
    # Random 80-20 train-test split for now
    ran = random.Random(42)
    ran.shuffle(ids)
    train_ids = ids[: int(0.8 * len(ids))]
    test_ids = ids[int(0.8 * len(ids)) :]
    ###TEMPORARY FOR DEBUGGING
    train_ids = test_ids

    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)
    one = dataset.dataset_train[0]
    two = dataset.dataset_train[0]
    batch = dataset.collate_fn([one, two])

    phyla_flow = return_model(config)

    model = TrainingModule(
        model=phyla_flow,
        lr=config["trainer"]["lr"],
        record=config["trainer"]["record"],
        epochs=config["trainer"]["epochs"],
        dataset=dataset,
        lr_scheduler="default",
        num_annealing_steps=10000,
        num_warmup_steps=1000,
        deepspeed=False,
        logger=None,
        phyla_checkpoint_path=config["trainer"].get("phyla_checkpoint_path"),
        velocity_loss_mode=config["trainer"].get("velocity_loss_mode", "weighted"),
        velocity_loss_plain_weight=config["trainer"].get(
            "velocity_loss_plain_weight", 0.5
        ),
        velocity_sign_eps=config["trainer"].get("velocity_sign_eps", 1e-3),
        training_step_velocity_weight=config["trainer"].get(
            "training_step_velocity_weight", 1.0
        ),
        training_step_autoregressive_weight=config["trainer"].get(
            "training_step_autoregressive_weight", 1.0
        ),
        training_step_autoregressive_grad_ratio=config["trainer"].get(
            "training_step_autoregressive_grad_ratio"
        ),
        autoregressive_use_time=config["trainer"].get(
            "autoregressive_use_time", False
        ),
        autoregressive_target_mode=config["trainer"].get(
            "autoregressive_target_mode", "scheduled"
        ),
        autoregressive_rollin_prob=config["trainer"].get(
            "autoregressive_rollin_prob", 0.0
        ),
        autoregressive_dagger_prob=config["trainer"].get(
            "autoregressive_dagger_prob", 0.0
        ),
        autoregressive_dagger_max_steps=config["trainer"].get(
            "autoregressive_dagger_max_steps", 4
        ),
        autoregressive_structure_perturb_prob=config["trainer"].get(
            "autoregressive_structure_perturb_prob", 0.0
        ),
        autoregressive_structure_perturb_mode=config["trainer"].get(
            "autoregressive_structure_perturb_mode", "random_wrong_pair"
        ),
        velocity_length_jitter_prob=config["trainer"].get(
            "velocity_length_jitter_prob", 0.0
        ),
        velocity_length_jitter_scale=config["trainer"].get(
            "velocity_length_jitter_scale", 0.0
        ),
        velocity_dt_candidate_weight=config["trainer"].get(
            "velocity_dt_candidate_weight", 0.0
        ),
        velocity_dt_hit_weight=config["trainer"].get("velocity_dt_hit_weight", 0.0),
        velocity_dt_eps=config["trainer"].get("velocity_dt_eps", 1e-6),
        velocity_event_weight=config["trainer"].get("velocity_event_weight", 0.5),
        velocity_event_temp=config["trainer"].get("velocity_event_temp", 0.5),
        velocity_event_rate_beta=config["trainer"].get("velocity_event_rate_beta", 5.0),
        velocity_event_normalize_by_log_candidates=config["trainer"].get(
            "velocity_event_normalize_by_log_candidates", True
        ),
        training_sampling_mode=config["trainer"].get(
            "training_sampling_mode", "batch_compare"
        ),
        training_sampling_dt_base=config["trainer"].get(
            "training_sampling_dt_base", 0.02
        ),
        sample_metrics_trace_path=config["trainer"].get("sample_metrics_trace_path"),
    )
    # res = model(batch['tokenized_trees'], batch['batched_time'], batch['phyla_embeddings'])
    # This fails now btw non-autoregressive LOL NEED TO FIX!

    # Initialize wandb (mock) or disable logging in step for test
    import wandb

    wandb.init(mode="disabled")

    res = model.step(batch)

    # This works below
    # res = model.step(batch, autoregressive=True)
    rt = Tree(num_leaves=50, random=True)
    import time

    start = time.time()
    num_trees = 100
    trees_to_sample = []
    print(f"Generating {num_trees} trees...")
    for _ in range(num_trees):
        trees_to_sample.append(str(Tree(num_leaves=50, random=True)))

    print("Sampling with Multiprocessing (Initialized Workers)...")

    # Split into chunks of size 2
    batch_size_per_worker = 2
    chunks = [
        trees_to_sample[i : i + batch_size_per_worker]
        for i in range(0, len(trees_to_sample), batch_size_per_worker)
    ]

    # We use 0 as device_id assuming single GPU
    # Use 'spawn' context usually safer for CUDA
    ctx = multiprocessing.get_context("spawn")
    num_workers = min(4, os.cpu_count())

    pool_start = time.time()
    try:
        # Pass initializer to set up model ONCE per worker
        with ctx.Pool(
            num_workers, initializer=init_worker, initargs=(config_file, 0)
        ) as pool:
            results = pool.map(sample_worker_task, chunks)
    except Exception as e:
        print(f"Multiprocessing failed: {e}. Falling back to sequential.")
        final_tree = model.sample(trees_to_sample, None, num_samples=1, dt_base=0.1)
        results = [final_tree]
    pool_end = time.time()
    print(f"Pool execution time: {pool_end - pool_start}")

    # Flatten results
    final_tree = []
    for r in results:
        final_tree.extend(r)

    # final_tree = model.sample(trees_to_sample, None, num_samples=1, dt_base=0.1)
    res = time.time() - start

    print("Sampling time:", res)
    print(
        f"Sampling time for a million trees in seconds:",
        res * 1e6 / num_trees,
        " in days:",
        res * 1e6 / num_trees / 86400,
    )
    print(
        f"Sampling time for a thousand trees in seconds:",
        res * 1e3 / num_trees,
        " in minutes:",
        res * 1e3 / num_trees / 60,
    )


def run_overfit():
    # Get first command line argument as config file
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    _configure_torch_runtime()
    _set_global_seed(config["trainer"].get("seed"))

    # --- Start Overrides ---
    print("Running OVERFIT mode (Single Tree)")
    config["data"]["batch_size"] = 1
    config["data"]["num_workers"] = 0
    # config["trainer"]["epochs"] = 10000  # Train for a long time
    config["trainer"]["val_callback_freq"] = 0
    # config["trainer"]["record"] = True # Optional: force recording
    # --- End Overrides ---

    ids = get_possible_ids(config["data"]["nexus_root"])

    if not ids:
        print("No IDs found!")
        return

    # Override IDs to just one (deterministic)
    single_id = sorted(ids)[0]
    train_ids = [single_id]
    test_ids = [single_id]
    print(f"Overfitting on ID: {single_id}")

    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)

    phyla_flow = return_model(config)

    model = TrainingModule(
        model=phyla_flow,
        lr=config["trainer"]["lr"],
        record=config["trainer"]["record"],
        epochs=config["trainer"]["epochs"],
        dataset=dataset,
        lr_scheduler="default",
        num_annealing_steps=10000,
        num_warmup_steps=config["trainer"].get("num_warmup_steps", 1000),
        num_samples=1,
        deepspeed=False,
        logger=None,
        phyla_checkpoint_path=config["trainer"].get("phyla_checkpoint_path"),
        velocity_loss_mode=config["trainer"].get("velocity_loss_mode", "weighted"),
        velocity_loss_plain_weight=config["trainer"].get(
            "velocity_loss_plain_weight", 0.5
        ),
        velocity_sign_eps=config["trainer"].get("velocity_sign_eps", 1e-3),
        training_step_velocity_weight=config["trainer"].get(
            "training_step_velocity_weight", 1.0
        ),
        training_step_autoregressive_weight=config["trainer"].get(
            "training_step_autoregressive_weight", 1.0
        ),
        training_step_autoregressive_grad_ratio=config["trainer"].get(
            "training_step_autoregressive_grad_ratio"
        ),
        autoregressive_use_time=config["trainer"].get(
            "autoregressive_use_time", False
        ),
        autoregressive_target_mode=config["trainer"].get(
            "autoregressive_target_mode", "scheduled"
        ),
        autoregressive_rollin_prob=config["trainer"].get(
            "autoregressive_rollin_prob", 0.0
        ),
        autoregressive_dagger_prob=config["trainer"].get(
            "autoregressive_dagger_prob", 0.0
        ),
        autoregressive_dagger_max_steps=config["trainer"].get(
            "autoregressive_dagger_max_steps", 4
        ),
        autoregressive_structure_perturb_prob=config["trainer"].get(
            "autoregressive_structure_perturb_prob", 0.0
        ),
        autoregressive_structure_perturb_mode=config["trainer"].get(
            "autoregressive_structure_perturb_mode", "random_wrong_pair"
        ),
        velocity_length_jitter_prob=config["trainer"].get(
            "velocity_length_jitter_prob", 0.0
        ),
        velocity_length_jitter_scale=config["trainer"].get(
            "velocity_length_jitter_scale", 0.0
        ),
        velocity_dt_candidate_weight=config["trainer"].get(
            "velocity_dt_candidate_weight", 0.0
        ),
        velocity_dt_hit_weight=config["trainer"].get("velocity_dt_hit_weight", 0.0),
        velocity_dt_eps=config["trainer"].get("velocity_dt_eps", 1e-6),
        velocity_event_weight=config["trainer"].get("velocity_event_weight", 0.5),
        velocity_event_temp=config["trainer"].get("velocity_event_temp", 0.5),
        velocity_event_rate_beta=config["trainer"].get("velocity_event_rate_beta", 5.0),
        velocity_event_normalize_by_log_candidates=config["trainer"].get(
            "velocity_event_normalize_by_log_candidates", True
        ),
        training_sampling_frequency=config["trainer"].get(
            "training_sampling_frequency", 200
        ),
        training_sampling_start=config["trainer"].get(
            "training_sampling_start", 500
        ),
        training_sampling_mode=config["trainer"].get(
            "training_sampling_mode", "harness_sanity"
        ),
        training_sampling_dt_base=config["trainer"].get(
            "training_sampling_dt_base", 0.02
        ),
        dt=config["trainer"].get("dt", 0.1),
        sample_metrics_trace_path=config["trainer"].get("sample_metrics_trace_path"),
        verbose=True,  # Enable verbose logging for overfitting
    )

    checkpoint_base = config["trainer"]["checkpoint_dir"]
    checkpoint_dir = os.path.join(checkpoint_base, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Saving checkpoints to: {checkpoint_dir}")

    save_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="overfit-{epoch:02d}",
        every_n_epochs=50,
        save_top_k=-1,
    )

    trainer_args = {}
    if config["trainer"]["record"]:
        _init_wandb_run(config, default_project="phylaflow_overfit")
    else:
        trainer_args["logger"] = False

    trainer_args["max_epochs"] = config["trainer"]["epochs"]
    trainer_args["callbacks"] = [save_callback]

    # Log frequently for overfitting
    trainer_args["log_every_n_steps"] = 1

    if config["trainer"]["val_callback_freq"] != 0:
        trainer_args["val_check_interval"] = config["trainer"]["val_callback_freq"]

    trainer_args["accelerator"] = "gpu"
    trainer_args["devices"] = 1

    trainer = Trainer(**trainer_args)
    trainer.fit(
        model,
        train_dataloaders=dataset.train_dataloader(),
        val_dataloaders=dataset.val_dataloader(),
    )


def main():
    # Get first command line argument as config file
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    _configure_torch_runtime()
    _set_global_seed(config["trainer"].get("seed"))

    ids = get_possible_ids(config["data"]["nexus_root"])
    # Random 80-20 train-test split for now
    ran = random.Random(42)
    ran.shuffle(ids)
    if len(ids) < 2:
        train_ids = ids
        test_ids = ids
    else:
        train_ids = ids[: int(0.8 * len(ids))]
        test_ids = ids[int(0.8 * len(ids)) :]

    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)

    phyla_flow = return_model(config)

    model = TrainingModule(
        model=phyla_flow,
        lr=config["trainer"]["lr"],
        record=config["trainer"]["record"],
        epochs=config["trainer"]["epochs"],
        dataset=dataset,
        lr_scheduler="default",
        num_annealing_steps=10000,
        num_warmup_steps=1000,
        deepspeed=False,
        logger=None,
        phyla_checkpoint_path=config["trainer"].get("phyla_checkpoint_path"),
        velocity_loss_mode=config["trainer"].get("velocity_loss_mode", "weighted"),
        velocity_loss_plain_weight=config["trainer"].get(
            "velocity_loss_plain_weight", 0.5
        ),
        velocity_sign_eps=config["trainer"].get("velocity_sign_eps", 1e-3),
        training_step_velocity_weight=config["trainer"].get(
            "training_step_velocity_weight", 1.0
        ),
        training_step_autoregressive_weight=config["trainer"].get(
            "training_step_autoregressive_weight", 1.0
        ),
        training_step_autoregressive_grad_ratio=config["trainer"].get(
            "training_step_autoregressive_grad_ratio"
        ),
        autoregressive_use_time=config["trainer"].get(
            "autoregressive_use_time", False
        ),
        autoregressive_target_mode=config["trainer"].get(
            "autoregressive_target_mode", "scheduled"
        ),
        autoregressive_rollin_prob=config["trainer"].get(
            "autoregressive_rollin_prob", 0.0
        ),
        autoregressive_dagger_prob=config["trainer"].get(
            "autoregressive_dagger_prob", 0.0
        ),
        autoregressive_dagger_max_steps=config["trainer"].get(
            "autoregressive_dagger_max_steps", 4
        ),
        autoregressive_structure_perturb_prob=config["trainer"].get(
            "autoregressive_structure_perturb_prob", 0.0
        ),
        autoregressive_structure_perturb_mode=config["trainer"].get(
            "autoregressive_structure_perturb_mode", "random_wrong_pair"
        ),
        velocity_length_jitter_prob=config["trainer"].get(
            "velocity_length_jitter_prob", 0.0
        ),
        velocity_length_jitter_scale=config["trainer"].get(
            "velocity_length_jitter_scale", 0.0
        ),
        velocity_dt_candidate_weight=config["trainer"].get(
            "velocity_dt_candidate_weight", 0.0
        ),
        velocity_dt_hit_weight=config["trainer"].get("velocity_dt_hit_weight", 0.0),
        velocity_dt_eps=config["trainer"].get("velocity_dt_eps", 1e-6),
        velocity_event_weight=config["trainer"].get("velocity_event_weight", 0.5),
        velocity_event_temp=config["trainer"].get("velocity_event_temp", 0.5),
        velocity_event_rate_beta=config["trainer"].get("velocity_event_rate_beta", 5.0),
        velocity_event_normalize_by_log_candidates=config["trainer"].get(
            "velocity_event_normalize_by_log_candidates", True
        ),
        training_sampling_frequency=config["trainer"].get(
            "training_sampling_frequency", 200
        ),
        training_sampling_start=config["trainer"].get(
            "training_sampling_start", 500
        ),
        training_sampling_mode=config["trainer"].get(
            "training_sampling_mode", "batch_compare"
        ),
        training_sampling_dt_base=config["trainer"].get(
            "training_sampling_dt_base", 0.02
        ),
        dt=config["trainer"].get("dt", 0.1),
        sample_metrics_trace_path=config["trainer"].get("sample_metrics_trace_path"),
    )

    checkpoint_base = config["trainer"]["checkpoint_dir"]
    checkpoint_dir = os.path.join(checkpoint_base, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Saving checkpoints to: {checkpoint_dir}")

    save_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{step:06d}",  # Include metric value in the filename
        every_n_train_steps=config["trainer"]["steps_callback"],  # Save every N steps
        save_top_k=-1,  # Save all checkpoints
    )

    trainer_args = {}
    if config["trainer"]["record"]:
        _init_wandb_run(config, default_project="phylaflow")
    else:
        trainer_args["logger"] = False

    trainer_args["max_epochs"] = config["trainer"]["epochs"]
    trainer_args["callbacks"] = [save_callback]  # For validation callback runs
    if config["trainer"]["val_callback_freq"] != 0:
        trainer_args["val_check_interval"] = config["trainer"]["val_callback_freq"]
    if config["trainer"]["limit_val_batches"] == 0:
        trainer_args["limit_val_batches"] = 0.0  # Disable validation

    trainer_args["accelerator"] = "gpu"
    trainer = Trainer(**trainer_args)
    trainer.fit(
        model,
        train_dataloaders=dataset.train_dataloader(),
        val_dataloaders=dataset.val_dataloader(),
    )


if __name__ == "__main__":
    main()
