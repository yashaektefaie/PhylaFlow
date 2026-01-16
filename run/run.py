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


def run_test():
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

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
    )
    # res = model(batch['tokenized_trees'], batch['batched_time'], batch['phyla_embeddings'])
    # This fails now btw non-autoregressive LOL NEED TO FIX!
    res = model.step(batch)

    # This works below
    # res = model.step(batch, autoregressive=True)
    rt = Tree(num_leaves=50, random=True)
    import time

    start = time.time()
    num_trees = 2
    trees_to_sample = []
    for _ in range(num_trees):
        trees_to_sample.append(str(Tree(num_leaves=50, random=True)))
    final_tree = model.sample(trees_to_sample, None, num_samples=1, dt_base=0.1)
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


def main():
    # Get first command line argument as config file
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

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
    )

    save_callback = ModelCheckpoint(
        dirpath=config["trainer"]["checkpoint_dir"],
        filename="{epoch:02d}-{step:06d}",  # Include metric value in the filename
        every_n_train_steps=config["trainer"]["steps_callback"],  # Save every N steps
        save_top_k=-1,  # Save all checkpoints
    )

    trainer_args = {}
    if config["trainer"]["record"]:
        run_name = "test_run"
        wandb.init(project="phylaflow", group=f"{run_name}")
        wandb.watch(model, log_freq=100)

    trainer_args["max_epochs"] = config["trainer"]["epochs"]
    trainer_args["callbacks"] = [save_callback]  # For validation callback runs
    if config["trainer"]["val_callback_freq"] != 0:
        trainer_args["val_check_interval"] = config["trainer"]["val_callback_freq"]

    trainer_args["accelerator"] = "gpu"
    trainer = Trainer(**trainer_args)
    trainer.fit(
        model,
        train_dataloaders=dataset.train_dataloader(),
        val_dataloaders=dataset.val_dataloader(),
    )


if __name__ == "__main__":
    # main()
    # run_test()
