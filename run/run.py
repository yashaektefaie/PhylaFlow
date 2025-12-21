from model.model import return_model
from data.dataset import PhylaDataModule
import yaml
import sys
from utils.utils import get_possible_ids
from run.TrainingModule import TrainingModule
import random
import wandb

def run_test():
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    ids = get_possible_ids(config['data']['nexus_root'])
    #Random 80-20 train-test split for now
    ran = random.Random(42)
    ran.shuffle(ids)
    train_ids = ids[:int(0.8*len(ids))]
    test_ids = ids[int(0.8*len(ids)):]
    ###TEMPORARY FOR DEBUGGING
    train_ids = test_ids

    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)
    one = dataset.dataset_train[0]
    two = dataset.dataset_train[0]
    batch = dataset.collate_fn([one, two])

    phyla_flow = return_model(config)

    model = TrainingModule(
        model=phyla_flow,
        lr=config['trainer']['lr'],
        record=config['trainer']['record'],
        epochs=config['trainer']['epochs'],
        dataset=dataset,
        lr_scheduler = 'default',
        num_annealing_steps = 10000,
        num_warmup_steps = 1000,
        deepspeed = False,
        logger = None
    )
    res = model(batch['tokenized_trees'], batch['batched_time'], batch['phyla_embeddings'])
    import pdb; pdb.set_trace()


def main():
    #Get first command line argument as config file
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    ids = get_possible_ids(config['data']['nexus_root'])
    #Random 80-20 train-test split for now
    ran = random.Random(42)
    ran.shuffle(ids)
    train_ids = ids[:int(0.8*len(ids))]
    test_ids = ids[int(0.8*len(ids)):]

    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)
    
    phyla_flow = return_model(config)

    model = TrainingModule(
        model=phyla_flow,
        lr=config['trainer']['lr'],
        record=config['trainer']['record'],
        epochs=config['trainer']['epochs'],
        dataset=dataset,
        lr_scheduler = 'default',
        num_annealing_steps = 10000,
        num_warmup_steps = 1000,
        deepspeed = False,
        logger = None
    )

    save_callback = ModelCheckpoint(
        dirpath=config.trainer.checkpoint_dir,
        filename="{epoch:02d}-{step:06d}",  # Include metric value in the filename
        every_n_train_steps=config.trainer.steps_callback,  # Save every N steps
        save_top_k=-1  # Save all checkpoints
    )


    trainer_args = {}
    if config.trainer.record:

        wandb.init(project = 'phylaflow', 
        group = f'{run_name}')
        wandb.watch(model, log_freq=100)

    
    trainer_args['max_epochs'] = config.trainer.epochs
    trainer_args['callbacks'] = [save_callback] # For validation callback runs
    if config.trainer.val_callback_freq != 0:
        trainer_args['val_check_interval'] = config.trainer.val_callback_freq

    trainer_args['accelerator'] = "gpu"
    trainer = Trainer(**trainer_args)
    #trainer.fit(model, train_dataloaders=dataset.train_dataloader(), val_dataloaders=dataset.val_dataloader())


if __name__ == "__main__":
    # main()
    run_test()