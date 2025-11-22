from model.model import return_model
from dataset.dataset import PhylaDataModule
import yaml
import sys
from utils.utils import get_possible_ids



def main():
    #Get first command line argument as config file
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    ids = get_possible_ids(config['data']['nexus_root'])
    #Random 80-20 train-test split for now
    random.shuffle(ids)
    train_ids = ids[:int(0.8*len(ids))]
    test_ids = ids[int(0.8*len(ids)):]

    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)

    phyla_flow = return_model(config)





if __name__ == "__main__":
    main()