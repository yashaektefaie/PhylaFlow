from model.model import return_model
from dataset.dataset import PhylaDataModule
import yaml
import sys



def main():
    #Get first command line argument as config file
    config_file = sys.argv[1]

    dataset = PhylaDataModule(config)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    phyla_flow = return_model(config)





if __name__ == "__main__":
    main()