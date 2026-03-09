#!/bin/bash
#SBATCH -c 1
#SBATCH -t 8:00:00
#SBATCH --mem=50G

#SBATCH -p kempner
#SBATCH --account kempner_mzitnik_lab
#SBATCH --gres=gpu:1

#SBATCH -o  logs/3_8/experiment_1_0_weight.out
#SBATCH -e logs/3_8/experiment_1_0_weight.err

module load cuda/12.4
python -m run.run configs/sanity_train.yaml
