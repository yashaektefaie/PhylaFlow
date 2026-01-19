#!/bin/bash
#SBATCH -c 1
#SBATCH -t 12:00:00
#SBATCH --mem=50G

#SBATCH -p kempner
#SBATCH --account kempner_mzitnik_lab
#SBATCH --gres=gpu:1

#SBATCH -o  logs/1_19/initial_run.out
#SBATCH -e logs/1_19/initial_run.err

module load cuda/12.4
python -m run.run configs/train.yaml
