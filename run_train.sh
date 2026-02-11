#!/bin/bash
#SBATCH -c 1
#SBATCH -t 4:00:00
#SBATCH --mem=50G

#SBATCH -p kempner_h100
#SBATCH --account kempner_mzitnik_lab
#SBATCH --gres=gpu:1

#SBATCH -o  logs/2_9/initial_run_2.out
#SBATCH -e logs/2_9/initial_run_2.err

module load cuda/12.4
python -m run.run configs/train.yaml
