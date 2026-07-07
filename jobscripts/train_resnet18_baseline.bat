#!/bin/bash --login
#SBATCH --job-name=train_resnet18
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --output=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.out
#SBATCH --error=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.err
#SBATCH --ntasks 1
#SBATCH -t 0-3

set -eo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

source activate Self_Model

cd ~/scratch/Self_Model/FER_Project

python analysis/train_resnet18_baseline.py \
    --csv data/FER2013/train.csv \
    --out-dir models/resnet18_baseline_retrain \
    --epochs 50 \
    --patience 10 \
    --batch-size 128 \
    --num-workers 1 \
    --lr 1e-3 \
    --weight-decay 1e-4
