#!/bin/bash --login
#SBATCH --job-name=stats_resnet18
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --output=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.out
#SBATCH --error=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.err
#SBATCH --ntasks 1
#SBATCH -t 0-1

set -eo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1
source activate Self_Model

cd ~/scratch/Self_Model/FER_Project

python analysis/evaluate_resnet18_baseline.py \
    --csv data/FER2013/train.csv \
    --checkpoints models/resnet18_baseline_retrain/best.pth \
    --out reports/resnet18_baseline_retrain_eval.csv \
    --predictions-out reports/resnet18_baseline_retrain_predictions.csv \
    --batch-size 128 \
    --num-workers 1

python analysis/statistical_tests.py \
    reports/resnet18_baseline_retrain_predictions.csv \
    --out reports/resnet18_baseline_retrain_statistical_tests.csv \
    --bootstrap 2000
