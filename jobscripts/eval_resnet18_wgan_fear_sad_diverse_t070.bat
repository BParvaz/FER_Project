#!/bin/bash --login
#SBATCH --job-name=eval_r18_wgan_fsd
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
    --checkpoints models/resnet18_wgan_curated_fear_sad_diverse_t070/best.pth \
    --out reports/resnet18_wgan_curated_fear_sad_diverse_t070_eval.csv \
    --predictions-out reports/resnet18_wgan_curated_fear_sad_diverse_t070_predictions.csv \
    --batch-size 128 \
    --num-workers 1
