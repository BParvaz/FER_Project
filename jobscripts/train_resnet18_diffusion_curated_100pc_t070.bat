#!/bin/bash --login
#SBATCH --job-name=train_r18_diff100
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --output=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.out
#SBATCH --error=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.err
#SBATCH --ntasks 1
#SBATCH -t 0-2

set -eo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1
source activate Self_Model

cd ~/scratch/Self_Model/FER_Project

python analysis/train_resnet18_baseline.py \
    --csv data/FER2013/train.csv \
    --synthetic-manifest reports/diffusion_7class_polished_t070_curated_100pc_manifest.csv \
    --out-dir models/resnet18_diffusion_curated_100pc_t070 \
    --epochs 50 \
    --patience 10 \
    --batch-size 128 \
    --num-workers 1
