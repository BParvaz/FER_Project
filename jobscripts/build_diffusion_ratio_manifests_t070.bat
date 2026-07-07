#!/bin/bash --login
#SBATCH --job-name=build_diff_ratios
#SBATCH -p serial
#SBATCH --output=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.out
#SBATCH --error=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.err
#SBATCH --ntasks 1
#SBATCH -t 0-1

set -eo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1
source activate Self_Model

cd ~/scratch/Self_Model/FER_Project

python analysis/build_synthetic_ratio_manifest.py \
    --manifest reports/diffusion_7class_polished_t070_curated_minority_manifest.csv \
    --out reports/diffusion_7class_polished_t070_curated_minority_ratio025_manifest.csv \
    --fraction 0.25

python analysis/build_synthetic_ratio_manifest.py \
    --manifest reports/diffusion_7class_polished_t070_curated_minority_manifest.csv \
    --out reports/diffusion_7class_polished_t070_curated_minority_ratio050_manifest.csv \
    --fraction 0.50
