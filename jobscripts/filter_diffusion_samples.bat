#!/bin/bash --login
#SBATCH --job-name=filter_diff_samples
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

python analysis/filter_synthetic_samples.py \
    --npz Diffusion/classifier_sample_7class_v1_polished_ddim250/samples_70x64x64x3.npz \
    --checkpoint models/resnet18_baseline_retrain/best.pth \
    --out-dir Diffusion/classifier_sample_7class_v1_polished_ddim250/resnet18_filtered_t070 \
    --summary reports/diffusion_polished_resnet18_filter_t070.csv \
    --threshold 0.70 \
    --target-classes disgust,fear,sad
