#!/bin/bash --login
#SBATCH --job-name=filter_wgan_disg
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
    --npz WGAN/generated_samples/version_013_disgust_rescue_5000/samples_5000x48x48x3.npz \
    --checkpoint models/resnet18_baseline_retrain/best.pth \
    --out-dir WGAN/generated_samples/version_013_disgust_rescue_5000/resnet18_filtered_t070 \
    --summary reports/wgan_version013_disgust_rescue_filter_t070.csv \
    --threshold 0.70 \
    --target-classes disgust

