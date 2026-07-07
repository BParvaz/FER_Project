#!/bin/bash --login
#SBATCH --job-name=sample_wgan_disg
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

python analysis/sample_wgan_samples.py \
    --checkpoint logs/version_013/checkpoints_013/latest.pt \
    --out WGAN/generated_samples/version_013_disgust_rescue_5000/samples_5000x48x48x3.npz \
    --contact-sheet WGAN/generated_samples/version_013_disgust_rescue_5000/contact_sheet_first120.png \
    --classes disgust \
    --samples-per-class 5000 \
    --batch-size 128

