#!/bin/bash --login
#SBATCH --job-name=balance_diff_700
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

python analysis/build_balanced_synthetic_subset.py \
    --accepted-dir Diffusion/classifier_sample_7class_v1_polished_ddim250_700/resnet18_filtered_t070/accepted \
    --out-dir data/synthetic/diffusion_7class_polished_t070_balanced_minority \
    --manifest reports/diffusion_7class_polished_t070_balanced_minority_manifest.csv \
    --classes disgust,fear,sad
