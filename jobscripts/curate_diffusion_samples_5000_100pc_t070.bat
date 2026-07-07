#!/bin/bash --login
#SBATCH --job-name=curate_diff100
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

python analysis/curate_synthetic_samples.py \
    --accepted-dir Diffusion/classifier_sample_7class_v1_polished_ddim250_5000/resnet18_filtered_t070/accepted \
    --summary reports/diffusion_polished_5000_resnet18_filter_t070.csv \
    --out-dir data/synthetic/diffusion_7class_polished_t070_curated_100pc \
    --manifest reports/diffusion_7class_polished_t070_curated_100pc_manifest.csv \
    --scores-out reports/diffusion_7class_polished_t070_curated_100pc_scores.csv \
    --classes disgust,fear,sad \
    --per-class 100 \
    --min-confidence 0.70 \
    --diversity-weight 0.25

python utils/make_image_dir_contact_sheet.py \
    data/synthetic/diffusion_7class_polished_t070_curated_100pc \
    --out data/synthetic/diffusion_7class_polished_t070_curated_100pc/contact_sheet.png \
    --limit 120
