#!/bin/bash --login
#SBATCH --job-name=curate_wgan_t070
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
    --accepted-dir WGAN/generated_samples/version_013_minority_2100/resnet18_filtered_t070/accepted \
    --summary reports/wgan_version013_resnet18_filter_t070.csv \
    --out-dir data/synthetic/wgan_version013_t070_curated_minority \
    --manifest reports/wgan_version013_t070_curated_minority_manifest.csv \
    --scores-out reports/wgan_version013_t070_curated_scores.csv \
    --classes disgust,fear,sad \
    --per-class 35 \
    --min-confidence 0.70

python utils/make_image_dir_contact_sheet.py \
    data/synthetic/wgan_version013_t070_curated_minority \
    --out data/synthetic/wgan_version013_t070_curated_minority/contact_sheet.png \
    --limit 120

