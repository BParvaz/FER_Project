#!/bin/bash --login
#SBATCH --job-name=balance_wgan_t070
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
    --accepted-dir WGAN/generated_samples/version_013_minority_2100/resnet18_filtered_t070/accepted \
    --out-dir data/synthetic/wgan_version013_t070_balanced_minority \
    --manifest reports/wgan_version013_t070_balanced_minority_manifest.csv \
    --classes disgust,fear,sad \
    --per-class 35

python utils/make_image_dir_contact_sheet.py \
    data/synthetic/wgan_version013_t070_balanced_minority \
    --out data/synthetic/wgan_version013_t070_balanced_minority/contact_sheet.png \
    --limit 120

