#!/bin/bash --login
#SBATCH --job-name=stats_r18_curated
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

python analysis/statistical_tests.py \
    reports/resnet18_baseline_retrain_predictions.csv \
    reports/resnet18_wgan_curated_minority_t070_predictions.csv \
    reports/resnet18_diffusion_curated_minority_t070_predictions.csv \
    --out reports/resnet18_baseline_vs_wgan_vs_diffusion_curated_t070_stats.csv \
    --bootstrap 2000

python analysis/build_dissertation_eval_skeleton.py \
    --baseline reports/resnet18_baseline_vs_wgan_vs_diffusion_curated_t070_stats.csv \
    --out reports/dissertation_eval_skeleton.md

python analysis/build_report_visuals.py \
    --out-dir report_visuals_results_section
