#!/bin/bash --login
#SBATCH --job-name=convert_fer_images
#SBATCH -p serial
#SBATCH --output=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.out
#SBATCH --error=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.err
#SBATCH -t 0-1

set -eo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

source activate Self_Model

cd ~/scratch/Self_Model/FER_Project/Diffusion/img-conversion/

python convert.py --clean
