#!/bin/bash --login
#SBATCH -p gpuL 
#SBATCH -G 1
#SBATCH --output stdout-%j.log # stdout
#SBATCH --error stderr-%j.log #stderr
#SBATCH --ntasks-per-node 8
#SBATCH -t 0-4

set -eo pipefail


# Clean modules
module purge

export ADDR2LINE=addr2line
# Load Conda
module load apps/binapps/anaconda3/2024.10

# Load Env
source activate Self_Model

# Enter working dir
cd ~/scratch/Self_Model/FER_Project

# Run
python -u -m CGAN.train