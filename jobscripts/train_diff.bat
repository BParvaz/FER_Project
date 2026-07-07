#!/bin/bash --login
#SBATCH --job-name=train_diff
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --output=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.out
#SBATCH --error=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.err
#SBATCH --ntasks-per-node 8
#SBATCH -t 1-0

set -eo pipefail


# Clean modules
module purge

export ADDR2LINE=addr2line
# Load Conda
module load apps/binapps/conda/miniforge3/25.9.1

# Load Env
source activate Self_Model

# Enter working dir
cd ~/scratch/Self_Model/FER_Project/Diffusion/

# Define flags
MODEL_FLAGS="--image_size 64 --num_channels 32 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64"
export OPENAI_LOGDIR="./logs/"
# Run
python improved-diffusion/scripts/image_train.py --data_dir img-conversion/fer_images/train/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
