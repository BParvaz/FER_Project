#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --output stdout-%j.log # stdout
#SBATCH --error stderr-%j.log #stderr
#SBATCH --ntasks 1
#SBATCH -t 0-4

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
SAMPLE_FLAGS="--num_samples 64 --batch_size 16"
export OPENAI_LOGDIR="./logs/img_samples/"
# Run
python improved-diffusion/scripts/image_sample.py --model_path ./logs/ema_0.9999_100000.pt $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
