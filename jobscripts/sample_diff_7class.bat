#!/bin/bash --login
#SBATCH --job-name=sample_diff_7class
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --output=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.out
#SBATCH --error=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.err
#SBATCH --ntasks 1
#SBATCH -t 0-1

set -eo pipefail

module purge

export ADDR2LINE=addr2line
module load apps/binapps/conda/miniforge3/25.9.1
source activate Self_Model

cd ~/scratch/Self_Model/FER_Project/Diffusion/

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True --use_fp16 True --model_num_classes 7"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"
SAMPLE_FLAGS="--num_samples 70 --batch_size 14 --timestep_respacing ddim50 --use_ddim True --num_classes 7"

export OPENAI_LOGDIR="./image_sample_7class_v1_smoke_ddim50"

python guided-diffusion/scripts/image_sample.py \
    --model_path intensive_trained_models/guided_diffusion_7class_v1/ema_0.9999_150000.pt \
    $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
