#!/bin/bash --login
#SBATCH --job-name=sample_class
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --output=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.out
#SBATCH --error=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.err
#SBATCH --ntasks-per-node 8
#SBATCH -t 0-1

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


export OPENAI_LOGDIR="./classifier_sample_v3_fer_smoke_ddim50"
G_MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True --use_fp16 True"
G_DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"
C_CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True"
SAMPLE_FLAGS="--batch_size 14 --num_samples 70 --timestep_respacing ddim50 --use_ddim True --num_classes 7"



python guided-diffusion/scripts/classifier_sample.py \
    --model_path intensive_trained_models/guided_diffusion_v3/ema_0.9999_150000.pt \
    --classifier_path intensive_trained_models/guided_diffusion_classifier_v2/model110000.pt \
    $G_MODEL_FLAGS $G_DIFFUSION_FLAGS $C_CLASSIFIER_FLAGS $SAMPLE_FLAGS
