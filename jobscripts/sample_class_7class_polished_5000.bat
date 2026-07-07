#!/bin/bash --login
#SBATCH --job-name=sample_diff_5k
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --output=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.out
#SBATCH --error=/net/scratch/b84547bp/Self_Model/FER_Project/jobscripts/slurm/%x-%j.err
#SBATCH --ntasks-per-node 8
#SBATCH -t 0-4

set -eo pipefail

module purge
export ADDR2LINE=addr2line
module load apps/binapps/conda/miniforge3/25.9.1
source activate Self_Model

cd ~/scratch/Self_Model/FER_Project/Diffusion/

export OPENAI_LOGDIR="./classifier_sample_7class_v1_polished_ddim250_5000"

G_MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True --use_fp16 True --model_num_classes 7"
G_DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"
C_CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_use_fp16 True --classifier_num_classes 7 --classifier_scale 1.0"
SAMPLE_FLAGS="--batch_size 10 --num_samples 5000 --timestep_respacing ddim250 --use_ddim True --num_classes 7"

python guided-diffusion/scripts/classifier_sample.py \
    --model_path intensive_trained_models/guided_diffusion_7class_v1/ema_0.9999_150000.pt \
    --classifier_path intensive_trained_models/guided_diffusion_classifier_7class_v1/model110000.pt \
    $G_MODEL_FLAGS $G_DIFFUSION_FLAGS $C_CLASSIFIER_FLAGS $SAMPLE_FLAGS
