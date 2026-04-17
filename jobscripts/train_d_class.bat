#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH --output stdout-%j.log # stdout
#SBATCH --error stderr-%j.log #stderr
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

# Define flags # NEED TO ADJUST PARAMETERS FOR PERFORMANCE
# G_MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True"
# G_DIFFUSION_FLAGS="--diffusion_steps 2000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
# G_TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --microbatch 8"
export OPENAI_LOGDIR="./classifier_logs_greedy/"

# Run image_trainer
# python guided-diffusion/scripts/image_train.py --data_dir img-conversion/fer_images/train/ $G_MODEL_FLAGS $G_DIFFUSION_FLAGS $G_TRAIN_FLAGS


# Include classifier flags

C_TRAIN_FLAGS="--iterations 150000 --anneal_lr True --batch_size 64 --microbatch 16 --lr 3e-4 --save_interval 5000 --weight_decay 0.05"
C_CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 64 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"


# Obtain best model so far
# LATEST_MODEL=$(ls ./guided_logs_intensive/ | grep "ema_0.9999_" | -t'_' -k3 -n | tail -n 1)

# Run classifier_trainer
python guided-diffusion/scripts/classifier_train.py --data_dir img-conversion/fer_images/train/ $C_TRAIN_FLAGS $C_CLASSIFIER_FLAGS 

