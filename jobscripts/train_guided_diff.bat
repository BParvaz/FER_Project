#!/bin/bash --login
#SBATCH --job-name=train_guided_diff
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

# Enter working dirA
cd ~/scratch/Self_Model/FER_Project/Diffusion/

# Define flags # NEED TO ADJUST PARAMETERS FOR PERFORMANCE
G_MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True --model_num_classes 7 --use_fp16 True"
G_DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"
G_TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --microbatch 16"
export OPENAI_LOGDIR="intensive_trained_models/guided_diffusion_7class_v1"

# Dump Settings
mkdir -p ~/scratch/Self_Model/FER_Project/Diffusion/$OPENAI_LOGDIR
touch ~/scratch/Self_Model/FER_Project/Diffusion/$OPENAI_LOGDIR/settings.txt
echo -e "$G_MODEL_FLAGS\n$G_DIFFUSION_FLAGS\n$G_TRAIN_FLAGS" > ~/scratch/Self_Model/FER_Project/Diffusion/$OPENAI_LOGDIR/settings.txt


# Run image_trainer
python guided-diffusion/scripts/image_train.py --data_dir img-conversion/fer_images/train/ $G_MODEL_FLAGS $G_DIFFUSION_FLAGS $G_TRAIN_FLAGS
