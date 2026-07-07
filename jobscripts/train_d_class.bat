#!/bin/bash --login
#SBATCH --job-name=train_d_class
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
export OPENAI_LOGDIR="./intensive_trained_models/guided_diffusion_classifier_7class_v1"

# Include classifier flags

C_TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 256 --microbatch 16 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
C_CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_num_classes 7"



# Run classifier_trainer
python guided-diffusion/scripts/classifier_train.py --data_dir img-conversion/fer_images/train/ $C_TRAIN_FLAGS $C_CLASSIFIER_FLAGS 
