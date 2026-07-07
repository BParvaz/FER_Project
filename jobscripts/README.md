# Jobscript Index

This directory collects the Slurm entry points used to run training,
generation, curation, and evaluation jobs on HPC.

## Setup

- `env_setup.txt`: create the Conda environment from `env.yaml`
- `convert_fer_images.bat`: export FER2013 CSV rows into image folders

All jobs write scheduler output into `jobscripts/slurm/`.

## Core Generative Jobs

### WGAN

- `run_WGAN.bat`: train the WGAN model
- `sample_wgan_minority_t070.bat`: generate minority-class WGAN samples
- `sample_wgan_disgust_rescue_t070.bat`: generate additional disgust-focused
  WGAN samples

### Diffusion

- `train_guided_diff.bat`: train the class-conditional diffusion model
- `train_d_class.bat`: train the diffusion classifier
- `sample_diff_7class.bat`: unconditional class-conditional diffusion sampling
- `sample_class_7class_polished.bat`: classifier-guided diffusion sampling
- `sample_class_7class_polished_700.bat`: smaller curated diffusion batch
- `sample_class_7class_polished_5000.bat`: larger diffusion batch for stronger
  augmentation runs

## Curation And Filtering

- `filter_diffusion_samples.bat`
- `filter_diffusion_samples_700.bat`
- `filter_diffusion_samples_5000.bat`
- `curate_diffusion_samples_t070.bat`
- `curate_diffusion_samples_5000_100pc_t070.bat`
- `curate_diffusion_samples_5000_200pc_t070.bat`
- `filter_wgan_samples_t070.bat`
- `filter_wgan_disgust_rescue_t070.bat`
- `curate_wgan_samples_t070.bat`
- `curate_wgan_fear_sad_t070.bat`
- `curate_wgan_fear_sad_diverse_t070.bat`

## FER Training And Evaluation

- `train_resnet18_baseline.bat`
- `eval_resnet18_baseline.bat`
- `train_resnet18_diffusion_minority_t070.bat`
- `eval_resnet18_diffusion_minority_t070.bat`
- `train_resnet18_diffusion_curated_minority_t070.bat`
- `eval_resnet18_diffusion_curated_minority_t070.bat`
- `train_resnet18_diffusion_curated_ratio025_t070.bat`
- `train_resnet18_diffusion_curated_ratio050_t070.bat`
- `train_resnet18_diffusion_curated_100pc_t070.bat`
- `train_resnet18_diffusion_curated_200pc_t070.bat`
- `eval_resnet18_diffusion_curated_ratio025_t070.bat`
- `eval_resnet18_diffusion_curated_ratio050_t070.bat`
- `eval_resnet18_diffusion_curated_100pc_t070.bat`
- `eval_resnet18_diffusion_curated_200pc_t070.bat`
- `train_resnet18_wgan_minority_t070.bat`
- `eval_resnet18_wgan_minority_t070.bat`
- `train_resnet18_wgan_fear_sad_t070.bat`
- `eval_resnet18_wgan_fear_sad_t070.bat`
- `train_resnet18_wgan_fear_sad_diverse_t070.bat`
- `eval_resnet18_wgan_fear_sad_diverse_t070.bat`

## Batch Submission Helpers

- `submit_curated_bridge.sh`
- `submit_diffusion_ratio_sweep_t070.sh`
- `submit_diffusion_strong_dose_t070.sh`

## Notes

- The older `sample_class.bat` and `sample_diff.bat` files are retained as
  earlier diffusion smoke-test entry points.
- File names with `t070` indicate curation/filtering runs that use a threshold
  of `0.70`.
