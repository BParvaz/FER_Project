#!/bin/bash
set -eo pipefail

build=$(sbatch --parsable jobscripts/build_diffusion_ratio_manifests_t070.bat)

train025=$(sbatch --parsable --dependency=afterok:${build} jobscripts/train_resnet18_diffusion_curated_ratio025_t070.bat)
eval025=$(sbatch --parsable --dependency=afterok:${train025} jobscripts/eval_resnet18_diffusion_curated_ratio025_t070.bat)

train050=$(sbatch --parsable --dependency=afterok:${build} jobscripts/train_resnet18_diffusion_curated_ratio050_t070.bat)
eval050=$(sbatch --parsable --dependency=afterok:${train050} jobscripts/eval_resnet18_diffusion_curated_ratio050_t070.bat)

stats=$(sbatch --parsable --dependency=afterok:${eval025}:${eval050} jobscripts/stats_resnet18_diffusion_ratio_sweep_t070.bat)

cat <<EOF
build=${build}
train025=${train025}
eval025=${eval025}
train050=${train050}
eval050=${eval050}
stats=${stats}
EOF
