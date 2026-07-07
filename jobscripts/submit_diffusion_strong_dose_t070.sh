#!/bin/bash
set -eo pipefail

sample=$(sbatch --parsable jobscripts/sample_class_7class_polished_5000.bat)
filter=$(sbatch --parsable --dependency=afterok:${sample} jobscripts/filter_diffusion_samples_5000.bat)

curate100=$(sbatch --parsable --dependency=afterok:${filter} jobscripts/curate_diffusion_samples_5000_100pc_t070.bat)
train100=$(sbatch --parsable --dependency=afterok:${curate100} jobscripts/train_resnet18_diffusion_curated_100pc_t070.bat)
eval100=$(sbatch --parsable --dependency=afterok:${train100} jobscripts/eval_resnet18_diffusion_curated_100pc_t070.bat)

curate200=$(sbatch --parsable --dependency=afterok:${filter} jobscripts/curate_diffusion_samples_5000_200pc_t070.bat)
train200=$(sbatch --parsable --dependency=afterok:${curate200} jobscripts/train_resnet18_diffusion_curated_200pc_t070.bat)
eval200=$(sbatch --parsable --dependency=afterok:${train200} jobscripts/eval_resnet18_diffusion_curated_200pc_t070.bat)

stats=$(sbatch --parsable --dependency=afterok:${eval100}:${eval200} jobscripts/stats_resnet18_diffusion_dose_strong_t070.bat)
visuals=$(sbatch --parsable --dependency=afterok:${stats} jobscripts/build_diffusion_dose_visuals.bat)

cat <<EOF
sample=${sample}
filter=${filter}
curate100=${curate100}
train100=${train100}
eval100=${eval100}
curate200=${curate200}
train200=${train200}
eval200=${eval200}
stats=${stats}
visuals=${visuals}
EOF
