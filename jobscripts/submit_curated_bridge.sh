#!/bin/bash
set -eo pipefail

diff_curate=$(sbatch --parsable jobscripts/curate_diffusion_samples_t070.bat)
diff_train=$(sbatch --parsable --dependency=afterok:${diff_curate} jobscripts/train_resnet18_diffusion_curated_minority_t070.bat)
diff_eval=$(sbatch --parsable --dependency=afterok:${diff_train} jobscripts/eval_resnet18_diffusion_curated_minority_t070.bat)

wgan_sample=$(sbatch --parsable jobscripts/sample_wgan_minority_t070.bat)
wgan_filter=$(sbatch --parsable --dependency=afterok:${wgan_sample} jobscripts/filter_wgan_samples_t070.bat)
wgan_curate=$(sbatch --parsable --dependency=afterok:${wgan_filter} jobscripts/curate_wgan_samples_t070.bat)
wgan_train=$(sbatch --parsable --dependency=afterok:${wgan_curate} jobscripts/train_resnet18_wgan_minority_t070.bat)
wgan_eval=$(sbatch --parsable --dependency=afterok:${wgan_train} jobscripts/eval_resnet18_wgan_minority_t070.bat)

stats=$(sbatch --parsable --dependency=afterok:${diff_eval}:${wgan_eval} jobscripts/stats_resnet18_wgan_diffusion_t070.bat)

cat <<EOF
diff_curate=${diff_curate}
diff_train=${diff_train}
diff_eval=${diff_eval}
wgan_sample=${wgan_sample}
wgan_filter=${wgan_filter}
wgan_curate=${wgan_curate}
wgan_train=${wgan_train}
wgan_eval=${wgan_eval}
stats=${stats}
EOF

