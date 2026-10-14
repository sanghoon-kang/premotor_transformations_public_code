#!/usr/bin/env bash
#SBATCH --output=%j-generate_beta_maps.out
#SBATCH -p psych_week
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu=128GB
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=sanghoon.kang@yale.edu
#SBATCH --mail-type=ALL

# trial-wise beta maps for one subject

module miniconda

conda init
conda activate mrdecode

python -u /gpfs/milgram/pi/mcdougle/sk2949/github/mrdecode/analysis/fmri/glm_mvpa/first_level_glm/generate_beta_maps.py "$1" $2 "$3" "$4"
