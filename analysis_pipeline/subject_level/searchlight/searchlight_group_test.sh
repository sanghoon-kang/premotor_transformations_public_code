#!/usr/bin/env bash
#SBATCH --output=%j-searchlight_group_test.out
#SBATCH -p psych_day
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=8

# group-level cluster tests for the searchlight (Figs 6 and S5)

source ~/.bashrc
conda activate mrdecode

python -u /gpfs/milgram/project/mcdougle/sk2949/github/mrdecode/analysis/fmri/glm_mvpa/searchlight/searchlight_group_test.py
