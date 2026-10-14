#!/usr/bin/env bash
#SBATCH --output=%j-searchlight_rsa.out
#SBATCH -p psych_day
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-task=2

# searchlight RSA for one subject (Figs 6 and S5)

source ~/.bashrc
conda activate mrdecode

python -u /gpfs/milgram/project/mcdougle/sk2949/github/mrdecode/analysis/fmri/glm_mvpa/searchlight/searchlight_rsa.py "$1" $2 $3 $4 "$5" "$6"
