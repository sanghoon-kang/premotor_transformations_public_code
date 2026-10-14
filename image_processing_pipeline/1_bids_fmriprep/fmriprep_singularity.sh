#!/usr/bin/env bash
#SBATCH --output=%j-fmriprep_singularity.out
#SBATCH -p psych_day
#SBATCH -t 24:00:00
#SBATCH --mem 64GB
#SBATCH -n 1

# builds the fMRIPrep singularity image


set -e

module load fmriprep

singularity build home/sk2949/project/mrdecode/fmriprep/fmriprep.simg docker://poldracklab/fmriprep:latest
