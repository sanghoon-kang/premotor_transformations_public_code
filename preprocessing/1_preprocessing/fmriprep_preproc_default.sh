#!/usr/bin/env bash
#SBATCH --output=%j-fmriprep_preproc_default.out
#SBATCH -p psych_week
#SBATCH -t 25:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sanghoon.kang@yale.edu

module load fmriprep

subject_name=$1
subject="sub-${subject_name}"
preproc_version="fmriprep_default"

echo preprocessing $subject_name without recon, $preproc_version

fmriprep /home/sk2949/project/mrdecode/bids_imgs/ /home/sk2949/project/mrdecode/${preproc_version}/output/ participant --participant-label ${subject} -w /home/sk2949/project/mrdecode/${preproc_version}/work/ --fs-license-file /home/sk2949/project/github/mrdecode/analysis/fmri/preprocessing/license.txt -vv --resource-monitor --nprocs 16 --omp-nthreads 8