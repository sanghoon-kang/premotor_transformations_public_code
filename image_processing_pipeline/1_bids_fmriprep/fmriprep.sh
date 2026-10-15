#!/usr/bin/env bash
#SBATCH --output=%j-fmriprep.out
#SBATCH -p psych_week
#SBATCH -t 25:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --cpus-per-task=16

# fMRIPrep 23.2.1 for one subject.

module load fmriprep

subject="sub-$1"
out=/home/sk2949/project/mrdecode/fmriprep_aroma

fmriprep /home/sk2949/project/mrdecode/bids_imgs/ ${out}/output/ participant \
    --participant-label ${subject} -w ${out}/work/ \
    --fs-license-file /home/sk2949/project/github/mrdecode/analysis/fmri/preprocessing/license.txt \
    -vv --resource-monitor --nprocs 16 --omp-nthreads 8 --use-aroma \
    --skull-strip-t1w auto \
    --output-spaces T1w MNI152NLin6Asym MNI152NLin2009cAsym fsnative fsaverage
