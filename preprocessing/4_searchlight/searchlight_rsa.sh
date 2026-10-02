#!/usr/bin/env bash
#!/usr/bin/env bash
#SBATCH --output=%j-subject_rsa_full.out
#SBATCH -p psych_day
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-task=2


# use below if wanna do things interactively
# salloc -p day -c4 --mem-per-cpu=16G --time=24:00:00

module miniconda

# conda init
conda activate mrdecode
conda list


echo "mrdecode activated, running python script"

chmod 777 searchlight/searchlight_rsa.py

python -u searchlight/searchlight_rsa.py "$1" $2 $3 $4 "$5" "$6"