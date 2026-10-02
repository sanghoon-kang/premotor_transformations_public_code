# use the following code (use whatever subject name necessary) to generate skeleton

# heudiconv --files /home/sk2949/project/mrdecode/raw_imgs/mrd1005/raw/SCANS/*/DICOM/*.dcm -o /home/sk2949/project/mrdecode/bids_imgs/mrd1005 -f convertall -s mrd1005 -c none

# python version in mrdecode environment is not compatible with heudiconv. Using ycrc_default where i manually installed heudiconv

conda activate ycrc_default

subject="mrd1029"

# the configuration is necessary to ignore partial volumes (-i y options ignorempartial volumes)
# the configuration file specifies the options
# heuristics.py file has to be in the same directory as where heudiconv is being run (within the 'project' directory' so made a copy of the file there

heudiconv --files /home/sk2949/project/mrdecode/raw_imgs/${subject}/raw/SCANS/*/DICOM/*.dcm -o /home/sk2949/project/mrdecode/bids_imgs/ -f /home/sk2949/project/mrdecode/code/heuristic.py -s ${subject} -ss itbs -c dcm2niix -b --overwrite --dcmconfig /home/sk2949/Documents/mrdecode/analysis/fmri/preprocessing/dcm2niix_config.json

# Maybe make slurm script for both bids conversion and add_intendedfor?

