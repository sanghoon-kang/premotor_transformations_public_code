# DICOM -> BIDS with heudiconv, one subject at a time.
# heudiconv doesn't run in the mrdecode env, so this uses a separate one.

conda activate ycrc_default

subject="mrd1029"

# dcm2niix_config.json ignores partial volumes; heuristic.py has to be where heudiconv runs

heudiconv --files /home/sk2949/project/mrdecode/raw_imgs/${subject}/raw/SCANS/*/DICOM/*.dcm -o /home/sk2949/project/mrdecode/bids_imgs/ -f /home/sk2949/project/mrdecode/code/heuristic.py -s ${subject} -ss itbs -c dcm2niix -b --overwrite --dcmconfig /home/sk2949/Documents/mrdecode/analysis/fmri/preprocessing/dcm2niix_config.json

