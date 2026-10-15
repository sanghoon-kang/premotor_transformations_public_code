# DICOM -> BIDS with heudiconv for one subject.

conda activate ycrc_default

subject="$1"

heudiconv --files /home/sk2949/project/mrdecode/raw_imgs/${subject}/raw/SCANS/*/DICOM/*.dcm -o /home/sk2949/project/mrdecode/bids_imgs/ -f /home/sk2949/project/mrdecode/code/heuristic.py -s ${subject} -ss itbs -c dcm2niix -b --overwrite --dcmconfig /home/sk2949/Documents/mrdecode/analysis/fmri/preprocessing/dcm2niix_config.json

