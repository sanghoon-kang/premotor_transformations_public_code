# premotor_transformations_public_code

Data and code for Kang, Trach, & McDougle (2026), *Abstract representations of sensorimotor transformations in human premotor cortex*.

- `data/`: subject-level data (columns in `data/README.md`).
- `analysis_pipeline/`: scripts that reproduce the reported statistics from `data/`, numbered in the order of the paper (R 4.2.0, Python 3.10.13; `9_mds.py` needs Python ≥ 3.11). Run from this folder.
  - `1_behavior.R`: Fig. 2
  - `2_rsa_stats.py`: Figs. 3, S5
  - `3_brain_behavior.py`: Fig. 4c–d
  - `4_caching_behavior.R`: Fig. 5b
  - `5_caching_overlap.py`: Fig. 5c–d
  - `6_model_comparison.py`: Fig. S1
  - `7_movement_controls.py`: Fig. S3 and the hand-direction control
  - `8_region_anova.py`: premotor vs. M1 ANOVA
  - `9_mds.py`: Figs. 4a–b, S4
  - `subject_level/`: code that produced `data/` from the beta maps, and the searchlight and univariate analyses (Figs. 6, S2, S6). Needs the images; included as a record.
- `image_processing_pipeline/`: BIDS conversion, fMRIPrep and first-level GLM. Included as a record.
- `requirements.txt`: Python packages.
- `LICENSE`: MIT for the code; data are CC0.

Contact: sanghoon.kang@yale.edu
