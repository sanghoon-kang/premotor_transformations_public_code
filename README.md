# premotor_transformations_public_code

Subject-level data and analysis code for Kang, Trach, & McDougle (2026), *Abstract representations of sensorimotor transformations in human premotor cortex*. The code regenerates the statistics reported in the paper and records how they were produced. It does not include figure code.

## Files

- `data/`: subject-level data that every statistic is computed from (columns in `data/README.md`).
  - `behavior.csv`: scanning-phase behavior, one row per trial.
  - `tutorial.csv`: tutorial-phase behavior (n = 24).
  - `rsm/`: cross-validated 8×8 similarity matrices, one per participant and ROI.
  - `rsa_betas.csv`: RSA regression coefficients per participant and ROI.
  - `overlap.csv`: rule–movement overlap per participant and ROI.
- `analysis_pipeline/`: the paper's statistics, computed from `data/`. Run any numbered script from this folder; each prints its results (R 4.2.0, Python 3.10.13).
  - `config.py`: subject list, ROIs and model matrices shared by the scripts below.
  - `1_behavior.R`: scanning and tutorial behavior (Fig. 2).
  - `2_rsa_stats.py`: RSA regression coefficients (Fig. 3; set `SUPP = True` for Fig. S5).
  - `3_brain_behavior.py`: transformation-context coding vs. task performance (Fig. 4c–d).
  - `4_caching_behavior.R`: behavioral caching effect (Fig. 5b).
  - `5_caching_overlap.py`: neural rule–movement overlap vs. caching (Fig. 5c–d).
  - `6_model_comparison.py`: full vs. reduced RSA model and noise ceiling (Fig. S1).
  - `7_movement_controls.py`: behavioral vs. neural transformation coding (Fig. S3), and the RSA refit with participants' actual hand directions.
  - `8_region_anova.py`: premotor vs. M1, region × hemisphere ANOVA.
  - `9_mds.py`: MDS coordinates with scikit-learn's ClassicalMDS (Figs. 4a–b and S4). Needs Python ≥ 3.11.
  - `subject_level/`: lab-cluster code that runs on the trial-wise beta maps. It needs the images, so it is included as a record only.
    - `roi_rsa/`: per-subject similarity matrices and RSA coefficients (`rsm/`, `rsa_betas.csv`).
    - `roi_overlap/`: rule–movement overlap (`overlap.csv`).
    - `searchlight/`: whole-brain searchlight RSA and its group cluster tests (Figs. 6 and S6).
    - `univariate_contrast/`: whole-brain Rotation vs. Reflection contrast (Fig. S2).
    - `utils/`: data loading, ROI masks and first-level design helpers.
- `image_processing_pipeline/`: lab-cluster code from raw images to trial-wise beta maps (BIDS conversion, fMRIPrep, first-level GLM). Included as a record only.
- `requirements.txt`: Python packages for `analysis_pipeline/`.
- `LICENSE`: MIT license for the code. The data are released under CC0.

Contact: sanghoon.kang@yale.edu
