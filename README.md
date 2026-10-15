# premotor_transformations_public_code

Summarized data and code for Kang, Trach, & McDougle (2026), *Abstract representations of sensorimotor transformations in human premotor cortex*.

- `data/`: subject-level data, released under CC0.
- `analysis_pipeline/`: scripts that reproduce the reported statistics from `data/`.
  - `1_behavior.R`: Fig. 2
  - `2_rsa_stats.py`: Figs. 3, S5
  - `3_brain_behavior.py`: Fig. 4c–d
  - `4_caching_behavior.R`: Fig. 5b
  - `5_caching_overlap.py`: Fig. 5c–d
  - `6_model_comparison.py`: Fig. S1
  - `7_movement_controls.py`: Fig. S3 and the hand-direction control
  - `8_region_anova.py`: premotor vs. M1 ANOVA
  - `9_mds.py`: Figs. 4a–b, S4
  - `subject_level/`: code that produced `data/` from the beta maps, and the searchlight and univariate analyses (Figs. 6, S2, S6), for the record.
- `image_processing_pipeline/`: BIDS conversion, fMRIPrep and first-level GLM, for the record.

Contact: sanghoon.kang@yale.edu
