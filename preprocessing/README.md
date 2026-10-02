# preprocessing/

**Nothing in this folder runs outside our lab, and none of it is needed to
reproduce the paper's statistics.** The nine numbered scripts in the repository
root do that, starting from the data in `data/`.

These scripts are here so the pipeline from the raw MRI to `data/` can be
inspected rather than taken on faith. They ran on a SLURM cluster against lab
paths (`/home/sk2949/...`), expect software versions pinned to that
environment, and will not execute as-is anywhere else. They are a record, not
a tool — please read them rather than trying to run them.

## The chain

Starting from the raw BIDS data:

| Step | Scripts | Produces |
|---|---|---|
| 1. Preprocessing | `1_preprocessing/` | Defaced BIDS data, then fMRIPrep output |
| 2. First-level GLM | `2_first_level_glm/generate_beta_maps.py` | One activation map per trial |
| 3. ROI masks and voxelwise RSA | `3_voxelwise_rsa/` | `rsa_betas.csv`, `rsm/`, `overlap.csv` |
| 4. Searchlight RSA | `4_searchlight/searchlight_rsa.py` | The unthresholded maps in Figs 6 and S5 |
| 5. Export | `export_data.py` | Everything else in `data/` |

`5_figures/mds.py` is the script behind Figures 4a-b and S3. It is kept here
rather than in the root because it draws the panels; `9_mds.py` in the root is
the same computation reduced to the coordinates and the variance they capture.

## Reading notes

- **`3_voxelwise_rsa/` is the paper's actual method.** If you read one thing
  here, read `subject_rsa_full.py`: it turns a voxel pattern into one of the
  8x8 matrices in `data/rsm/`, including the odd-against-even run
  cross-validation that makes negative entries meaningful, and then fits the
  four model RDMs to produce `rsa_betas.csv`.
- **`pipeline_config.py` is the lab-side single source of truth** for the
  subject list, the ROI sets and the four model matrices. The repository's own
  `config.py` restates them so the analysis scripts stand alone, and
  `export_data.py` checks the two against each other before exporting.
  `pipeline_config.py` is also where the FDR families are defined: ROIs are
  grouped `cortical` and `subcortical` and corrected within family, which is
  why `2_rsa_stats.py` corrects across all sixteen cortical ROIs at once rather
  than the ten in the main figures.
- **ROI masks are not stored here.** They are built at run time by
  `get_roi_mask()` in `utils.py` from two published atlases — the Jülich
  probabilistic atlas and HMAT — which are cited in the Methods and are better
  obtained from their own sources than redistributed here.
- **`representation_angle_distance_*.py` computes the overlap measure** in
  `data/overlap.csv` (Figs 5c-d). It needs voxel-level patterns, which the 8x8
  matrices do not contain, so that one column cannot be recomputed from the
  public data and had to be exported instead.
- **Step 4 produces no statistic.** The searchlight maps are reported as
  unthresholded visualizations; no number in the paper comes from them, so
  there is nothing in the root scripts to recompute.
- **`export_data.py` is also the definition of `tutorial.csv`.** The original
  tutorial analysis script did not survive; the `process_trial()` port in it is
  how those columns are computed.

## Not included

Scripts for analyses that were run but are not reported in the paper
(classifiers, crossnobis distances, striatal and cerebellar parcellations,
permutation tests) are left out, as are the data-transfer and
licence-management scripts, which contain credentials and licensed material.
