# analysis_pipeline/subject_level/

The analyses that run on the trial-wise beta maps from
`image_processing_pipeline/`; their outputs are what `data/` holds. **They need the beta maps and our cluster, so they will not run
outside our lab**; the numbered scripts one folder up reproduce every
statistic from `data/` instead. These are here so that step can be read rather
than taken on faith. They ran on a SLURM cluster against lab paths
(`/home/sk2949/...`).

| Folder | Produces |
|---|---|
| `roi_rsa/subject_rsa_full.py` | `rsa_betas.csv` and `rsm/` (Fig 3) |
| `roi_overlap/` | `overlap.csv` (Figs 5c-d) |
| `searchlight/` | The searchlight maps and their cluster-corrected tests (Figs 6 and S5) |
| `univariate_contrast/` | The whole-brain Rotation vs Reflection control (Fig S6) |
| `utils/` | Helpers shared by the folders above and by the first-level GLM |

## Reading notes

- **`roi_rsa/` is the paper's main method.** If you read one thing here, read
  `subject_rsa_full.py`: it turns the voxel pattern in an ROI into one of the
  8x8 matrices in `data/rsm/`, including the odd-against-even run
  cross-validation that makes negative entries meaningful, and then fits the
  four model matrices to produce `rsa_betas.csv`.
- **`utils/pipeline_config.py` is the lab-side single source of truth** for
  the subject list, the ROI sets and the four model matrices. `config.py` one
  folder up restates them so the numbered scripts stand alone.
  `pipeline_config.py` is also where the FDR families are defined: ROIs are
  grouped `cortical` and `subcortical` and corrected within family, which is
  why `2_rsa_stats.py` corrects across all sixteen cortical ROIs at once rather
  than the ten in the main figures.
- **The scripts import `from utils.utils import ...`** because on the cluster
  `utils/` sits beside them, as it does here.
- **ROI masks are not stored here.** They are built at run time by
  `get_roi_mask()` in `utils/utils.py` from the Jülich atlas's maximum
  probability maps, which are cited in the Methods and are better obtained from
  their own source than redistributed here.
- **`roi_overlap/` computes the overlap measure** in `data/overlap.csv`. It
  needs voxel-level patterns, which the 8x8 matrices do not contain, so that
  one column cannot be recomputed from the public data and is provided as is.
- **Thresholds for the whole-brain tests.** `searchlight/` and
  `univariate_contrast/` run every combination of mask (whole brain, gray
  matter) and cluster-forming threshold (p < .001, .005, .01, .05, and TFCE),
  10,000 sign-flip permutations each, and print a cluster report for all of
  them. The paper reports whole brain at p < .05; the rest are sensitivity
  checks, kept so they can be compared.
- **No figure code.** This repository is for regenerating the numbers and
  keeping a record of how they were made, so plotting code is left out. The
  figures were drawn from the outputs of these scripts and of the numbered
  scripts one folder up. `searchlight/searchlight_group_test.py` and
  `univariate_contrast/rotation_vs_reflection.py` are the lab scripts with
  their plotting removed; the tests, their settings and the cluster reports
  are unchanged.

## Not included

Scripts for analyses that were run but are not reported in the paper
(classifiers, crossnobis distances, striatal and cerebellar parcellations) are
left out, as are the lab's job-submission notebook and the data-transfer and
licence-management scripts, which contain credentials and licensed material.
