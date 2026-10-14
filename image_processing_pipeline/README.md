# image_processing_pipeline/

From the raw scanner images to one activation map per trial. **These ran on our
cluster and will not run elsewhere**; they are here so the processing can be
read. Nothing in `data/` or the numbered scripts depends on running them.

| Step | Scripts | Produces |
|---|---|---|
| 1. BIDS and fMRIPrep | `1_bids_fmriprep/` | Defaced BIDS data, then fMRIPrep output |
| 2. First-level GLM | `2_first_level_glm/` | One activation (beta) map per trial |

The GLM imports helpers from `analysis_pipeline/subject_level/utils/`, which
sits beside it on the cluster. The trial-wise maps it writes are the input to
everything in `analysis_pipeline/subject_level/`.

The FreeSurfer licence and the scanner-side data-transfer script are not
included: the first is licensed material and the second contains credentials.
