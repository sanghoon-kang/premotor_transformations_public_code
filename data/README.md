# data/

Subject-level data, committed so that cloning the repository is enough to
reproduce every statistic in the paper: no raw MRI, no cluster, no
preprocessing. `analysis_pipeline/subject_level/` and
`image_processing_pipeline/` document how these files were derived from the
raw images.

| File | Contents |
|---|---|
| `behavior.csv` | One row per trial per subject |
| `rsa_betas.csv` | RSA regression coefficients, subject x ROI |
| `rsm/<subject>_<ROI>.npy` | 8x8 cross-validated similarity matrices |
| `overlap.csv` | Rule/movement overlap, subject x ROI |
| `tutorial.csv` | Trial-level learning-phase behavior (n = 24) |

## Columns

**`behavior.csv`**: `subject`, `run`, `trial`, `phase`, `manipulation_type`
(`rotation`/`mirror`), `critical`, `which_target` (1-4), `target_angle_deg`,
`initial_heading_deg`, `endpoint_angle_deg` (degrees), `rt`, `mt` (seconds).

This is the scanning session only; `phase` is the task name (`switch`) and has
a single value, so there is nothing to filter on.

`initial_heading_deg` is the signed error of the *cursor* relative to the
target, so perfect compensation is 0. `initial_heading_deg + target_angle_deg`
is therefore the direction the cursor initially travelled, the movement in
visual space, not the hand's direction, which differs from it by the rotation
or reflection. That cursor direction is what `7_movement_controls.py` builds its
similarity matrix from.

**`rsa_betas.csv`**: `subject`, `roi`, then one column per predictor:
`Rot/Ref`, `Target`, `Move`, `Trial`.

These were fit **with an intercept**. If you refit the four model matrices
against the 36 upper-triangle entries of the matching `rsm/*.npy` without one,
you will not reproduce these columns, and the `Rot/Ref` coefficient in
particular will come out very different, it is a block of ones, and so
collinear with a constant. `config.py` explains this. Every other fit in the
repository, including `6_model_comparison.py`, includes the intercept too.

**`overlap.csv`**: `subject`, `roi`, `n_vox`, `sq_overlap` (the measure),
`cos1`/`cos2` (the two cross-fold cosines it is the product of),
`mag_rule`/`mag_move` (cross-validated squared magnitude of each vector, the
dot product of its odd- and even-run estimates, unbiased, so it goes negative
when a vector is not reliably present), `rel_rule`/`rel_move` (split-half
reliability of each vector).

**`tutorial.csv`**: the tutorial (learning) session, which is separate from
scanning: `subject`, `block` (1 = Rotation, 2 = Reflection), `trial` (1-32
across both blocks), `manipulation_type`, `which_target`, `target_angle_deg`,
`initial_heading_deg`, `endpoint_angle_deg`, `rt`, `mt`.

Three subjects have no tutorial data, so this file has **n = 24** where the
others have 27. Where the learning phase was repeated, the latest attempt is
used. Movement measures were rebuilt from the raw cursor frames: each trial is
rotated so the target is at 0°, movement onset is the first frame after the go
signal where speed (on Savitzky-Golay-smoothed position, window 7, order 2)
reaches 2 cm/s, `initial_heading_deg` is the direction of cursor travel between
1 cm and 3 cm from the onset position, and `endpoint_angle_deg` the direction
to where the trial completed. `rt` is onset minus go signal, `mt` completion
minus onset.

**`rsm/*.npy`**: load with `np.load`. Conditions in order
`rot1-4, ref1-4`; the numeral is target location. The analyses fit the upper
triangle including the diagonal (36 entries): the diagonal is informative
because the matrices are cross-validated, and the trial-type predictor is the
identity matrix, so dropping it would reduce that predictor to zeros.

Noise ceilings are not stored here, `6_model_comparison.py` computes them
from these matrices.

## Four things that will otherwise confuse you

- **`critical` means MATCHED**, `uncritical` means UNMATCHED. The naming is
  historical and reads backwards; it is kept so the files match the code.
- **Similarity matrices are cross-validated** (odd runs vs even runs), so
  entries can be negative. That's meaningful, so don't clip them.
- **Matched trials are targets 1 and 3**, where both transformation contexts
  require the same movement.
- **n = 27** in every analysis except tutorial-phase learning, which is n = 24
  (three participants are missing tutorial behavior due to a data-saving
  error).

`.npy` rather than `.pkl`: unpickling a file executes code in it, so pickles
are a poor thing to hand to strangers. `.npy` is just an array.
