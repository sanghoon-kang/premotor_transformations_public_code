# Abstract representations of sensorimotor transformations in human premotor cortex

Data and analysis code for Kang, Trach, & McDougle (submitted).

## Layout

```
data/                        subject-level data: everything the statistics start from
analysis_pipeline/           all analysis code
  1_behavior.R ... 9_mds.py    the statistics in the paper, run from data/
  config.py                    subjects, ROIs and model matrices for those scripts
  subject_level/               the analyses that produced data/ from trial-wise
                               beta maps: ROI RSA, overlap, searchlight,
                               univariate contrast
image_processing_pipeline/   raw images to trial-wise beta maps: BIDS
                             conversion, fMRIPrep, first-level GLM
```

The nine numbered scripts reproduce every statistic in the paper from `data/`,
with nothing else needed. `subject_level/` and `image_processing_pipeline/`
need the raw images and our cluster, so they will not run elsewhere; they are
included so every step from the scanner to `data/` can be read.

The repository is for regenerating the numbers and keeping a record of how
they were made. It does not include the code that drew the figures.

## Setup

Behavioral analyses are in R, neuroimaging analyses in Python, matching how
they were run and how the Methods describe them.

```bash
cd analysis_pipeline
Rscript 1_behavior.R                    # R 4.2.0
pip install -r ../requirements.txt      # Python 3.10.13
python 2_rsa_stats.py
```

Run the scripts from `analysis_pipeline/` (or from the repository root, as
`python analysis_pipeline/2_rsa_stats.py`; both find `data/`). Each prints its
results to the terminal; they don't depend on each other.

The scripts run top to bottom, with `# %%` cell markers, so they work the same
in a terminal or an interactive window. Options are constants near the top of
each file (e.g. `SUPP = False` in `2_rsa_stats.py`). Edit those rather than
passing command-line flags. No function definitions to call, no imports
between scripts: open a file and read it from the top.

Both R scripts use base R only, so there's nothing to install for those.

## What reproduces what

Numbering roughly follows the order the paper reports things, with `8` and `9`
covering results that sit earlier in the text. Run them in any order; the
numbering is for reading, not dependency.

| Script | Language | Reproduces |
|---|---|---|
| `1_behavior.R` | R | Scanning behavior (Fig 2d-f) and tutorial learning (Fig 2b-c) |
| `2_rsa_stats.py` | Python | RSA regression coefficients (Fig 3; set `SUPP = True` for Fig S4) |
| `3_brain_behavior.py` | Python | Transformation-context beta x task performance (Fig 4c-d) |
| `4_caching_behavior.R` | R | Behavioral caching effect, matched vs unmatched (Fig 5b) |
| `5_caching_overlap.py` | Python | Neural overlap x behavioral caching index (Fig 5c-d) |
| `6_model_comparison.py` | Python | Full vs reduced model vs noise ceiling (Fig S1) |
| `7_movement_controls.py` | Python | Movement controls: behavioral vs neural Rot/Ref coding (Fig S2), and the main RSA refit with actual movements |
| `8_region_anova.py` | Python | Premotor vs M1 difference-of-differences, region x hemisphere |
| `9_mds.py` | Python | MDS coordinates and fit for Figs 4a-b and S3 |

`config.py` holds the subject list, ROI names and model matrices. Nothing else
defines them.

### Not covered by the numbered scripts, and why

- **The searchlight and univariate contrast (Figs 6, S5 and S6).** These are
  whole-brain analyses, so they need the beta maps, not the data in `data/`.
  Their cluster-corrected tests are in `analysis_pipeline/subject_level/`. The
  paper reports whole-brain maps at a cluster-forming threshold of p < .05
  (FWER < .05); the scripts also run and print stricter thresholds, TFCE and a
  gray-matter mask as sensitivity checks.

## Data

- `behavior.csv`: one row per trial per subject
- `rsa_betas.csv`: RSA regression coefficients per subject per ROI
- `rsm/`: 8x8 cross-validated similarity matrices (`.npy`)
- `overlap.csv`: rule/movement overlap per subject per ROI
- `tutorial.csv`: trial-level learning-phase behavior (n = 24)

Noise ceilings are not stored: `6_model_comparison.py` recomputes them from the
matrices, so there is one less file to take on trust.

Four things that will otherwise confuse you:

- **`critical` means *matched***, `uncritical` means *unmatched*. The naming is
  historical and backwards; kept so the data matches the code.
- **Similarity matrices are cross-validated** (odd vs even runs), so entries
  can be negative. That's meaningful, so don't clip them.
- **Condition order** in every matrix is `rot1-4, ref1-4`, numeral = target
  location. Matched trials are targets 1 and 3.
- **n = 27** everywhere, except the tutorial-phase analyses (n = 24; three
  participants are missing tutorial behavior due to a data-saving error).

Full column documentation is in `data/README.md`.

## Citation

> Kang, S., Trach, J. E., & McDougle, S. D. (2026). Abstract representations of
> sensorimotor transformations in human premotor cortex.

## License

MIT (code), CC0 (data). Questions: sanghoon.kang@yale.edu
