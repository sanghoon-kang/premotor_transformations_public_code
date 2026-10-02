# Abstract representations of sensorimotor transformations in human premotor cortex

Data and analysis code for Kang, Trach, & McDougle.

The nine numbered scripts in this folder reproduce every statistic reported in
the paper.

The numbered scripts start from the subject-level data in `data/`. The cluster pipeline that built `data/` from the raw
images is in `preprocessing/` for reference.

## Setup

Behavioral analyses are in R, neuroimaging analyses in Python, matching how
they were run and how the Methods describe them.

```bash
Rscript 1_behavior.R        # R 4.2.0
pip install -r requirements.txt      # Python 3.10.13
python 2_rsa_stats.py
```

Run every script from the repository root — paths are relative to it. Each
prints its results to the terminal; they don't depend on each other.

The scripts run top to bottom, with `# %%` cell markers, so they work the same
in a terminal or an interactive window. Options are constants near the top of
each file (e.g. `SUPP = False` in `2_rsa_stats.py`) — edit those rather than
passing command-line flags. No function definitions to call, no imports
between scripts: open a file and read it from the top.

Both R scripts use base R only — there is nothing to install for those.

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
| `7_behavioral_rsa.py` | Python | Behavioral vs neural Rot/Ref coding (Fig S2) |
| `8_region_anova.py` | Python | Premotor vs M1 difference-of-differences, region x hemisphere |
| `9_mds.py` | Python | MDS coordinates and fit for Figs 4a-b and S3 |

`config.py` holds the subject list, ROI names and model matrices. Nothing else
defines them.

Everything above runs from `data/`. The pipeline that produced `data/` from the
raw MRI — preprocessing, first-level GLM, mask generation, the voxelwise and
searchlight RSA, and the export itself — is in `preprocessing/`, for reading
rather than running: those scripts need our cluster and are not required for
anything in the table.

### Not covered here, and why

- **Figures 6 and S5 (searchlight).** The maps are reported unthresholded, as
  visualizations — no statistic is quoted, so there is nothing to recompute.
  The code that produced them is in `preprocessing/`.
- **Figures 1 and 2a (task schematics).** Drawn by hand.

## Data

- `behavior.csv` — one row per trial per subject
- `rsa_betas.csv` — RSA regression coefficients per subject per ROI
- `rsm/` — 8×8 cross-validated similarity matrices (`.npy`)
- `overlap.csv` — rule/movement overlap per subject per ROI
- `tutorial.csv` — trial-level learning-phase behavior (n = 24)

Noise ceilings are not stored: `6_model_comparison.py` recomputes them from the
matrices, so there is one less file to take on trust.

Four things that will otherwise confuse you:

- **`critical` means *matched***, `uncritical` means *unmatched*. The naming is
  historical and backwards; kept so the data matches the code.
- **Similarity matrices are cross-validated** (odd vs even runs), so entries
  can be negative. That's meaningful — don't clip them.
- **Condition order** in every matrix is `rot1–4, ref1–4`, numeral = target
  location. Matched trials are targets 1 and 3.
- **n = 27** everywhere, except the tutorial-phase analyses (n = 24; three
  participants are missing tutorial behavior due to a data-saving error).

Full column documentation is in `data/README.md`.

Questions: sanghoon.kang@yale.edu
