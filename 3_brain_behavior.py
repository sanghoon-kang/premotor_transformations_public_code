#!/usr/bin/env python3
# Transformation-context coding vs task performance — Figure 4c-d.
# Each subject's Rot/Ref coefficient against their average absolute initial
# heading error. Pearson and Spearman, as in the paper.


# %%
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import config

# All six motor/premotor ROIs the paper reports a correlation for. Figures 4c-d
# show left PMd and left M1; the remaining four appear in the text.
BRAIN_BEHAV_ROIS = ["L PMd", "R PMd", "L PMv", "R PMv", "L M1", "R M1"]

behavior = pd.read_csv(os.path.join(config.DATA, "behavior.csv"))
betas = pd.read_csv(os.path.join(config.DATA, "rsa_betas.csv"))

# %% subject-level error
# Mean absolute initial error over each participant's trials. Grouping by
# trial first changes nothing (one row per trial); it is kept to mirror the
# reported analysis. No phase filter: that analysis used every trial.
b = behavior.dropna(subset=["initial_heading_deg"])
cell = (b.assign(abs_err=b.initial_heading_deg.abs())
         .groupby(["subject", "trial", "manipulation_type", "critical"],
                  dropna=False)["abs_err"].mean().reset_index())
error = cell.groupby("subject")["abs_err"].mean()

# %% correlations
print("=" * 70)
print("Transformation-context coefficient vs absolute initial error (Fig 4c-d)")
print("=" * 70)
print()

for roi in BRAIN_BEHAV_ROIS:
    d = betas[betas.roi == roi].set_index("subject")["Rot/Ref"]
    m = pd.concat([d, error.rename("error")], axis=1, join="inner").dropna()
    x, y = m["Rot/Ref"].to_numpy(), m["error"].to_numpy()

    r, p = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)

    # Fisher-z CI, matching what cor.test reports
    z, se = np.arctanh(r), 1 / np.sqrt(len(m) - 3)
    lo, hi = np.tanh([z - 1.96 * se, z + 1.96 * se])

    print(f"{roi}  (n = {len(m)})")
    print(f"  Pearson   r = {config.fmt_r(r):>6}, {config.fmt_p(p)}"
          f"   95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  Spearman  ρ = {config.fmt_r(rho):>6}, {config.fmt_p(p_rho)}")
    print()
# %% supplementary ROIs (S1, SPL, hippocampus)
# For the sentence that transformation encoding in these regions "did not
# correlate with task performance (all ps > X)". Pearson, as in the text; the
# bound is the SMALLEST p over the six, floored so that it stays true.
ps = []
for roi in config.SUPP_ROIS:
    d = betas[betas.roi == roi].set_index("subject")["Rot/Ref"]
    m = pd.concat([d, error.rename("error")], axis=1, join="inner").dropna()
    r, p = pearsonr(m["Rot/Ref"], m["error"])
    ps.append(p)
    print(f"{roi:<7} (n = {len(m)})  Pearson r = {config.fmt_r(r):>6}, {config.fmt_p(p)}")
bound = f"{np.floor(min(ps) * 100) / 100:.2f}".replace("0.", ".", 1)
print(f"  all ps > {bound}")
