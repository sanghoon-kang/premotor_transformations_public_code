#!/usr/bin/env python3
# Rot/Ref coefficient vs mean absolute initial heading error, Fig 4c-d.


# %%
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import config

# Fig 4c-d shows L PMd and L M1; the other four are in the text.
BRAIN_BEHAV_ROIS = ["L PMd", "R PMd", "L PMv", "R PMv", "L M1", "R M1"]

behavior = pd.read_csv(os.path.join(config.DATA, "behavior.csv"))
betas = pd.read_csv(os.path.join(config.DATA, "rsa_betas.csv"))

# %% subject-level error
b = behavior.dropna(subset=["initial_heading_deg"])
cell = (b.assign(abs_err=b.initial_heading_deg.abs())
         .groupby(["subject", "trial", "manipulation_type", "critical"],
                  dropna=False)["abs_err"].mean().reset_index())
error = cell.groupby("subject")["abs_err"].mean()

# %% correlations
print("Transformation-context coefficient vs absolute initial error (Fig 4c-d)")

for roi in BRAIN_BEHAV_ROIS:
    d = betas[betas.roi == roi].set_index("subject")["Rot/Ref"]
    m = pd.concat([d, error.rename("error")], axis=1, join="inner").dropna()
    x, y = m["Rot/Ref"].to_numpy(), m["error"].to_numpy()

    r, p = pearsonr(x, y)
    print(f"{roi:<7} (n = {len(m)})  r = {config.fmt_r(r):>6}, {config.fmt_p(p)}")

# %% supplementary ROIs (S1, SPL, hippocampus)
print()
# "all ps > X" bound: smallest Pearson p over the six, floored.
ps = []
for roi in config.SUPP_ROIS:
    d = betas[betas.roi == roi].set_index("subject")["Rot/Ref"]
    m = pd.concat([d, error.rename("error")], axis=1, join="inner").dropna()
    r, p = pearsonr(m["Rot/Ref"], m["error"])
    ps.append(p)
    print(f"{roi:<7} (n = {len(m)})  r = {config.fmt_r(r):>6}, {config.fmt_p(p)}")
bound = f"{np.floor(min(ps) * 100) / 100:.2f}".replace("0.", ".", 1)
print(f"  all ps > {bound}")
