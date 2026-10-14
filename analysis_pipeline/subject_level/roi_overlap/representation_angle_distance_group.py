#!/usr/bin/env python3
"""Overlap vs behavioral caching index in left PMd and left M1 (correlations in Figs 5c-d).
Reads the per-subject overlap table from representation_angle_distance_subject.py."""

# %% imports
import os
import sys

import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, "/home/sk2949/project/github/mrdecode/analysis/fmri/glm_mvpa")
from utils.pipeline_config import cfg

MEASURE = "sq_overlap"      # cos1 * cos2 across folds: sign-free, unbiased
ROIS = {"PMd_left_volume": "L PMd", "M1_left_volume": "L M1"}

# %% load
df = pd.read_csv(os.path.join(cfg.rsa_dir,
                              f"subspace_angle_{cfg.event}_{cfg.censor_label}.csv"))
behavior = pd.read_csv(cfg.behav_csv)

# %% caching index: unmatched minus matched |RT_mirror - RT_rotation|
g = (behavior.groupby(["subject", "critical", "manipulation_type"])["rt"]
             .mean().reset_index())
w = g.pivot_table(index=["subject", "critical"], columns="manipulation_type", values="rt")
w["delta"] = (w["mirror"] - w["rotation"]).abs()
ci = w["delta"].reset_index().pivot(index="subject", columns="critical", values="delta")
caching = (ci["uncritical"] - ci["critical"]).rename("caching_index")   # critical = matched

# %% correlations (two-tailed)
for roi, name in ROIS.items():
    m = pd.concat([caching, df[df.roi == roi].set_index("subject")[[MEASURE]]],
                  axis=1, join="inner").dropna()
    x, y = m[MEASURE].to_numpy(), m["caching_index"].to_numpy()
    r, p = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)
    print(f"{name} (n = {len(m)}): Pearson r = {r:.2f}, p = {p:.3f}; "
          f"Spearman rho = {rho:.2f}, p = {p_rho:.3f}")
