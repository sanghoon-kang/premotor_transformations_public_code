#!/usr/bin/env python3
# Neural rule/movement overlap vs behavioral caching, Fig 5c-d.
# Overlap needs voxel patterns, so it's read from overlap.csv, not recomputed.


# %%
import os

import pandas as pd
from scipy.stats import pearsonr, spearmanr

import config

behavior = pd.read_csv(os.path.join(config.DATA, "behavior.csv"))
# sq_overlap = product of two cross-fold cosines (unbiased cos^2).
overlap = pd.read_csv(os.path.join(config.DATA, "overlap.csv"))

# %% behavioral caching index
# Same index as 4_caching_behavior.R: unmatched minus matched |RT_ref - RT_rot|.
b = behavior.dropna(subset=["rt"])
cell = (b.groupby(["subject", "critical", "manipulation_type"])["rt"]
          .mean().reset_index())
ctx = cell.pivot_table(index=["subject", "critical"],
                       columns="manipulation_type", values="rt")
ctx["delta"] = (ctx["mirror"] - ctx["rotation"]).abs()
wide = ctx["delta"].reset_index().pivot(index="subject", columns="critical",
                                        values="delta")
# critical = matched, uncritical = unmatched (historical naming)
caching = (wide["uncritical"] - wide["critical"]).rename("caching")

# %% correlations
print("Neural overlap vs behavioral caching index (Fig 5c-d)")
print()

for roi in config.CACHING_ROIS:
    d = overlap[overlap.roi == roi].set_index("subject")["sq_overlap"]
    m = pd.concat([d, caching], axis=1, join="inner").dropna()
    x, y = m["sq_overlap"].to_numpy(), m["caching"].to_numpy()

    r, p = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)

    print(f"{roi}  (n = {len(m)})")
    print(f"  Pearson   r = {config.fmt_r(r):>6}, {config.fmt_p(p)}")
    print(f"  Spearman  ρ = {config.fmt_r(rho):>6}, {config.fmt_p(p_rho)}")
    print()
