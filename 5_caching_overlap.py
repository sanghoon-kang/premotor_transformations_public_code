#!/usr/bin/env python3
# Neural rule/movement overlap vs behavioral caching — Figure 5c-d.
# The overlap measure comes from voxel-level patterns, which the 8x8 matrices
# do not contain, so it is read from data/overlap.csv rather than recomputed.


# %%
import os

import pandas as pd
from scipy.stats import pearsonr, spearmanr

import config

behavior = pd.read_csv(os.path.join(config.DATA, "behavior.csv"))
# sq_overlap = cos1 x cos2, the product of two cross-fold cosines (rule from
# odd runs vs movement from even runs, and the reverse). The product is
# unbiased for cos-squared theta because the two carry independent noise,
# which cancels in expectation; squaring a single noisy cosine would inflate
# it by its variance. The cost is that magnitude is attenuated by the vectors'
# reliability — so the zero point and sign are exact, absolute size is not.
overlap = pd.read_csv(os.path.join(config.DATA, "overlap.csv"))

# %% behavioral caching index
# The same quantity 4_caching_behavior.R computes: within each trial type,
# |RT_reflection - RT_rotation|, then unmatched minus matched. Larger = behavior
# converged more across contexts on matched trials.
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
print("=" * 70)
print("Neural overlap vs behavioral caching index (Fig 5c-d)")
print("=" * 70)
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
