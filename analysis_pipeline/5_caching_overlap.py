#!/usr/bin/env python3
# Neural rule-movement overlap vs behavioral caching index (Fig 5c-d).
import os

import pandas as pd
from scipy.stats import pearsonr, spearmanr

import config

behavior = pd.read_csv(os.path.join(config.DATA, "behavior.csv"))
overlap = pd.read_csv(os.path.join(config.DATA, "overlap.csv"))

# caching index: unmatched minus matched |RT_reflection - RT_rotation|, as in 4_caching_behavior.R
# (critical = matched, uncritical = unmatched)
b = behavior.dropna(subset=["rt"])
ctx = b.pivot_table(index=["subject", "critical"], columns="manipulation_type",
                    values="rt", aggfunc="mean")
delta = (ctx["mirror"] - ctx["rotation"]).abs().unstack("critical")
caching = (delta["uncritical"] - delta["critical"]).rename("caching")

rows = []
for roi in config.CACHING_ROIS:
    m = pd.concat([overlap[overlap.roi == roi].set_index("subject")["sq_overlap"], caching],
                  axis=1, join="inner").dropna()
    r, p = pearsonr(m.sq_overlap, m.caching)
    rho, p_rho = spearmanr(m.sq_overlap, m.caching)
    rows.append(dict(roi=roi, n=len(m), r=r, p=p, rho=rho, p_rho=p_rho))
print(pd.DataFrame(rows).to_string(index=False))
