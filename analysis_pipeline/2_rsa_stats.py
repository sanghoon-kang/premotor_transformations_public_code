#!/usr/bin/env python3
# RSA regression coefficients vs 0, one-tailed, FDR across ROIs (Figs 3 and S5).
import os

import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

import config

betas = pd.read_csv(os.path.join(config.DATA, "rsa_betas.csv"))

for predictor in config.PREDICTORS:
    rows = []
    for roi in config.ALL_ROIS:
        v = betas.loc[betas.roi == roi, predictor].dropna()
        t, p = ttest_1samp(v, 0, alternative="greater")
        rows.append(dict(roi=roi, df=len(v) - 1, t=t, p=p))
    out = pd.DataFrame(rows)
    out["p_fdr"] = multipletests(out.p, method="fdr_bh")[1]
    print(predictor)
    print(out.to_string(index=False), "\n")
