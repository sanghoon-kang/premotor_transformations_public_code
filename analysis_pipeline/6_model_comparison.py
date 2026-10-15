#!/usr/bin/env python3
# Full vs reduced (no Rot/Ref) RSA model: adjusted R-squared, noise ceiling and summed delta-AIC (Fig S1).
import os

import numpy as np
import pandas as pd

import config


def fit(y, X):
    """OLS with an intercept; returns (adjusted R-squared, RSS)."""
    D = np.column_stack([np.ones(len(y)), X])
    resid = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
    rss = resid @ resid
    n, k = len(y), X.shape[1]
    r2 = 1 - rss / ((y - y.mean()) ** 2).sum()
    return 1 - (1 - r2) * (n - 1) / (n - k - 1), rss


full = np.column_stack([config.MODEL_RDMS[p][config.TRI] for p in config.PREDICTORS])
reduced = np.column_stack([config.MODEL_RDMS[p][config.TRI] for p in config.PREDICTORS
                           if p != "Rot/Ref"])

rows = []
for roi in config.MAIN_ROIS:
    V = np.vstack([np.load(os.path.join(config.DATA, "rsm", f"{s}_{roi.replace(' ', '')}.npy"))[config.TRI]
                   for s in config.SUBJECTS])
    group = V.mean(axis=0)
    adj_f, adj_r, ceiling, d_aic = [], [], [], []
    for y in V:
        af, rss_f = fit(y, full)
        ar, rss_r = fit(y, reduced)
        adj_f.append(af)
        adj_r.append(ar)
        ceiling.append(fit(y, group[:, None])[0])       # noise ceiling: group mean including this participant
        d_aic.append(len(y) * np.log(rss_r / rss_f) - 2)  # AIC(reduced) - AIC(full)
    sem = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
    rows.append(dict(roi=roi, adjR2_full=np.mean(adj_f), sem_full=sem(adj_f),
                     adjR2_reduced=np.mean(adj_r), sem_reduced=sem(adj_r),
                     ceiling=np.mean(ceiling), sum_dAIC=np.sum(d_aic)))
print(pd.DataFrame(rows).to_string(index=False))
