#!/usr/bin/env python3
# Full vs reduced (no Rot/Ref) RSA model against the noise ceiling, Fig S1.
# Every fit includes an intercept (see config.py).


# %%
import os

import numpy as np

import config


def adjusted_r2(y, X):
    """OLS with an intercept; returns (adjusted R-squared, RSS)."""
    D = np.column_stack([np.ones(len(y)), X])
    resid = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
    rss = float(resid @ resid)
    n, k = len(y), X.shape[1]
    r2 = 1 - rss / float(((y - y.mean()) ** 2).sum())
    return 1 - (1 - r2) * (n - 1) / (n - k - 1), rss


def sem(v):
    return np.std(v, ddof=1) / np.sqrt(len(v))


# %% design matrices: the 36 upper-triangle entries, including the diagonal
full = np.column_stack([config.MODEL_RDMS[p][config.TRI]
                        for p in config.PREDICTORS])
reduced = np.column_stack([config.MODEL_RDMS[p][config.TRI]
                           for p in config.PREDICTORS if p != "Rot/Ref"])

# %%
print("=" * 94)
print("Model comparison: full vs reduced (no Rot/Ref), Fig S1")
print("=" * 94)
print(f"{'ROI':<9} {'n':>3} {'adjR2 full':>17} {'adjR2 red':>16} "
      f"{'ceiling':>8} {'sum dAIC':>9}  evidence")
print("-" * 94)

for roi in config.MAIN_ROIS:
    V = np.vstack([np.load(f)[config.TRI] for f in
                   (os.path.join(config.DATA, "rsm",
                                 f"{s}_{roi.replace(' ', '')}.npy")
                    for s in config.SUBJECTS)
                   if os.path.exists(f)])
    group = V.mean(axis=0)

    # Upper noise ceiling (Nili et al. 2014): fit on the group mean including this subject.
    adj_f, adj_r, ceiling, d_aic = [], [], [], []
    for i, y in enumerate(V):
        af, rss_f = adjusted_r2(y, full)
        ar, rss_r = adjusted_r2(y, reduced)
        adj_f.append(af)
        adj_r.append(ar)
        ceiling.append(adjusted_r2(y, group[:, None])[0])
        # delta-AIC = AIC(reduced) - AIC(full)
        d_aic.append(len(y) * np.log(rss_r / rss_f) - 2)

    # Summed over participants, not averaged.
    total = sum(d_aic)
    # Burnham & Anderson bands, applied to the summed evidence
    band = ("very strong" if total > 10 else "strong" if total > 7 else
            "moderate" if total > 4 else "weak" if total > 2 else "negligible")

    print(f"{roi:<9} {len(V):>3} "
          f"{np.mean(adj_f):>10.3f} ±{sem(adj_f):<5.3f} "
          f"{np.mean(adj_r):>9.3f} ±{sem(adj_r):<5.3f} "
          f"{np.mean(ceiling):>8.3f} "
          f"{total:>9.1f}  {band}")

# Descriptive only, not a significance test.
print("\nAdjusted R-squared is mean ± 1 s.e.m. across participants (Fig S1a error")
print("bars); the ceiling is its upper bound. Positive summed delta-AIC favors")
print("the full model. Bands (Burnham & Anderson): <2 negligible, 2-4 weak,")
print("4-7 moderate, 7-10 strong, >10 very strong.")
