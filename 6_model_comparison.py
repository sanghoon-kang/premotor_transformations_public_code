#!/usr/bin/env python3
# Full vs reduced RSA model, against the noise ceiling — Figure S1.
# Each participant's matrix is fit twice, with all four model RDMs and without
# Rot/Ref. Every fit includes an intercept (see config.py).


# %%
import os

import numpy as np

import config


def adjusted_r2(y, X):
    """OLS with an intercept. Returns (ADJUSTED R-squared, RSS).

    k counts predictors, not the intercept, in the usual adjustment
    1 - (1 - R2)(n - 1)/(n - k - 1). Adjusted because the full model has four
    predictors and the noise ceiling one: on raw R-squared the model would gain
    on the ceiling from its extra parameters alone. The ceilings are adjusted
    the same way, with k = 1."""
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
print("Model comparison: full vs reduced (no Rot/Ref) — Fig S1")
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

    # Noise ceiling, Nili et al. (2014): the upper bound, a regression of the
    # participant's matrix on the group average INCLUDING that participant.
    # The ceiling measures how well the structure shared across participants
    # predicts one of them. The model, fit to each participant's own matrix,
    # can also capture structure particular to that participant, so its fit is
    # not bounded by the ceiling and in several ROIs here exceeds it.
    adj_f, adj_r, ceiling, d_aic = [], [], [], []
    for i, y in enumerate(V):
        af, rss_f = adjusted_r2(y, full)
        ar, rss_r = adjusted_r2(y, reduced)
        adj_f.append(af)
        adj_r.append(ar)
        ceiling.append(adjusted_r2(y, group[:, None])[0])
        # delta-AIC = AIC(reduced) - AIC(full). Both models carry the same
        # intercept, so only the dropped Rot/Ref predictor adds to the penalty.
        d_aic.append(len(y) * np.log(rss_r / rss_f) - 2)

    # SUMMED, not averaged: this is one joint model with participant-specific
    # coefficients, so the evidence adds. A sum is a single quantity, not a
    # sample mean, so it carries no error bar.
    total = sum(d_aic)
    # Burnham & Anderson bands, applied to the summed evidence
    band = ("very strong" if total > 10 else "strong" if total > 7 else
            "moderate" if total > 4 else "weak" if total > 2 else "negligible")

    print(f"{roi:<9} {len(V):>3} "
          f"{np.mean(adj_f):>10.3f} ±{sem(adj_f):<5.3f} "
          f"{np.mean(adj_r):>9.3f} ±{sem(adj_r):<5.3f} "
          f"{np.mean(ceiling):>8.3f} "
          f"{total:>9.1f}  {band}")

# Descriptive, not a significance test: the 36 entries of an RDM are not
# independent observations (each shares a condition with 13 others), which is
# exactly the assumption an information criterion makes.
print("\nAdjusted R-squared is mean ± 1 s.e.m. across participants (Fig S1a error")
print("bars); the ceiling is its upper bound. Positive summed delta-AIC favors")
print("the full model. Bands (Burnham & Anderson): <2 negligible, 2-4 weak,")
print("4-7 moderate, 7-10 strong, >10 very strong.")
