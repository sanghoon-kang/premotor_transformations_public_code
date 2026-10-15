#!/usr/bin/env python3
# Movement controls (Results; Fig S3): neural vs behavioral Rot/Ref coefficients, and the RSA
# refit with participants' actual hand directions in place of the optimal-movement matrix.
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_1samp

import config

ROIS = ["L PMd", "R PMd", "L PMv", "R PMv", "L M1", "R M1"]

behavior = pd.read_csv(os.path.join(config.DATA, "behavior.csv")).dropna(
    subset=["initial_heading_deg", "target_angle_deg"])
behavior["pos"] = (behavior.manipulation_type == "mirror") * 4 + behavior.which_target - 1


def load_rsm(subject, roi):
    return np.load(os.path.join(config.DATA, "rsm", f"{subject}_{roi.replace(' ', '')}.npy"))


# %% (1) neural vs behavioral Rot/Ref coefficients (Fig S3)
def behavioral_rsm(subject):
    """Odd vs even run similarity of mean initial cursor directions (target + initial error)."""
    b = behavior[behavior.subject == subject]
    cursor = np.deg2rad((b.initial_heading_deg + b.target_angle_deg + 180) % 360 - 180)
    v = pd.DataFrame({"x": np.cos(cursor), "y": np.sin(cursor),
                      "pos": b.pos, "fold": b.run % 2}).groupby(["fold", "pos"]).mean()
    odd, even = v.loc[1].to_numpy(), v.loc[0].to_numpy()
    return 0.5 * (odd @ even.T + even @ odd.T)


def rotref_beta(S):
    """Rot/Ref coefficient with the similarity matrix and model matrices z-scored."""
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    X = np.column_stack([z(config.MODEL_RDMS[p][config.TRI]) for p in config.PREDICTORS])
    return np.linalg.lstsq(X, z(S[config.TRI]), rcond=None)[0][0]


behav = pd.Series({s: rotref_beta(behavioral_rsm(s)) for s in config.SUBJECTS})

rows = []
for roi in ROIS:
    neural = pd.Series({s: rotref_beta(load_rsm(s, roi)) for s in config.SUBJECTS})
    fit = sm.OLS(neural, sm.add_constant(behav)).fit()
    (b0, b1), (t0, t1), (p0, p1) = fit.params, fit.tvalues, fit.pvalues
    rows.append(dict(roi=roi, b1=b1, p_b1=p1,                       # slope, two-tailed
                     b0=b0, p_b0=p0 / 2 if t0 > 0 else 1 - p0 / 2))  # intercept, one-tailed
print(pd.DataFrame(rows).to_string(index=False), "\n")


# %% (2) RSA refit with actual hand directions in place of the optimal-movement matrix
OPTIMAL_HAND = np.array([110, 150, 290, 330, 110, 70, 290, 250], dtype=float)  # deg, CONDITIONS order


def movement_rsm(subject):
    """cos of the angle between trial types' mean hand directions (all runs)."""
    b = behavior[behavior.subject == subject]
    err = np.where(b.manipulation_type == "mirror", -1, 1) * b.initial_heading_deg
    hand = np.deg2rad(OPTIMAL_HAND[b.pos] + err)
    theta = pd.DataFrame({"s": np.sin(hand), "c": np.cos(hand), "pos": b.pos}).groupby("pos").mean()
    theta = np.arctan2(theta.s, theta.c).to_numpy()
    return np.cos(theta[:, None] - theta[None, :])


rows = []
for s in config.SUBJECTS:
    models = dict(config.MODEL_RDMS, Move=movement_rsm(s))
    D = np.column_stack([np.ones(36)] + [models[p][config.TRI] for p in config.PREDICTORS])
    for roi in ROIS:
        coef = np.linalg.lstsq(D, load_rsm(s, roi)[config.TRI], rcond=None)[0]
        rows.append(dict(subject=s, roi=roi, rotref=coef[1]))
refit = pd.DataFrame(rows)

rows = []
for roi in ROIS:
    t, p = ttest_1samp(refit.loc[refit.roi == roi, "rotref"], 0, alternative="greater")
    rows.append(dict(roi=roi, t=t, p=p))
print(pd.DataFrame(rows).to_string(index=False))
