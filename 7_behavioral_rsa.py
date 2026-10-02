#!/usr/bin/env python3
# Behavioral vs neural transformation-context coding — Figure S2.
# A behavioral similarity matrix is built from each participant's initial
# movement directions, regressed on the same four model RDMs as the neural
# matrices (z-scored, so behavioral and neural coefficients share a scale), and
# the two Rot/Ref coefficients are then regressed against each other across
# participants.

# %%
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t as tdist
from statsmodels.stats.multitest import multipletests

import config

# The motor and premotor ROIs, the only ones this figure reports.
ROIS = ["L PMd", "R PMd", "L PMv", "R PMv", "L M1", "R M1"]
ONE_TAILED = True      # the intercept test is directional: neural coding > 0

# Needs target_angle_deg and which_target as well as the usual columns.
# initial_heading_deg is the signed error of the CURSOR relative to the target
# (perfect compensation = 0), so adding target_angle_deg gives the direction
# the cursor initially travelled — the movement in visual space, not the hand's
# direction, which differs from it by the rotation or reflection.
behavior = pd.read_csv(os.path.join(config.DATA, "behavior.csv"))

# All four model RDMs, as in the main RSA and in the script that produced
# Fig S2.
MODELS = config.MODEL_RDMS
RULE_POSITION = {"rotation": 0, "reflection": 1, "mirror": 1}


# %% behavioral similarity matrix
def behavioral_rsm(subject):
    """Cross-validated similarity of initial movement direction between the 8
    conditions.

    Each condition's "pattern" is the MEAN UNIT VECTOR of the initial cursor
    directions the participant produced: signed error + target angle gives the
    direction, which becomes (cos, sin). Averaging those within a condition
    gives a 2-D vector whose direction is the mean heading and whose length
    falls as the headings scatter.

    Cross-validated the same way as the neural matrices: odd runs against
    even, averaged over both orderings, so the noise in the two folds is
    independent and the diagonal is not inflated.
    """
    b = behavior[behavior.subject == subject].dropna(
        subset=["initial_heading_deg", "target_angle_deg"]).copy()
    b = b[b.manipulation_type.astype(str).str.strip().str.lower()
           .isin(RULE_POSITION)]
    heading = np.deg2rad((b.initial_heading_deg.astype(float)
                          + b.target_angle_deg.astype(float) + 180.0)
                         % 360.0 - 180.0)
    b["pos"] = [RULE_POSITION[str(m).strip().lower()] * 4 + (int(t) - 1)
                for m, t in zip(b.manipulation_type, b.which_target)]
    b["fold"] = b.run.astype(int) % 2
    b["cx"], b["cy"] = np.cos(heading), np.sin(heading)

    mean_vec = {k: g[["cx", "cy"]].mean().to_numpy()
                for k, g in b.groupby(["pos", "fold"])}

    S = np.full((8, 8), np.nan)
    for i in range(8):
        for j in range(8):
            if all(k in mean_vec for k in [(i, 0), (j, 0), (i, 1), (j, 1)]):
                S[i, j] = 0.5 * float(mean_vec[(i, 0)] @ mean_vec[(j, 1)]
                                      + mean_vec[(i, 1)] @ mean_vec[(j, 0)])
    return S


def rotref_beta(S):
    """Rot/Ref coefficient from regressing a similarity matrix on the models.

    Both predictors and outcome are z-scored, so the coefficient is comparable
    between the behavioral and neural matrices even though they are on
    completely different scales. No explicit intercept is needed: z-scoring
    centres both sides, which makes this identical to a fit that includes one.
    """
    y = S[config.TRI]
    X = np.column_stack([M[config.TRI] for M in MODELS.values()])
    keep = np.isfinite(y)
    if keep.sum() < 4:
        return np.nan
    y, X = y[keep], X[keep]

    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    Xz = np.column_stack([z(X[:, k]) for k in range(X.shape[1])])
    coef, *_ = np.linalg.lstsq(Xz, z(y), rcond=None)
    return coef[list(MODELS).index("Rot/Ref")]


# %% behavioral beta per subject, neural beta per subject per ROI
beh = pd.Series({s: rotref_beta(behavioral_rsm(s)) for s in config.SUBJECTS},
                name="behavioral")

print("=" * 78)
print("Behavioral vs neural Rot/Ref coding — Fig S2")
print("=" * 78)
print(f"behavioral beta: n = {int(beh.notna().sum())}, "
      f"mean {beh.mean():.3f}\n")

rows = []
for roi in ROIS:
    neu = {}
    for s in config.SUBJECTS:
        f = os.path.join(config.DATA, "rsm", f"{s}_{roi.replace(' ', '')}.npy")
        if os.path.exists(f):
            neu[s] = rotref_beta(np.load(f))
    m = pd.concat([beh, pd.Series(neu, name="neural")], axis=1).dropna()
    x, y = m.behavioral.to_numpy(), m.neural.to_numpy()

    # OLS of neural on RAW behavioral beta. x is NOT centered, so the
    # intercept is neural coding where behavioral rule structure is literally
    # zero — a non-significant slope with a positive intercept is the result
    # the paper reports for premotor cortex.
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    s2 = float(resid @ resid) / (n - 2)
    xbar, Sxx = x.mean(), float(((x - x.mean()) ** 2).sum())
    se_slope = np.sqrt(s2 / Sxx)
    se_intercept = np.sqrt(s2 * (1.0 / n + xbar ** 2 / Sxx))

    t_slope, t_intercept = slope / se_slope, intercept / se_intercept
    p_slope = 2 * (1 - tdist.cdf(abs(t_slope), n - 2))
    p_intercept = 2 * (1 - tdist.cdf(abs(t_intercept), n - 2))
    if ONE_TAILED:
        p_intercept = p_intercept / 2 if intercept > 0 else 1 - p_intercept / 2

    r, p_r = pearsonr(x, y)
    rows.append(dict(roi=roi, n=n, r=r, p_r=p_r,
                     b1=slope, t_b1=t_slope, p_b1=p_slope,
                     b0=intercept, t_b0=t_intercept, p_b0=p_intercept))

tab = pd.DataFrame(rows)

# %% FDR across ROIs, within each test family
for col in ("p_r", "p_b1", "p_b0"):
    tab["q" + col[1:]] = multipletests(tab[col], alpha=config.ALPHA,
                                       method="fdr_bh")[1]

print(f"{'ROI':<8} {'n':>3} {'r':>6} {'slope b1':>9} {'p(b1)':>8} "
      f"{'intercept b0':>13} {'p(b0)':>8}  (b0 "
      f"{'one' if ONE_TAILED else 'two'}-tailed)")
print("-" * 78)
for _, d in tab.iterrows():
    print(f"{d.roi:<8} {int(d.n):>3} {config.fmt_r(d.r):>6} "
          f"{d.b1:>9.3f} {config.fmt_p(d.p_b1):>8} "
          f"{d.b0:>13.3f} {config.fmt_p(d.p_b0):>8}")

print("\nFDR-corrected across ROIs:")
for _, d in tab.iterrows():
    print(f"  {d.roi:<8} r {config.fmt_p(d.q_r):<11} "
          f"slope {config.fmt_p(d.q_b1):<11} "
          f"intercept {config.fmt_p(d.q_b0)}")

print("\nA non-significant slope with a positive intercept means neural")
print("transformation-context coding is present where behavioral rule")
print("structure is absent — i.e. not explained by movement differences.")
