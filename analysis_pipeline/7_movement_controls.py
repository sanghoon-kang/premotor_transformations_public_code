#!/usr/bin/env python3
# Movement controls: behavioral vs neural Rot/Ref (Fig S2), and the main RSA
# refit with actual movements in place of the Move model.

# %%
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_1samp, t as tdist
from statsmodels.stats.multitest import multipletests

import config

# The motor and premotor ROIs, the only ones this figure reports.
ROIS = ["L PMd", "R PMd", "L PMv", "R PMv", "L M1", "R M1"]
ONE_TAILED = True      # the intercept test is directional: neural coding > 0

# initial_heading_deg + target_angle_deg = initial CURSOR direction, not hand.
behavior = pd.read_csv(os.path.join(config.DATA, "behavior.csv"))

# all four model RDMs, as in the main RSA
MODELS = config.MODEL_RDMS
RULE_POSITION = {"rotation": 0, "reflection": 1, "mirror": 1}


# %% ===== Part 1: behavioral vs neural Rot/Ref (Fig S2) =====
def behavioral_rsm(subject):
    """Cross-validated (odd vs even runs) similarity of mean cursor-direction unit vectors."""
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
    """Rot/Ref coefficient, all z-scored (same as fitting with an intercept)."""
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
print("Behavioral vs neural Rot/Ref coding: Fig S2")
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

    # OLS of neural on raw behavioral beta; x not centered, so b0 = coding at zero behavior.
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
print("structure is absent, i.e. not explained by movement differences.")


# %% ===== Part 2: actual movements in place of the optimal-movement model =====
# Otherwise as 2_rsa_stats.py. Optimal hand direction (deg, CONDITIONS order):
# rotation hand = target + 40, reflection hand = 180 - target.
TARGET = np.array([70, 110, 250, 290])
OPTIMAL_HAND = np.r_[(TARGET + 40) % 360, (180 - TARGET) % 360].astype(float)


# %% each participant's movement similarity matrix
def movement_rsm(subject):
    """cos(theta_i - theta_j) between circular-mean hand directions."""
    b = behavior[behavior.subject == subject].dropna(
        subset=["initial_heading_deg"])
    is_ref = (b.manipulation_type == "mirror").to_numpy()
    pos = np.where(is_ref, 4, 0) + b.which_target.astype(int).to_numpy() - 1
    err = np.where(is_ref, -1, 1) * b.initial_heading_deg.to_numpy()
    hand = np.deg2rad(OPTIMAL_HAND[pos] + err)
    theta = np.array([np.arctan2(np.sin(hand[pos == k]).mean(),
                                 np.cos(hand[pos == k]).mean())
                      for k in range(8)])
    return np.cos(theta[:, None] - theta[None, :])


MOVE = {s: movement_rsm(s) for s in config.SUBJECTS}


def predictors(subject):
    """The main model, with the empirical matrix in place of Move."""
    return {"Rot/Ref": config.MODEL_RDMS["Rot/Ref"],
            "Target": config.MODEL_RDMS["Target"],
            "Move": MOVE[subject],
            "Trial": config.MODEL_RDMS["Trial"]}


# %% collinearity: does the empirical movement matrix overlap with Rot/Ref?
r = [np.corrcoef(config.MODEL_RDMS["Rot/Ref"][config.TRI],
                 MOVE[s][config.TRI])[0, 1] for s in config.SUBJECTS]
print("\n" + "=" * 78)
print("Main RSA refit with actual movements in place of the optimal Move model")
print("=" * 78)
print(f"r(Rot/Ref, empirical Move) over the 36 entries: mean {np.mean(r):.2f}, "
      f"range [{min(r):.2f}, {max(r):.2f}]")
print("one-sample t vs 0, one-tailed, FDR-corrected across all 16 ROIs\n")

# %% refit
refit = []
for s in config.SUBJECTS:
    X = predictors(s)
    D = np.column_stack([np.ones(len(config.TRI[0]))]
                        + [X[p][config.TRI] for p in config.PREDICTORS])
    for roi in config.ALL_ROIS:
        f = os.path.join(config.DATA, "rsm", f"{s}_{roi.replace(' ', '')}.npy")
        if os.path.exists(f):
            coef = np.linalg.lstsq(D, np.load(f)[config.TRI], rcond=None)[0]
            refit.append(dict(subject=s, roi=roi,
                             **dict(zip(config.PREDICTORS, coef[1:]))))
betas = pd.DataFrame(refit)

for predictor in ("Rot/Ref", "Move"):
    out = []
    for roi in config.ALL_ROIS:
        v = betas.loc[betas.roi == roi, predictor].to_numpy()
        t, p = ttest_1samp(v, 0.0, alternative="greater")
        out.append(dict(roi=roi, n=len(v), t=t, p=p))
    out = pd.DataFrame(out)
    out["p_fdr"] = multipletests(out.p, alpha=config.ALPHA, method="fdr_bh")[1]

    print(f"--- {predictor} ---")
    for _, d in out.iterrows():
        print(f"  {d.roi:<9} t({int(d.n) - 1}) = {d.t:5.2f}, "
              f"{config.fmt_p(d.p):<11} FDR {config.fmt_p(d.p_fdr):<11} "
              f"{'*' if d.p_fdr < config.ALPHA else ''}")
    print()
