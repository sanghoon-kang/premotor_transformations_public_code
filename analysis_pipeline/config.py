"""Shared settings: subjects, ROIs, model RDMs and p/r formatting."""

import os
import numpy as np

# data/ sits beside this folder; run from here or from the repo root.
DATA = next((d for d in (os.path.join("..", "data"), "data")
             if os.path.isfile(os.path.join(d, "README.md"))), None)

if DATA is None:
    raise SystemExit(
        f"Cannot find data/ from {os.getcwd()}\n"
        "Run from analysis_pipeline/ or from the repository root.")

# the 27 subjects in every reported analysis
SUBJECTS = [
    "mrd1011", "mrd1012", "mrd1013", "mrd1014", "mrd1017", "mrd1018", "mrd1019",
    "mrd1020", "mrd1021", "mrd1022", "mrd1023", "mrd1025", "mrd1026", "mrd1027",
    "mrd1028", "mrd1029", "mrd1031", "mrd1033", "mrd1034", "mrd1035", "mrd1036",
    "mrd1037", "mrd1039", "mrd1040", "mrd1041", "mrd1043", "mrd1044",
]

# mrd1011, mrd1013, mrd1029 have no tutorial data, so tutorial analyses are n = 24.

# ROIs
MAIN_ROIS = ["L M1", "R M1", "L PMd", "R PMd", "L PMv", "R PMv",
             "L DLPFC", "R DLPFC", "L LOC", "R LOC"]
SUPP_ROIS = ["L S1", "R S1", "L SPL", "R SPL", "L Hipp", "R Hipp"]
ALL_ROIS = MAIN_ROIS + SUPP_ROIS

# The two ROIs the caching hypothesis is about.
CACHING_ROIS = ["L PMd", "L M1"]

# Condition order in every matrix; matched targets are 1 and 3.
CONDITIONS = ["rot1", "rot2", "rot3", "rot4", "ref1", "ref2", "ref3", "ref4"]

# Column order of the RSA regression coefficients. Do not reorder.
PREDICTORS = ["Rot/Ref", "Target", "Move", "Trial"]

ALPHA = 0.05

# Model RDMs, rows and columns in CONDITIONS order.
# Rot/Ref: same transformation context.
TASK_RULE = np.array([
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1],
], dtype=float)

# Target: same target location.
GOAL_LOCATION = np.array([
    [1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1],
], dtype=float)

# Move: same optimal movement, only on matched targets 1 and 3.
OPTIMAL_MOVEMENT = np.array([
    [1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
], dtype=float)

# Trial: identity.
TRIAL_TYPE = np.eye(8)

MODEL_RDMS = {"Rot/Ref": TASK_RULE, "Target": GOAL_LOCATION,
              "Move": OPTIMAL_MOVEMENT, "Trial": TRIAL_TYPE}

# The 36 entries every fit uses: upper triangle INCLUDING the diagonal.
TRI = np.triu_indices(8)

# Every regression here includes an intercept; drop it and Rot/Ref changes
# (it absorbs the mean similarity), so rsa_betas.csv won't reproduce.


# reporting helpers
def fmt_p(p):
    """APA p: no leading zero, 3 decimals below .01 and near .05, floored at .001."""
    if p < 0.001:
        return "p < .001"
    s = f"{p:.3f}" if p < 0.01 or 0.045 <= p < 0.055 else f"{p:.2f}"
    return "p = " + s.replace("0.", ".", 1)


def fmt_r(r):
    """Correlation with no leading zero."""
    s = f"{r:.2f}"
    return s.replace("0.", ".", 1) if s.startswith("0.") else s.replace("-0.", "-.", 1)
