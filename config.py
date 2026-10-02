"""
configuration file for storing presets.
"""

import os
import numpy as np

DATA = "data"

if not os.path.isdir(DATA):
    raise SystemExit(
        f"Cannot find {DATA}/ from {os.getcwd()}\n"
        "Run from the repository root, e.g.  python 2_rsa_stats.py")

# 27 subjects included in analyses (three of these (mrd1011, mrd1013, mrd1029) are missing tutorial data)
SUBJECTS = [
    "mrd1011", "mrd1012", "mrd1013", "mrd1014", "mrd1017", "mrd1018", "mrd1019",
    "mrd1020", "mrd1021", "mrd1022", "mrd1023", "mrd1025", "mrd1026", "mrd1027",
    "mrd1028", "mrd1029", "mrd1031", "mrd1033", "mrd1034", "mrd1035", "mrd1036",
    "mrd1037", "mrd1039", "mrd1040", "mrd1041", "mrd1043", "mrd1044",
]

# defining roi lists for different anlyses
MAIN_ROIS = ["L M1", "R M1", "L PMd", "R PMd", "L PMv", "R PMv",
             "L DLPFC", "R DLPFC", "L LOC", "R LOC"]

SUPP_ROIS = ["L S1", "R S1", "L SPL", "R SPL", "L Hipp", "R Hipp"]

ALL_ROIS = MAIN_ROIS + SUPP_ROIS

CACHING_ROIS = ["L PMd", "L M1"]

# condition labels (1 and 3 are matched)
CONDITIONS = ["rot1", "rot2", "rot3", "rot4", "ref1", "ref2", "ref3", "ref4"]

# rsa regression predictor order. don't change pls
PREDICTORS = ["Rot/Ref", "Target", "Move", "Trial"]

ALPHA = 0.05

# defining model matrices

# rot/ref
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

# target location
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

# optimal movement
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

# trial type (identity)
TRIAL_TYPE = np.eye(8)

MODEL_RDMS = {"Rot/Ref": TASK_RULE, "Target": GOAL_LOCATION,
              "Move": OPTIMAL_MOVEMENT, "Trial": TRIAL_TYPE}

TRI = np.triu_indices(8)


# ---------------------------------------------------------------------------
# reporting — kept as functions because four scripts use them
# ---------------------------------------------------------------------------
def fmt_p(p):
    """APA: no leading zero, 2 decimals, 3 below .01, floored at .001 — and 3
    within .005 of .05, so a value can never round across the threshold
    (.0504 would otherwise print as .05)."""
    if p < 0.001:
        return "p < .001"
    s = f"{p:.3f}" if p < 0.01 or 0.045 <= p < 0.055 else f"{p:.2f}"
    return "p = " + s.replace("0.", ".", 1)


def fmt_r(r):
    """No leading zero for a correlation; keeps the sign."""
    s = f"{r:.2f}"
    return s.replace("0.", ".", 1) if s.startswith("0.") else s.replace("-0.", "-.", 1)
