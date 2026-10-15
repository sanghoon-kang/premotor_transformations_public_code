"""Shared settings: subjects, ROIs and model matrices."""

import os
import numpy as np

# data/ sits beside this folder; run from analysis_pipeline/ or the repo root.
DATA = next(d for d in (os.path.join("..", "data"), "data") if os.path.isdir(d))

# the 27 participants in every analysis
SUBJECTS = [
    "mrd1011", "mrd1012", "mrd1013", "mrd1014", "mrd1017", "mrd1018", "mrd1019",
    "mrd1020", "mrd1021", "mrd1022", "mrd1023", "mrd1025", "mrd1026", "mrd1027",
    "mrd1028", "mrd1029", "mrd1031", "mrd1033", "mrd1034", "mrd1035", "mrd1036",
    "mrd1037", "mrd1039", "mrd1040", "mrd1041", "mrd1043", "mrd1044",
]

# ROIs
MAIN_ROIS = ["L M1", "R M1", "L PMd", "R PMd", "L PMv", "R PMv",
             "L DLPFC", "R DLPFC", "L LOC", "R LOC"]
SUPP_ROIS = ["L S1", "R S1", "L SPL", "R SPL", "L Hipp", "R Hipp"]
ALL_ROIS = MAIN_ROIS + SUPP_ROIS

# ROIs for the caching analysis (Fig 5c-d)
CACHING_ROIS = ["L PMd", "L M1"]

# condition order in every matrix; matched targets are 1 and 3
CONDITIONS = ["rot1", "rot2", "rot3", "rot4", "ref1", "ref2", "ref3", "ref4"]

# RSA regression predictors
PREDICTORS = ["Rot/Ref", "Target", "Move", "Trial"]

# model matrices, rows and columns in CONDITIONS order
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

# the 36 entries every fit uses: upper triangle including the diagonal
TRI = np.triu_indices(8)
