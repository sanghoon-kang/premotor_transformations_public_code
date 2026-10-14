"""Shared settings: paths, subjects, thresholds, ROI sets and model RSMs.
This is the lab-side source of truth; scripts import from here."""

import os
from types import SimpleNamespace

import numpy as np

# paths
PROJECT_DIR = "/home/sk2949/project/mrdecode/"
GITHUB_DIR  = "/home/sk2949/project/github/mrdecode/"

cfg = SimpleNamespace(
    project_dir     = PROJECT_DIR,
    mri_dir         = os.path.join(PROJECT_DIR, "fmriprep_aroma_volume/output/"),
    rsa_dir         = os.path.join(PROJECT_DIR, "outputs/rsa/"),
    searchlight_dir = os.path.join(PROJECT_DIR, "outputs/searchlight_rsa/"),
    parcel_dir      = os.path.join(PROJECT_DIR, "outputs/parcel_rsa/"),
    fig_dir         = os.path.join(PROJECT_DIR, "outputs/rsa/figure2_panels/"),
    glm_mvpa_dir    = os.path.join(GITHUB_DIR, "analysis/fmri/glm_mvpa/"),
    behav_csv       = os.path.join(GITHUB_DIR, "analysis/trialwise_mt_rt.csv"),
    roi_mvpa_dir    = os.path.join(PROJECT_DIR, "outputs/roi_mvpa/"),
    atlas_dir       = os.path.join(GITHUB_DIR, "analysis/fmri/atlas/juelich_atlas/"),

    # first-level / loading parameters, as in the Methods
    space           = "MNI152NLin2009cAsym",
    space_idx       = 1,                     # index into SPACE_LIST for CLI scripts
    run_censor_thr  = 70,
    subj_runs_thr   = 3,
    event           = "all",
    censor_label    = "no_censoring",
    normalization   = "run_z",

    # RSA
    n_cond          = 8,
    atlas           = "juelich",             # what get_roi_mask() is called with

    # searchlight
    searchlight_radius_vox = 5,

    # parcellation
    parcel_atlas    = "Schaefer2018",
    parcel_desc     = "400Parcels7Networks",
    parcel_tag      = "schaefer400",
    parcel_min_vox  = 8,

    # group stats
    alpha           = 0.05,
    fdr_method      = "fdr_bh",
    alternative     = "greater",             # one-tailed, as in the manuscript
)

SPACE_LIST = ["T1w", "MNI152NLin2009cAsym"]

# the 27 subjects in every reported analysis
SUBJECT_LIST = [
    "mrd1011", "mrd1012", "mrd1013", "mrd1014", "mrd1017", "mrd1018", "mrd1019",
    "mrd1020", "mrd1021", "mrd1022", "mrd1023", "mrd1025", "mrd1026", "mrd1027",
    "mrd1028", "mrd1031", "mrd1033", "mrd1034", "mrd1035", "mrd1036", "mrd1037",
    "mrd1039", "mrd1040", "mrd1041", "mrd1043", "mrd1044", "mrd1029",
]
assert len(SUBJECT_LIST) == 27

# ROI sets; FDR is corrected within each family
ROI_SETS = {
    "main": dict(
        regions=["M1", "PMd", "PMv", "DLPFC", "LOC"],
        family="cortical",
    ),
    "supp_cortical": dict(
        regions=["S1", "SPL", "Hippocampus"],
        family="cortical",
    ),
    "striatum": dict(
        regions=["Caudate", "Putamen", "Accumbens"],
        family="subcortical",
    ),
}

def roi_strings(set_names):
    """ROI set names -> ['M1_left_volume', 'M1_right_volume', ...]"""
    out = []
    for s in set_names:
        for r in ROI_SETS[s]["regions"]:
            out += [f"{r}_left_volume", f"{r}_right_volume"]
    return out

def roi_family(roi):
    region = roi.split("_")[0]
    for s in ROI_SETS.values():
        if region in s["regions"]:
            return s["family"]
    return "unknown"

# model RSMs
def model_rsms(n=8):
    """{name: 8x8 model RSM}; key order is the saved column order, don't change it."""
    task_rule = np.zeros((n, n)); task_rule[:4, :4] = 1; task_rule[4:, 4:] = 1

    goal_location = np.eye(n)
    for a, b in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        goal_location[a, b] = goal_location[b, a] = 1

    # Rot and Ref share a movement only on T1 and T3
    optimal_movement = np.eye(n)
    for a, b in [(0, 4), (2, 6)]:
        optimal_movement[a, b] = optimal_movement[b, a] = 1

    return {
        "task_rule":        task_rule,
        "goal_location":    goal_location,
        "optimal_movement": optimal_movement,
        "trial_type":       np.eye(n),
    }

MODEL_NAMES = list(model_rsms().keys())

# figure names and colors
MODEL_DISPLAY = {
    "task_rule":        dict(short="Rot/Ref",          color="#2C6FBB"),
    "optimal_movement": dict(short="Optimal Movement", color="#D1452B"),
    "goal_location":    dict(short="Target Location",  color="#7B4FA3"),
    "trial_type":       dict(short="Trial Type",       color="#6E6E6E"),
}

# CLI argument helpers
def cli_args(subject=None):
    """Positional args for the subject-level scripts."""
    return [str(cfg.space_idx), str(cfg.run_censor_thr), str(cfg.subj_runs_thr),
            cfg.event, cfg.censor_label]


def _running_interactively():
    """True in a Jupyter/IPython session."""
    import sys
    return ("ipykernel" in sys.modules
            or "IPython" in sys.modules and hasattr(sys, "ps1")
            or any(a.startswith("-") for a in sys.argv[1:]))


def cli_arg(i, default, cast=str):
    """sys.argv[i] from the shell, otherwise `default` (so scripts also run in a notebook)."""
    import sys
    if _running_interactively() or len(sys.argv) <= i:
        return default
    return cast(sys.argv[i])