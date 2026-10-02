"""
pipeline_config.py

Single source of truth for the mrdecode fMRI pipeline. Every script imports
from here; nothing else defines a subject list, a threshold, a path, or a
model matrix. If a value has to change, it changes in one place.

    from pipeline_config import cfg, model_rsms, ROI_SETS

This module is data only. It has no side effects on import, so the notebook,
sbatch scripts and interactive sessions can all import it freely.

History this is meant to prevent: searchlight_rsa_permutation.py ran with 29
subjects while everything else used 27; variance_partition.sh listed 10 ROIs
while noise_ceiling.py listed 12; three scripts disagreed about whether the
model matrices carried a diagonal. All were copy-drift, none were decisions.
"""

import os
from types import SimpleNamespace

import numpy as np

# =============================================================================
# paths
# =============================================================================
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

    # ---- first-level / loading parameters --------------------------------
    # These are the values in the Methods (>70 volumes censored in >3 runs).
    space           = "MNI152NLin2009cAsym",
    space_idx       = 1,                     # index into SPACE_LIST for CLI scripts
    run_censor_thr  = 70,
    subj_runs_thr   = 3,
    event           = "all",
    censor_label    = "no_censoring",
    normalization   = "run_z",

    # ---- RSA --------------------------------------------------------------
    n_cond          = 8,
    atlas           = "juelich",             # what get_roi_mask() is called with

    # ---- searchlight ------------------------------------------------------
    searchlight_radius_vox = 5,

    # ---- parcellation -----------------------------------------------------
    parcel_atlas    = "Schaefer2018",
    parcel_desc     = "400Parcels7Networks",
    parcel_tag      = "schaefer400",
    parcel_min_vox  = 8,

    # ---- group stats ------------------------------------------------------
    alpha           = 0.05,
    fdr_method      = "fdr_bh",
    alternative     = "greater",             # one-tailed, as in the manuscript
)

SPACE_LIST = ["T1w", "MNI152NLin2009cAsym"]

# =============================================================================
# subjects — the 27 used in every reported analysis
# =============================================================================
SUBJECT_LIST = [
    "mrd1011", "mrd1012", "mrd1013", "mrd1014", "mrd1017", "mrd1018", "mrd1019",
    "mrd1020", "mrd1021", "mrd1022", "mrd1023", "mrd1025", "mrd1026", "mrd1027",
    "mrd1028", "mrd1031", "mrd1033", "mrd1034", "mrd1035", "mrd1036", "mrd1037",
    "mrd1039", "mrd1040", "mrd1041", "mrd1043", "mrd1044", "mrd1029",
]
assert len(SUBJECT_LIST) == 27

# =============================================================================
# ROI sets
# =============================================================================
# `regions` are base names; ROI strings passed to get_roi_mask() are derived
# as {region}_{hemi}_volume so a region can't be in one list and not another.
# `family` groups ROIs for FDR: correcting within family keeps the cortical
# p-values unchanged when subcortical ROIs are added.
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
    """['main', 'striatum'] -> ['M1_left_volume', 'M1_right_volume', ...]"""
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

# =============================================================================
# model RSMs — the ONE definition
# =============================================================================
def model_rsms(n=8):
    """
    Returns {name: 8x8 matrix} in the manuscript's parameterization: structural
    predictors keep their diagonal, trial_type = eye(8). Matches
    subject_rsa_full.py, searchlight_rsa.py and figure2_generate.py.

    Order of the dict is the pkl column order and must not change:
        task_rule, goal_location, optimal_movement, trial_type
    """
    task_rule = np.zeros((n, n)); task_rule[:4, :4] = 1; task_rule[4:, 4:] = 1

    goal_location = np.eye(n)
    for a, b in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        goal_location[a, b] = goal_location[b, a] = 1

    # rot & ref share a movement only on the matched targets (T1, T3)
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

# Display names / colors for figures, keyed by model name.
MODEL_DISPLAY = {
    "task_rule":        dict(short="Rot/Ref",          color="#2C6FBB"),
    "optimal_movement": dict(short="Optimal Movement", color="#D1452B"),
    "goal_location":    dict(short="Target Location",  color="#7B4FA3"),
    "trial_type":       dict(short="Trial Type",       color="#6E6E6E"),
}

# =============================================================================
# CLI argument helpers, so every subject-level script takes the same six args
# =============================================================================
def cli_args(subject=None):
    """Positional args for subject_rsa_full.py-style scripts, from config."""
    return [str(cfg.space_idx), str(cfg.run_censor_thr), str(cfg.subj_runs_thr),
            cfg.event, cfg.censor_label]


def _running_interactively():
    """True in a Jupyter/ipykernel session, where sys.argv holds the kernel's
    connection flags (--ip=..., -f=...) rather than our positional args."""
    import sys
    return ("ipykernel" in sys.modules
            or "IPython" in sys.modules and hasattr(sys, "ps1")
            or any(a.startswith("-") for a in sys.argv[1:]))


def cli_arg(i, default, cast=str):
    """
    sys.argv[i] when launched from a shell, otherwise `default`.

    Lets a subject-level script take positional args under sbatch AND run
    top-to-bottom in a notebook without editing anything — which is how the
    same file ends up both in the driver's sbatch stage and in an interactive
    session.
    """
    import sys
    if _running_interactively() or len(sys.argv) <= i:
        return default
    return cast(sys.argv[i])