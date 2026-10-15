"""Shared settings for the overlap analysis: paths, loading parameters, subjects and ROIs."""

import os
from types import SimpleNamespace

PROJECT_DIR = "/home/sk2949/project/mrdecode/"

cfg = SimpleNamespace(
    rsa_dir         = os.path.join(PROJECT_DIR, "outputs/rsa/"),

    # loading parameters, as in the Methods
    space_idx       = 1,                     # index into SPACE_LIST ("MNI152NLin2009cAsym")
    run_censor_thr  = 70,
    subj_runs_thr   = 6,                     # exclude a participant with < 6 of 9 runs left (> 3 runs over run_censor_thr)
    event           = "all",
    censor_label    = "no_censoring",
    normalization   = "run_z",

    n_cond          = 8,
    atlas           = "juelich",             # what get_roi_mask() is called with
)

# the 27 subjects in every reported analysis
SUBJECT_LIST = [
    "mrd1011", "mrd1012", "mrd1013", "mrd1014", "mrd1017", "mrd1018", "mrd1019",
    "mrd1020", "mrd1021", "mrd1022", "mrd1023", "mrd1025", "mrd1026", "mrd1027",
    "mrd1028", "mrd1031", "mrd1033", "mrd1034", "mrd1035", "mrd1036", "mrd1037",
    "mrd1039", "mrd1040", "mrd1041", "mrd1043", "mrd1044", "mrd1029",
]

ROI_SETS = {
    "main": ["M1", "PMd", "PMv", "DLPFC", "LOC"],
    "supp_cortical": ["S1", "SPL", "Hippocampus"],
}


def roi_strings(set_names):
    """ROI set names -> ['M1_left_volume', 'M1_right_volume', ...]"""
    out = []
    for s in set_names:
        for r in ROI_SETS[s]:
            out += [f"{r}_left_volume", f"{r}_right_volume"]
    return out


def cli_arg(i, default, cast=str):
    """sys.argv[i] from the shell, otherwise `default` (so the script also runs interactively)."""
    import sys
    interactive = "ipykernel" in sys.modules or ("IPython" in sys.modules and hasattr(sys, "ps1"))
    if interactive or len(sys.argv) <= i:
        return default
    return cast(sys.argv[i])
