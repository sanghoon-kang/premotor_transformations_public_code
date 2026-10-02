#!/usr/bin/env python3
# Builds the published data/ folder from the lab's analysis outputs.
#
# CLUSTER ONLY, run once before publishing. Nobody outside the lab can run
# this; it is here so the provenance of every file in data/ is on the record.
# Set RSA_DIR, BEHAV_CSV and LEARN_DIR below, then run it with the REPOSITORY
# ROOT as the working directory (it imports config and writes to data/).
#
# It is also the only surviving account of how tutorial.csv was built: the
# original tutorial analysis script was lost, so process_trial() below is the
# definition of those columns, not just a copy of it.
#
# .npy rather than .pkl because unpickling a file executes code in it, and
# asking strangers to do that is a poor default.


# %%
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import config

# ---------------------------------------------------------------------------
# EDIT THESE. The only absolute paths in the repository.
# ---------------------------------------------------------------------------
RSA_DIR = "/home/sk2949/project/mrdecode/outputs/rsa"
BEHAV_CSV = "/home/sk2949/project/github/mrdecode/analysis/trialwise_mt_rt.csv"
# raw task output holding the tutorial (learn-phase) frame files
LEARN_DIR = "/home/sk2949/project/github/mrdecode/task/data_tutorial"
OUT = config.DATA
EVENT = "all"

# Mask filename -> the label the paper uses.
ROI_FILES = {
    "M1_left_volume": "L M1", "M1_right_volume": "R M1",
    "PMd_left_volume": "L PMd", "PMd_right_volume": "R PMd",
    "PMv_left_volume": "L PMv", "PMv_right_volume": "R PMv",
    "DLPFC_left_volume": "L DLPFC", "DLPFC_right_volume": "R DLPFC",
    "LOC_left_volume": "L LOC", "LOC_right_volume": "R LOC",
    "S1_left_volume": "L S1", "S1_right_volume": "R S1",
    "SPL_left_volume": "L SPL", "SPL_right_volume": "R SPL",
    "Hippocampus_left_volume": "L Hipp", "Hippocampus_right_volume": "R Hipp",
}

# Columns kept in the public behavior file. Opt in, not out.
BEHAVIOR_COLUMNS = [
    "subject", "run", "trial", "trial_within_type", "phase",
    "manipulation_type", "critical", "target_location",
    "initial_heading_deg", "endpoint_angle_deg", "rt", "mt",
    # needed by 7_behavioral_rsa.py: the signed (cursor) error plus the target
    # angle gives the cursor's direction, and which_target the condition index
    "target_angle_deg", "which_target",
]
OVERLAP_COLUMNS = ["subject", "roi", "n_vox", "sq_overlap", "cos1", "cos2",
                   "mag_rule", "mag_move", "rel_rule", "rel_move"]

os.makedirs(os.path.join(OUT, "rsm"), exist_ok=True)
print(f"writing to {OUT}\n")

# %% check config.py still agrees with the lab's pipeline_config.py
# The repo defines its own subjects and model matrices so it can stand alone,
# which means the same facts live in two files — exactly the copy-drift
# pipeline_config.py exists to prevent. Skipped off-cluster.
try:
    import pipeline_config as lab
except ImportError:
    lab = None

if lab is not None:
    problems = []
    if sorted(lab.SUBJECT_LIST) != sorted(config.SUBJECTS):
        problems.append(
            f"subjects differ — only in pipeline_config: "
            f"{sorted(set(lab.SUBJECT_LIST) - set(config.SUBJECTS))}; "
            f"only in config.py: "
            f"{sorted(set(config.SUBJECTS) - set(lab.SUBJECT_LIST))}")
    # pipeline_config keys models by analysis name, config.py by column label,
    # in the same order — compare positionally.
    for (ln, lm), (rn, rm) in zip(lab.model_rsms().items(),
                                  config.MODEL_RDMS.items()):
        if not np.array_equal(lm, rm):
            problems.append(f"model matrix differs: {ln} (lab) vs {rn} (repo)")
    if problems:
        print("[DRIFT] config.py disagrees with pipeline_config.py:")
        for p in problems:
            print(f"    {p}")
        print("    Fix before exporting — the repo would otherwise publish")
        print("    different definitions from the ones used for the results.\n")
    else:
        print("config.py matches pipeline_config.py (subjects, matrices)\n")

# %% behavior.csv
b = pd.read_csv(BEHAV_CSV)
keep = [c for c in BEHAVIOR_COLUMNS if c in b.columns]
dropped = sorted(set(b.columns) - set(keep))
b[keep].to_csv(os.path.join(OUT, "behavior.csv"), index=False)
print(f"behavior.csv      {len(b)} rows, {b.subject.nunique()} subjects")
if dropped:
    print(f"                  dropped: {dropped}")

# %% rsa_betas.csv
rows = []
for s in config.SUBJECTS:
    for mask, label in ROI_FILES.items():
        fp = os.path.join(
            RSA_DIR, f"{s}_{mask}_regression_coefficients_{EVENT}_full.pkl")
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                c = np.asarray(pickle.load(f), dtype=float)[:4]
            rows.append({"subject": s, "roi": label,
                         **dict(zip(config.PREDICTORS, c))})
betas = pd.DataFrame(rows)
betas.to_csv(os.path.join(OUT, "rsa_betas.csv"), index=False)
print(f"rsa_betas.csv     {len(betas)} rows, {betas.roi.nunique()} ROIs")

# %% rsm/
n = 0
for s in config.SUBJECTS:
    for mask, label in ROI_FILES.items():
        fp = os.path.join(
            RSA_DIR, f"{s}_{mask}_correlation_matrix_full_{EVENT}.pkl")
        if not os.path.exists(fp):
            continue
        with open(fp, "rb") as f:
            M = np.asarray(pickle.load(f), dtype=float)
        np.save(os.path.join(OUT, "rsm", f"{s}_{label.replace(' ', '')}.npy"), M)
        n += 1
rsm_dir = os.path.join(OUT, "rsm")
mb = sum(os.path.getsize(os.path.join(rsm_dir, f))
         for f in os.listdir(rsm_dir)) / 1e6
print(f"rsm/              {n} matrices, {mb:.1f} MB")

# %% overlap.csv
# sq_overlap is computed from voxel-level patterns, which the 8x8 matrices do
# not contain, so unlike the ceilings it cannot be recomputed in the public
# repo and has to be exported.
fp = os.path.join(RSA_DIR, f"subspace_angle_{EVENT}_no_censoring.csv")
if os.path.exists(fp):
    ov = pd.read_csv(fp)
    # Keep ONLY the sixteen ROIs the paper reports. The source file also holds
    # striatal ROIs from an analysis that is not in the paper; mapping with
    # .fillna() used to let those through under their raw mask filenames, so
    # overlap.csv shipped 22 ROIs with two different naming conventions while
    # rsa_betas.csv held 16. Drop rather than rename: an unmapped ROI here means
    # the row is not part of this paper.
    ov["roi"] = ov["roi"].map(ROI_FILES)
    dropped_rois = int(ov.roi.isna().sum())
    ov = ov.dropna(subset=["roi"])
    keep = [c for c in OVERLAP_COLUMNS if c in ov.columns]
    ov[keep].to_csv(os.path.join(OUT, "overlap.csv"), index=False)
    print(f"overlap.csv       {len(ov)} rows, {ov.roi.nunique()} ROIs"
          + (f" (dropped {dropped_rois} unreported-ROI rows)"
             if dropped_rois else ""))
else:
    print(f"overlap.csv       [skip] not found: {fp}")

# %% tutorial.csv — trial-level learning-phase behavior (Fig 2b-c)
# The tutorial was never aggregated into a trialwise file, so it is built here
# from the raw frame files. process_trial() below is a port of the function in
# mrdecode_analysis.Rmd, verified to reproduce trialwise_mt_rt.csv exactly
# (max |difference| ~1e-13 on every switch trial).
SPEED_THRESH, SG_WIN, SG_ORD, DIST_A, DIST_B = 2.0, 7, 2, 1.0, 3.0
BAD_ABORTS = {"too_slow", "movement_during_hold_time"}
_wrap180 = lambda a: (a + 180.0) % 360.0 - 180.0


def process_trial(d, go_time):
    """(rt, mt, initial_heading_deg, endpoint_angle_deg, target_angle_deg).

    Rotates the trial so the target is at 0 deg, finds movement onset by a
    speed threshold, and takes the initial heading between 1 cm and 3 cm of
    cursor travel. Angles are in CURSOR space relative to the target, so
    perfect compensation is 0.
    """
    d = d.sort_values("trial_time").reset_index(drop=True)
    n = len(d)
    tgt = np.degrees(np.arctan2(d.target_y.iloc[0], d.target_x.iloc[0]))
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan, tgt
    th = np.radians(tgt)
    cx, cy = d.cursor_x.to_numpy(), d.cursor_y.to_numpy()
    cxr = np.cos(-th) * cx - np.sin(-th) * cy
    cyr = np.sin(-th) * cx + np.cos(-th) * cy
    hx, hy = (cx, cy) if n < SG_WIN else (savgol_filter(cx, SG_WIN, SG_ORD),
                                          savgol_filter(cy, SG_WIN, SG_ORD))
    t_ = d.trial_time.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.sqrt(np.concatenate([[np.nan], np.diff(hx)]) ** 2
                        + np.concatenate([[np.nan], np.diff(hy)]) ** 2) \
            / np.concatenate([[np.nan], np.diff(t_)])
    after = np.where(t_ >= go_time)[0]
    cand = after[np.nan_to_num(speed[after], nan=-1) >= SPEED_THRESH]
    comp = np.where(d["complete_for_metrics"].to_numpy())[0]
    if len(cand) == 0 or len(comp) == 0:
        return np.nan, np.nan, np.nan, np.nan, tgt
    on, off = cand[0], comp[0]
    x0, y0 = cxr[on], cyr[on]
    sl = slice(on, off + 1)
    r = np.sqrt((cxr[sl] - x0) ** 2 + (cyr[sl] - y0) ** 2)
    ia, ib = np.where(r >= DIST_A)[0], np.where(r >= DIST_B)[0]
    init = np.nan
    if len(ia) and len(ib):
        a, b = ia[0], ib[0]
        init = _wrap180(np.degrees(np.arctan2(cyr[sl][b] - cyr[sl][a],
                                              cxr[sl][b] - cxr[sl][a])))
    end = _wrap180(np.degrees(np.arctan2(cyr[off] - y0, cxr[off] - x0)))
    return t_[on] - go_time, t_[off] - t_[on], init, end, tgt


by_subj = {}
for f in glob.glob(os.path.join(LEARN_DIR, "*", "*_frame_events_*_learn_*.csv")):
    s = os.path.basename(f).split("_")[0]
    if s not in config.SUBJECTS:
        continue
    m = re.search(r"_frame_events_(?:\d+_)?(\d{14})_learn", os.path.basename(f))
    by_subj.setdefault(s, []).append((m.group(1) if m else "0", f))

# Methods: where the learning phase was repeated, use the latest attempt.
rows = []
for s in sorted(by_subj):
    fr = pd.read_csv(max(by_subj[s])[1])
    fr["complete_bool"] = fr["complete"].astype(str).str.lower().isin(["true", "1"])
    keys = ["subject", "phase", "run", "block", "trial"]
    bad = fr.groupby(keys)["abort"].apply(
        lambda x: x.astype(str).isin(BAD_ABORTS).any())
    ok = set(bad[~bad].index)
    for key, d in fr.groupby(keys, sort=False):
        if key not in ok:
            continue
        d = d.sort_values("trial_time").reset_index(drop=True)
        cb = d.complete_bool.to_numpy()
        mv = (d.trial_state == "move").to_numpy()
        fc = np.where(cb)[0][0] if cb.any() else None
        lm = np.where(mv)[0][-1] if mv.any() else None
        kc = np.zeros(len(d), bool)
        kf = np.zeros(len(d), bool)
        if fc is not None:
            kc[fc] = True
        elif lm is not None:
            kf[lm] = True
        keep = (mv & ~cb) | kc | kf
        if not keep.any():
            continue
        sub = d[keep].copy()
        sub["complete_for_metrics"] = (cb | kc | kf)[keep]
        rt, mt, init, end, tgt = process_trial(sub, sub.trial_time.iloc[0])
        rows.append(dict(subject=key[0], block=key[3], trial=key[4],
                         manipulation_type=d.manipulation_type.iloc[0],
                         which_target=int(d.which_target.iloc[0]),
                         target_angle_deg=tgt, initial_heading_deg=init,
                         endpoint_angle_deg=end, rt=rt, mt=mt))

if rows:
    tut = pd.DataFrame(rows).sort_values(["subject", "block", "trial"])
    tut.to_csv(os.path.join(OUT, "tutorial.csv"), index=False)
    print(f"tutorial.csv      {len(tut)} rows, {tut.subject.nunique()} subjects")
else:
    print(f"tutorial.csv      [skip] no learn files under {LEARN_DIR}")

# %%
print("\nCheck behavior.csv for anything identifying before committing.")
print("Then run each analysis script and compare against the paper.")
