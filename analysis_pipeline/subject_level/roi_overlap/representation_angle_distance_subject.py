#!/usr/bin/env python3
"""Per-subject, per-ROI overlap between the rule vector and the target-controlled
movement vector, cross-validated across odd/even runs (Figs 5c-d)."""

# %% imports
import os
import sys
import time

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import rankdata

sys.path.insert(0, "/home/sk2949/project/github/mrdecode/analysis/fmri/glm_mvpa")
from utils.utils import load_data_for_analysis, get_roi_mask
from utils.pipeline_config import (cfg, SUBJECT_LIST, ROI_SETS, roi_strings,
                             roi_family, cli_arg)

# %% config
ROI_STAGE_SETS = ["main", "supp_cortical", "striatum"]
ROI_LIST = roi_strings(ROI_STAGE_SETS)

N_COND = cfg.n_cond

# random 4-way run splits for the split-half reliability of sq_overlap (0 = skip)
N_SPLIT_REPEATS = 20
SPLIT_SEED = 0

# optimal movement direction per condition (deg), from the task code
TARGET_ANGLE = 20
TARGET_DIR = [(90 - TARGET_ANGLE), (90 + TARGET_ANGLE),
              (270 - TARGET_ANGLE), (270 + TARGET_ANGLE)]   # T1..T4
MOVE_DIR = np.array(
    [(d + 2 * TARGET_ANGLE) % 360 for d in TARGET_DIR] +     # rot T1..T4
    [(180 - d) % 360 for d in TARGET_DIR]                    # ref T1..T4
)

# condition weights over rot1-4, ref1-4; W_MOVE is movement with target partialled out (T2/T4 only)
W_RULE = np.r_[np.ones(4), -np.ones(4)] / 4.0
W_MOVE = np.array([0, -1, 0, 1, 0, 1, 0, -1], dtype=float)
W_MOVE /= np.linalg.norm(W_MOVE)

# W_MOVE must be orthogonal to rule and to every target
_target = np.r_[0, 1, 2, 3, 0, 1, 2, 3]
assert abs(W_RULE @ W_MOVE) < 1e-12, "W_MOVE not orthogonal to the rule axis"
assert abs(W_MOVE.sum()) < 1e-12, "W_MOVE does not sum to zero"
for _t in range(4):
    _c = (_target == _t).astype(float); _c -= _c.mean()
    assert abs(W_MOVE @ _c) < 1e-12, f"W_MOVE not orthogonal to target {_t+1}"

# args: space_idx run_censor_thr subj_runs_thr event censor_label
SPACE_LIST     = ["T1w", "MNI152NLin2009cAsym"]
space          = SPACE_LIST[cli_arg(1, cfg.space_idx, int)]
run_censor_thr = cli_arg(2, cfg.run_censor_thr, int)
subj_runs_thr  = cli_arg(3, cfg.subj_runs_thr, int)
event          = cli_arg(4, cfg.event)
censor_label   = cli_arg(5, cfg.censor_label)

print(f"{len(SUBJECT_LIST)} subjects x {len(ROI_LIST)} ROIs | {space} | {event}")


# %% helper functions
def fold_patterns(data_4d_flat, trial_types, run_mask, roi_idx, n_cond):
    """(n_cond, n_vox) average pattern per trial type within one fold."""
    out = np.empty((n_cond, roi_idx.size), dtype=np.float64)
    for k, tt in enumerate(range(1, n_cond + 1)):
        sel = run_mask & (trial_types == tt)
        if not sel.any():
            return None
        out[k] = data_4d_flat[sel][:, roi_idx].mean(axis=0)
    return out


def build_vectors(P):
    """Rule and movement vectors from (8, n_vox) fold patterns."""
    return W_RULE @ P, W_MOVE @ P


def cross_fold_overlap(a_odd, a_even, b_odd, b_even):
    """sq_overlap = cos1 * cos2 from disjoint folds (unbiased for cos^2), plus signed cos and magnitudes."""
    denom = np.sqrt((np.linalg.norm(a_odd) * np.linalg.norm(a_even)) *
                    (np.linalg.norm(b_odd) * np.linalg.norm(b_even)))
    if denom <= 0:
        return (np.nan,) * 6

    cos1 = (a_odd @ b_even) / denom
    cos2 = (a_even @ b_odd) / denom
    mag_a = float(a_odd @ a_even)          # cross-validated squared magnitude
    mag_b = float(b_odd @ b_even)
    return cos1 * cos2, 0.5 * (cos1 + cos2), cos1, cos2, mag_a, mag_b


def reliability(x_odd, x_even, method="cosine"):
    """Odd/even replication of a vector; default is the uncentered cosine."""
    a, b = np.asarray(x_odd, float), np.asarray(x_even, float)
    if method in ("spearman", "pearson"):
        if method == "spearman":
            a, b = rankdata(a), rankdata(b)
        a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan


def overlap_from_runs(flat, tt_arr, mask_a, mask_b, roi_idx):
    """sq_overlap computed from two given run subsets. None if a cell is empty."""
    Pa = fold_patterns(flat, tt_arr, mask_a, roi_idx, N_COND)
    Pb = fold_patterns(flat, tt_arr, mask_b, roi_idx, N_COND)
    if Pa is None or Pb is None:
        return np.nan
    ra, ma = build_vectors(Pa)
    rb, mb = build_vectors(Pb)
    return cross_fold_overlap(ra, rb, ma, mb)[0]


def split_half_overlaps(flat, tt_arr, runs, uniq, roi_idx, rng):
    """Two sq_overlap estimates from disjoint run halves, averaged over random splits."""
    if N_SPLIT_REPEATS < 1:
        return np.nan, np.nan
    a_vals, b_vals = [], []
    for _ in range(N_SPLIT_REPEATS):
        r = list(uniq)
        rng.shuffle(r)
        q = np.array_split(np.array(r), 4)
        a = overlap_from_runs(flat, tt_arr, np.isin(runs, q[0]),
                              np.isin(runs, q[1]), roi_idx)
        b = overlap_from_runs(flat, tt_arr, np.isin(runs, q[2]),
                              np.isin(runs, q[3]), roi_idx)
        if np.isfinite(a) and np.isfinite(b):
            a_vals.append(a); b_vals.append(b)
    if not a_vals:
        return np.nan, np.nan
    return float(np.mean(a_vals)), float(np.mean(b_vals))


# %% run
# ROI masks cached across subjects (same MNI grid)
_MASK_CACHE = {}

def roi_indices(roi, affine, shape):
    key = (roi, shape, affine.tobytes())
    if key not in _MASK_CACHE:
        m = get_roi_mask(roi=roi, atlas=cfg.atlas, affine=affine,
                         shape=shape, save_dir=cfg.rsa_dir)
        _MASK_CACHE[key] = np.flatnonzero(np.asarray(m).ravel())
    return _MASK_CACHE[key]


rows = []
t_start = time.time()
_split_rng = np.random.default_rng(SPLIT_SEED)

for subj in SUBJECT_LIST:
    t0 = time.time()
    (subject_betas, _, subject_runs, _, subject_trial_types,
     whole_brain_mask, _) = load_data_for_analysis(
        subj_name=subj, space=space,
        run_censored_volume_threshold=run_censor_thr,
        subj_remaining_runs_threshold=subj_runs_thr,
        normalization=cfg.normalization, event_of_interest=event,
        filter_trials="none", censor_label=censor_label,
    )

    affine, shape = subject_betas[0].affine, subject_betas[0].shape
    tt_arr = np.array(subject_trial_types)
    runs = np.array(subject_runs)
    uniq = sorted(set(subject_runs))
    odd = np.isin(runs, [r for r in uniq if r % 2 == 1])
    even = np.isin(runs, [r for r in uniq if r % 2 == 0])

    # load only the voxels some ROI needs
    all_idx = np.unique(np.concatenate(
        [roi_indices(r, affine, shape) for r in ROI_LIST]))
    pos = {v: k for k, v in enumerate(all_idx)}

    flat = np.empty((len(tt_arr), all_idx.size), dtype=np.float32)
    for i, img in enumerate(subject_betas):
        flat[i] = img.get_fdata().ravel()[all_idx]

    for roi in ROI_LIST:
        roi_idx = roi_indices(roi, affine, shape)
        if roi_idx.size < N_COND:
            print(f"  [SKIP] {roi}: {roi_idx.size} voxels")
            continue

        local = np.fromiter((pos[v] for v in roi_idx), dtype=int, count=roi_idx.size)
        P_odd = fold_patterns(flat, tt_arr, odd, local, N_COND)
        P_even = fold_patterns(flat, tt_arr, even, local, N_COND)
        if P_odd is None or P_even is None:
            print(f"  [SKIP] {roi}: a trial type is empty in one fold")
            continue

        r_odd, m_odd = build_vectors(P_odd)
        r_even, m_even = build_vectors(P_even)

        sq_overlap, cos_mean, cos1, cos2, mag_rule, mag_move = \
            cross_fold_overlap(r_odd, r_even, m_odd, m_even)

        sq_half_a, sq_half_b = split_half_overlaps(
            flat, tt_arr, runs, uniq, local, _split_rng)

        rows.append(dict(
            subject=subj, roi=roi, family=roi_family(roi),
            n_vox=roi_idx.size,
            sq_overlap=sq_overlap,
            cos1=cos1, cos2=cos2, cos_mean=cos_mean,
            mag_rule=mag_rule, mag_move=mag_move,
            sq_half_a=sq_half_a, sq_half_b=sq_half_b,
            rel_rule=reliability(r_odd, r_even),
            rel_move=reliability(m_odd, m_even),
            rel_rule_spearman=reliability(r_odd, r_even, "spearman"),
            rel_move_spearman=reliability(m_odd, m_even, "spearman"),
        ))

    done = SUBJECT_LIST.index(subj) + 1
    el = time.time() - t_start
    print(f"{subj}: {time.time() - t0:.1f} s  "
          f"[{done}/{len(SUBJECT_LIST)}, ~{el / done * (len(SUBJECT_LIST) - done):.0f} s left]")

df = pd.DataFrame(rows)
out = os.path.join(cfg.rsa_dir, f"subspace_angle_{event}_{censor_label}.csv")
df.to_csv(out, index=False)
print(f"\nSaved {out}  ({len(df)} subject x ROI rows, "
      f"{time.time() - t_start:.0f} s total)")
print(f"sq_overlap defined: {int(df.sq_overlap.notna().sum())} / {len(df)}")
# %%