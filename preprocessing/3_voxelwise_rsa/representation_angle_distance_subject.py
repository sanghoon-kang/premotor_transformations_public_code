#!/usr/bin/env python3
"""
subspace_angle_subject.py

Angle between the transformation-context (rule) and optimal-movement pattern
vectors, per subject per ROI. One sbatch job covers all subjects and ROIs:
loading dominates (~6 s/subject), the vector algebra is trivial.

WHAT THIS MEASURES, AND WHAT IT DOESN'T
---------------------------------------
This asks whether the rule difference and the movement difference lie along
overlapping directions in an ROI's pattern space. It is NOT a test that two
subspaces are orthogonal: there is one vector per factor, so there is one
angle. Framing it as subspace orthogonality invites a geometric reading
(Bernardi et al. 2020) that one contrast per factor cannot support.

WHY CROSS-VALIDATION IS NOT OPTIONAL
------------------------------------
In a p-dimensional ROI two independent noise vectors are nearly orthogonal:
E[cos] = 0 with SD ~ 1/sqrt(p). So a naive angle is biased toward 90 degrees,
and MORE so the noisier the ROI — an ROI with no rule signal would look
maximally "orthogonal". That inverts the interpretation you want.

Each vector is therefore built separately in odd and even runs and combined
across folds:

             ------------------------------------------
             sqrt( (a_odd . a_even) * (b_odd . b_even) )

Noise is independent between folds, so it contributes zero in expectation to
every term. The denominator is the cross-validated squared norm — an unbiased
estimate that can go negative when a vector is not reliably present, which is
exactly the case where the angle is meaningless. Those ROIs are flagged rather
than silently reported.

VECTOR CONSTRUCTION
-------------------
Conditions are ordered rot1..rot4, ref1..ref4. Two vectors, from orthogonal
weightings of the same eight patterns:

    v_rule = mean(rot) - mean(ref)
    v_move = the movement information that TARGET cannot explain

theta_i is condition i's optimal movement direction, from the task code
(targets at +/-20 deg from vertical; rotation +40 deg; mirror about vertical):

           T1     T2     T3     T4
    rot   110    150    290    330
    ref   110     70    290    250

v_move is obtained by projecting the movement-tuning plane (cos/sin of theta)
out of the rule axis and the target-tuning plane (cos/sin of the TARGET
direction). Exactly one dimension survives -- the target and movement planes
have principal angles of 0 and 90 deg, sharing one dimension exactly -- and it
comes out as

    rot1 rot2 rot3 rot4   ref1 ref2 ref3 ref4
      0  +.5    0  -.5      0  -.5    0  +.5

zero on the matched targets, +/-0.5 on the unmatched ones. On T1 and T3 both
rules require the same movement, so movement there is fully determined by
target and nothing is left; the only target-independent movement signal in this
design is the rule x target interaction on T2 and T4.

Everything is orthogonal by construction: v_move to v_rule, and v_move to both
target directions. Asserted at import.

Usage (same six args as the other subject-level scripts):
    python subspace_angle_subject.py <space_idx> <run_censor_thr>
                                     <subj_runs_thr> <event> <censor_label>

Output:
    {cfg.rsa_dir}/subspace_angle_{event}_{censor}.csv
"""

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

# Split-half reliability of sq_overlap ITSELF. The odd/even split gives one
# overlap value per subject, so it cannot be split again — instead the runs are
# divided into FOUR groups, sq_overlap computed twice from disjoint pairs, and
# the two values correlated across subjects at the group level.
#
# With 9 runs the quarters are unbalanced (3/2/2/2), and any single partition
# is arbitrary, so this is repeated over N_SPLIT_REPEATS random partitions and
# the two estimates averaged. Set N_SPLIT_REPEATS = 0 to skip.
N_SPLIT_REPEATS = 20
SPLIT_SEED = 0

# Optimal movement direction per condition, derived from the task code:
# drawer.py places targets at +/-target_angle from vertical (target_angle=20,
# radius=10), giving target directions 70, 110, 250, 290 deg. Rotation is
# +2*target_angle = +40 deg; mirror is reflection about the vertical axis.
# Matched targets (1 and 3, where both rules require the same movement) fall
# out of this rather than being assumed.
TARGET_ANGLE = 20
TARGET_DIR = [(90 - TARGET_ANGLE), (90 + TARGET_ANGLE),
              (270 - TARGET_ANGLE), (270 + TARGET_ANGLE)]   # T1..T4
MOVE_DIR = np.array(
    [(d + 2 * TARGET_ANGLE) % 360 for d in TARGET_DIR] +     # rot T1..T4
    [(180 - d) % 360 for d in TARGET_DIR]                    # ref T1..T4
)

# Weights over the 8 conditions.
#
#   W_RULE   +1 on rot, -1 on ref: the transformation-context axis.
#
#   W_MOVE   movement direction WITH TARGET PARTIALLED OUT.
#            Decomposing the cos/sin movement-tuning weights into a target main
#            effect plus a rule x target interaction shows that both have the
#            SAME interaction direction (their normalised interaction parts
#            correlate at exactly 1.0). So once target information is removed,
#            the 2D movement plane collapses to a single dimension:
#
#                (rot4 + ref2) - (rot2 + ref4)
#
#            supported only on targets 2 and 4 -- the unmatched ones, the only
#            place the two rules require different movements. Which is the
#            right answer: with target held constant, the only movement
#            information left IS the rule-dependent part.
#
# Orthogonality, all exact and asserted below:
#   W_MOVE . W_RULE        = 0   (rule balanced across the two sides)
#   W_MOVE . any target    = 0   (target balanced across the two sides)
#   W_MOVE . grand mean    = 0
# So the contrast cannot express rule or target, and neither can express it.
W_RULE = np.r_[np.ones(4), -np.ones(4)] / 4.0
W_MOVE = np.array([0, -1, 0, 1, 0, 1, 0, -1], dtype=float)
W_MOVE /= np.linalg.norm(W_MOVE)

_target = np.r_[0, 1, 2, 3, 0, 1, 2, 3]
assert abs(W_RULE @ W_MOVE) < 1e-12, "W_MOVE not orthogonal to the rule axis"
assert abs(W_MOVE.sum()) < 1e-12, "W_MOVE does not sum to zero"
for _t in range(4):
    _c = (_target == _t).astype(float); _c -= _c.mean()
    assert abs(W_MOVE @ _c) < 1e-12, f"W_MOVE not orthogonal to target {_t+1}"

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
    """
    P: (8, n_vox) fold-average patterns, rows rot1..rot4 then ref1..ref4.
    Returns (v_rule, v_move) — two fixed linear combinations of the same eight
    patterns, with orthogonal weights, so neither is built out of the other.
    """
    return W_RULE @ P, W_MOVE @ P


def cross_fold_overlap(a_odd, a_even, b_odd, b_even):
    """
    Overlap between two pattern vectors, cross-validated across run folds.

    The two cross-fold cosines use DISJOINT halves of the data:

        cos1 = (a_odd  . b_even) / denom
        cos2 = (a_even . b_odd ) / denom

    so they are independent estimates of the same underlying cosine mu.
    Their PRODUCT is therefore an unbiased estimate of mu^2:

        E[cos1 * cos2] = E[cos1] E[cos2] = mu^2

    That is the whole measure. Squaring makes it sign-free, which matters
    because the sign of a cosine between two difference-vectors depends both on
    which way each contrast was written and on which way a given subject's rule
    axis happens to lean — and there is no reason subjects should agree on that.
    Multiplying independent estimates is what keeps it unbiased: a naive |cos|
    or cos^2 from one estimate would be inflated by noise.

    denom is the geometric mean of each vector's two fold norms, so the result
    does not depend on which half is labelled "odd".

    Also returns each vector's cross-validated squared magnitude, a_odd . a_even.
    Like the overlap it is a product of two independent estimates, so it is
    unbiased and can go negative when a vector is not reliably present. These
    are the control measures: if caching tracks how STRONG a vector is rather
    than how much the two overlap, that shows up here.

    Returns (sq_overlap, cos_mean, cos1, cos2, mag_rule, mag_move).
    """
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
    """
    Does this pattern vector replicate across the odd/even split?

    method="cosine" (default) is the UNCENTRED cosine between the two fold
    vectors. This is the right one, for a reason beyond consistency: it is
    exactly the factor by which the overlap is attenuated, since

        cos_attn = cos_cv * sqrt(rel_rule * rel_move)

    holds identically when rel_* are these cosines. A Pearson or Spearman
    correlation centres each vector first, so it describes the reliability of
    the CENTRED pattern — a different vector from the one the overlap uses,
    differing by any voxel-general component.

    method="spearman" gives the rank correlation, matching
    calculate_trialtype_to_trialtype_corr in the RSA pipeline. Reported
    alongside as a familiar reference, not used for attenuation.
    """
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
    """
    Two independent sq_overlap estimates from disjoint halves of the runs.

    Runs are shuffled into four groups; groups 1+2 give one estimate and
    groups 3+4 the other, so the two use completely separate data. Averaged
    over N_SPLIT_REPEATS partitions because any single one is arbitrary.
    """
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
# ROI masks are cached across subjects: every subject is in the same MNI grid,
# so get_roi_mask() returns the same thing each time. Without this it runs
# n_subjects x n_rois times instead of n_rois times, which dominates the job.
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

    # Pull only the voxels any ROI needs, rather than the whole brain: the
    # full array is n_trials x ~315k voxels in float64, which is the other
    # thing making this slow.
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

        # two independent sq_overlap estimates, for the group-level
        # split-half reliability of the measure itself
        sq_half_a, sq_half_b = split_half_overlaps(
            flat, tt_arr, runs, uniq, local, _split_rng)

        rows.append(dict(
            subject=subj, roi=roi, family=roi_family(roi),
            n_vox=roi_idx.size,
            # PRIMARY: unbiased squared overlap, sign-free
            sq_overlap=sq_overlap,
            # the two independent estimates it is the product of, and their
            # mean — the signed overlap, kept as a secondary question
            cos1=cos1, cos2=cos2, cos_mean=cos_mean,
            # control measures: vector strength, separate from their overlap
            mag_rule=mag_rule, mag_move=mag_move,
            # two half-data estimates of sq_overlap (disjoint runs)
            sq_half_a=sq_half_a, sq_half_b=sq_half_b,
            # cosine reliability = the attenuation factor (see reliability())
            rel_rule=reliability(r_odd, r_even),
            rel_move=reliability(m_odd, m_even),
            # rank-correlation version, for reference
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