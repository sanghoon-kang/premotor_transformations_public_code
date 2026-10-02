#!/usr/bin/env python3
"""
we are doing searchlight rsa over the whole brain with this script.
------------------------------------
logic is that at each searchlight sphere, we:
  1. Split trials into odd / even runs (see subject_rsa_full.py)
  2. Average beta patterns per trial type within each half
  3. Compute 8×8 Spearman correlation RSM between halves, then symmetrize
  4. Regress the flattened RSM (upper triangle + diagonal, include_diag=True)
     against model RSMs using OLS with fit_intercept=False
  5. Save one beta-weight NIfTI per model predictor × standardization

Outputs (saved to SEARCHLIGHT_RSA_OUT_DIR):
    searchlight_rsa_{model}_{absolute|zscored}_{subj}_{event}_{censor}.nii.gz

Usage:
    python searchlight_rsa.py <subj> <space_idx:0|1> <run_censor_thr>
                              <subj_runs_thr> <event>
                              <censor_label:censoring|no_censoring>

OR just use manual arguments. I am going to do that at least for the first few searchlights.

Author: Sanghoon
"""
# %% load modules and functions
import sys
import os
import time
import warnings

import numpy as np
import nibabel as nib
from nilearn import image
from nilearn.plotting import plot_event, plot_stat_map, plot_img, show
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata
from joblib import Parallel, delayed
from templateflow import api as tf
import matplotlib.pyplot as plt

from rsatoolbox.util.searchlight import get_volume_searchlight

# Custom utility (same import path as subject_rsa_full.py)
sys.path.insert(0, "/home/sk2949/project/github/mrdecode/analysis/fmri/glm_mvpa")
from utils.utils import load_data_for_analysis

if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# %% configuration

PROJECT_DIR          = "/home/sk2949/project/mrdecode/"
SEARCHLIGHT_RSA_OUT_DIR = os.path.join(PROJECT_DIR, "outputs/searchlight_rsa/")
SPACE_LIST           = ["T1w", "MNI152NLin2009cAsym"]
RADIUS_VOX           = 5       # searchlight radius in voxels
N_JOBS               = -1      # -1 = use all available cores
N_COND               = 8       # number of trial types

os.makedirs(SEARCHLIGHT_RSA_OUT_DIR, exist_ok=True)

# argument parsing

args = sys.argv
# args = [999, "mrd1013", 1, 70, 6, "all", "no_censoring"] # first element of sys.argv is irrelevant

if len(args) < 7:
    print(
        "pls check that you've entered all requisite arguments"
    )
    sys.exit(1)

subj_name      = args[1]
space          = SPACE_LIST[int(args[2])]
run_censor_thr = int(args[3])
subj_runs_thr  = int(args[4])
event          = args[5]
censor_label   = args[6]
assert censor_label in ["censoring", "no_censoring"]

print(f"""
Subject:        {subj_name}
Space:          {space}
Run censor thr: {run_censor_thr}
Run thr:        {subj_runs_thr}
Event:          {event}
Censor label:   {censor_label}
Radius (vox):   {RADIUS_VOX}
""")

# %% model rsms

task_rule = np.zeros((N_COND, N_COND))
task_rule[:4, :4] = 1   # Rot trials share the same rule
task_rule[4:, 4:] = 1   # Ref trials share the same rule

trial_type = np.eye(N_COND)

goal_location = np.eye(N_COND)
goal_location[0, 4] = goal_location[4, 0] = 1
goal_location[1, 5] = goal_location[5, 1] = 1
goal_location[2, 6] = goal_location[6, 2] = 1
goal_location[3, 7] = goal_location[7, 3] = 1

optimal_movement = np.eye(N_COND)
optimal_movement[0, 4] = optimal_movement[4, 0] = 1
optimal_movement[2, 6] = optimal_movement[6, 2] = 1

MODEL_NAMES = ["task_rule", "goal_location", "optimal_movement", "trial_type"]
MODEL_MATS  = [task_rule, goal_location, optimal_movement, trial_type]

# %% helper functions from rsa_subject_full.py

def vec_upper(rdm, include_diag=True):
    """Flatten upper triangle of a square matrix (optionally including diagonal)."""
    k = 0 if include_diag else 1
    i, j = np.triu_indices_from(rdm, k=k)
    return rdm[i, j].astype(float)


def zscore_1d(x, eps=1e-12):
    """z-score a 1 dimensional vector"""
    return (x - x.mean()) / (x.std(ddof=1) + eps)


def fit_linear_betas(X_dict, y_vec, standardize=False):
    """
    OLS regression (fit_intercept=False)
    Refer to subject_rsa_full.py
    y_vec will be empirical matrix most likely
    X_dict: {name: 1D array} aligned to y_vec, model matrices most likely
    Returns: {name: beta}
    """
    names = list(X_dict.keys())
    
    X = np.column_stack([X_dict[n] for n in names]).astype(float)
    
    y = y_vec.astype(float)
    
    if standardize:
        X = np.column_stack([zscore_1d(X[:, j]) for j in range(X.shape[1])])
        y = zscore_1d(y)
    
    lr = LinearRegression(fit_intercept=False) # no intercept in accordance with the rsa regression definition
    
    lr.fit(X, y)

    return dict(zip(names, lr.coef_.tolist())) # names and coefficients should align well


def spearman_corrmat(A, B):
    """
    Compute Spearman correlation between every pair of rows in A and B.

    Exploiting the fact that unit-normalized dot product of two vectors

    A : (n_conds, n_vox) — averaged beta patterns for half-1
    B : (n_conds, n_vox) — averaged beta patterns for half-2
    Returns: (n_conds, n_conds) matrix where [i, j] = spearmanr(A[i], B[j])

    Equivalent to the nested spearmanr loop in subject_rsa_full.py's
    calculate_trialtype_to_trialtype_corr, vectorized via ranking then
    Pearson on ranks.
    """
    # Rank each condition's pattern independently across voxels
    A_r = np.array([rankdata(row) for row in A], dtype=float)   # (n_conds, n_vox)
    B_r = np.array([rankdata(row) for row in B], dtype=float)

    # Mean-centre and unit-normalise (= Pearson on ranks = Spearman)
    A_c = A_r - A_r.mean(axis=1, keepdims=True)
    B_c = B_r - B_r.mean(axis=1, keepdims=True)
    A_n = A_c / (np.linalg.norm(A_c, axis=1, keepdims=True) + 1e-12) # make unit vector with L2 norm (now unit vector)
    B_n = B_c / (np.linalg.norm(B_c, axis=1, keepdims=True) + 1e-12)

    return A_n @ B_n.T   # (n_conds, n_conds) matrix multiplication. So pairwise dot products between two 2d vectors
                         # every element is a correlation between trial type and trial type.
                         # this means it might be asymmetrical. Symmetry will have to be forced later


# %% load beta maps
print("Loading beta maps …")
(subject_betas, subject_labels, subject_runs,
 subject_critical_labels, subject_trial_types,
 whole_brain_mask, gm_mask) = load_data_for_analysis(
    subj_name=subj_name,
    space=space,
    run_censored_volume_threshold=run_censor_thr,
    subj_remaining_runs_threshold=subj_runs_thr,
    normalization="run_z",
    event_of_interest=event,
    filter_trials="none",
    censor_label=censor_label
)

trial_types_arr = np.array(subject_trial_types)
runs_arr        = np.array(subject_runs)
unique_tts      = sorted(np.unique(trial_types_arr))

# %% load modules and functions)

if len(unique_tts) != N_COND:
    print(f"WARNING: expected {N_COND} trial types, found {len(unique_tts)}: {unique_tts}")

# Concatenate betas → 4D array
beta_imgs_4d = image.concat_imgs(subject_betas)
data_4d      = beta_imgs_4d.get_fdata()          # (X, Y, Z, N)
N_trials     = data_4d.shape[3]

# %% flatten beta (n_trials, v_mask)

mask_vol  = whole_brain_mask.get_fdata().astype(bool)
mask_flat = mask_vol.ravel(order="C") # elements read row-by-row (default)
idx       = np.flatnonzero(mask_flat) # flat-volume indices of in-mask voxels

X_flat   = data_4d.reshape(-1, N_trials).T  # (N_trials, V_total). -1 is 'wild card' and will fit to whatever necessary to have sec dimension be trial n
                                            # and then we transpose matrix
X_masked = X_flat[:, idx]                   # (N_trials, V_mask) - with the idx we derived above

print(f"Brain voxels in mask: {idx.size}  |  Trials loaded: {N_trials}")

# %% split-half (odd/even)

unique_runs = sorted(set(subject_runs))
odd_runs    = [r for r in unique_runs if r % 2 == 1]
even_runs   = [r for r in unique_runs if r % 2 == 0]

odd_mask  = np.isin(runs_arr, odd_runs)
even_mask = np.isin(runs_arr, even_runs)

X_odd  = X_masked[odd_mask]        # (N_odd,  V_mask)
X_even = X_masked[even_mask]       # (N_even, V_mask)
y_odd  = trial_types_arr[odd_mask]
y_even = trial_types_arr[even_mask]

print(f"Odd-run trials: {odd_mask.sum()}  |  Even-run trials: {even_mask.sum()}")

# Pre-compute whole-brain averaged patterns per trial type for each half.
# These are the inputs to the per-sphere RSA.
# Shape: (N_COND, V_mask)
avg_odd_full  = np.array([X_odd [y_odd  == tt].mean(axis=0) for tt in unique_tts],
                          dtype=np.float32)
avg_even_full = np.array([X_even[y_even == tt].mean(axis=0) for tt in unique_tts],
                          dtype=np.float32)

print("Pre-averaged pattern shapes:", avg_odd_full.shape, avg_even_full.shape)

# %% building the model matrix dictionary to be put into the regression function

X_model_dict = {name: vec_upper(M, include_diag=True)
                for name, M in zip(MODEL_NAMES, MODEL_MATS)}
n_pairs = len(next(iter(X_model_dict.values())))
print(f"RSM vector length (upper tri + diag): {n_pairs}")

# %% build searchlight neighborhoods

print("Building searchlight neighborhoods …")
centers_mask, neighbors_mask = volume_searchlighget_t(
    mask=mask_vol,
    radius=RADIUS_VOX
)

# Map flat-volume indices → channel indices within the mask
map_flat_to_chan = -np.ones(mask_flat.shape[0], dtype=int)
map_flat_to_chan[idx] = np.arange(idx.size)

centers   = []
neighbors = []
for c_flat, neigh_flat in zip(centers_mask, neighbors_mask):
    c_chan = map_flat_to_chan[c_flat]
    n_chan = np.array(
        [map_flat_to_chan[j] for j in neigh_flat if map_flat_to_chan[j] != -1],
        dtype=int
    )
    # Keep centers that lie within the mask and have enough voxels for ranking
    if c_chan != -1 and len(n_chan) >= N_COND:
        centers.append(c_chan)
        neighbors.append(n_chan)

centers = np.array(centers, dtype=int)
print(f"Valid searchlight centers: {len(centers)}")

# %% per-center rsa function

def rsa_at_center(neigh_idx):
    """
    Compute RSA regression betas for one searchlight sphere.

    Steps (mirrors subject_rsa_full.py):
      1. Extract sphere-voxel patterns for odd/even halves
      2. Compute Spearman RSM (8×8)
      3. Symmetrize: RSM = (RSM + RSM.T) / 2
      4. Flatten upper triangle including diagonal (36 values for 8 conds)
      5. OLS regression against model RSMs — absolute and z-scored

    Returns: (betas_abs, betas_z) each shape (n_models,), or NaN arrays on failure.
    """
    n_models = len(MODEL_NAMES)
    nan_out  = (np.full(n_models, np.nan, dtype=np.float32),
                np.full(n_models, np.nan, dtype=np.float32))

    A = avg_odd_full [:, neigh_idx]   # (N_COND, V_sphere)
    B = avg_even_full[:, neigh_idx]   # (N_COND, V_sphere)

    if A.shape[1] == 0:
        return nan_out

    # Spearman RSM + symmetrize
    rsm = spearman_corrmat(A, B)
    rsm = (rsm + rsm.T) / 2          # symmetric, matches subject_rsa_full.py

    # Flatten (include diagonal = within-condition reliability)
    y_vec = vec_upper(rsm, include_diag=True)   # (36,)

    if np.any(np.isnan(y_vec)) or np.nanstd(y_vec) < 1e-12:
        return nan_out

    # OLS regression — absolute (as-is) and z-scored
    betas_abs_d = fit_linear_betas(X_model_dict, y_vec, standardize=False)
    betas_z_d   = fit_linear_betas(X_model_dict, y_vec, standardize=True)

    betas_abs = np.array([betas_abs_d[n] for n in MODEL_NAMES], dtype=np.float32)
    betas_z   = np.array([betas_z_d  [n] for n in MODEL_NAMES], dtype=np.float32)

    return betas_abs, betas_z


# %% running the searchlight (parallelize over cpus)

print(f"Running searchlight RSA across {len(centers)} centers "
      f"(n_jobs={N_JOBS}) …")
start = time.time()

results = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(rsa_at_center)(neigh_idx) for neigh_idx in neighbors
)

elapsed = time.time() - start
print(f"Searchlight complete in {elapsed:.1f} s")

# Unpack
betas_abs_all = np.array([r[0] for r in results])   # (n_centers, n_models)
betas_z_all   = np.array([r[1] for r in results])   # (n_centers, n_models)

# %% save the data
affine = whole_brain_mask.affine

for m_idx, m_name in enumerate(MODEL_NAMES):
    for betas_arr, suffix in [(betas_abs_all, "absolute"),
                               (betas_z_all,   "zscored")]:

        scores_flat          = np.full(mask_flat.shape, np.nan, dtype=np.float32)
        scores_flat[idx[centers]] = betas_arr[:, m_idx]
        scores_vol           = scores_flat.reshape(mask_vol.shape)

        out_img  = nib.Nifti1Image(scores_vol, affine)
        out_path = os.path.join(
            SEARCHLIGHT_RSA_OUT_DIR,
            f"searchlight_rsa_{m_name}_{suffix}_{subj_name}_{event}_{censor_label}.nii.gz"
        )
        out_img.to_filename(out_path)
        print(f"Saved → {out_path}")

print("All outputs written. Done.")
# %%
