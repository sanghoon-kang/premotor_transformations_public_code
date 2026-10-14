#!/usr/bin/env python3
"""Whole-brain searchlight RSA for one subject (Figs 6 and S5).
Usage: python searchlight_rsa.py <subj> <space_idx> <run_censor_thr> <subj_runs_thr> <event> <censor_label>"""
# %% load modules and functions
import sys
import os
import time
import warnings

import numpy as np
import nibabel as nib
from nilearn import image
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata
from joblib import Parallel, delayed

from rsatoolbox.util.searchlight import get_volume_searchlight

sys.path.insert(0, "/home/sk2949/project/github/mrdecode/analysis/fmri/glm_mvpa")
from utils.utils import load_data_for_analysis

if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# %% configuration

PROJECT_DIR          = "/home/sk2949/project/mrdecode/"
SEARCHLIGHT_RSA_OUT_DIR = os.path.join(PROJECT_DIR, "outputs/searchlight_rsa_intercept/")
SPACE_LIST           = ["T1w", "MNI152NLin2009cAsym"]
RADIUS_VOX           = 5
N_JOBS               = -1
N_COND               = 8

os.makedirs(SEARCHLIGHT_RSA_OUT_DIR, exist_ok=True)

args = sys.argv

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
task_rule[:4, :4] = 1
task_rule[4:, 4:] = 1

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

# %% helpers (same as the ROI analysis)

def vec_upper(rdm, include_diag=True):
    """Flatten the upper triangle, optionally with the diagonal."""
    k = 0 if include_diag else 1
    i, j = np.triu_indices_from(rdm, k=k)
    return rdm[i, j].astype(float)


def zscore_1d(x, eps=1e-12):
    """z-score a 1D vector."""
    return (x - x.mean()) / (x.std(ddof=1) + eps)


def fit_linear_betas(X_dict, y_vec, standardize=False):
    """OLS of the data RSM on the model RSMs; returns {name: beta}."""
    names = list(X_dict.keys())
    
    X = np.column_stack([X_dict[n] for n in names]).astype(float)
    
    y = y_vec.astype(float)
    
    if standardize:
        X = np.column_stack([zscore_1d(X[:, j]) for j in range(X.shape[1])])
        y = zscore_1d(y)
    
    lr = LinearRegression(fit_intercept=True)  # includes an intercept, as in the ROI analysis
    
    lr.fit(X, y)

    return dict(zip(names, lr.coef_.tolist()))


def spearman_corrmat(A, B):
    """Spearman correlation between every row of A and every row of B (Pearson on ranks)."""
    A_r = np.array([rankdata(row) for row in A], dtype=float)
    B_r = np.array([rankdata(row) for row in B], dtype=float)

    A_c = A_r - A_r.mean(axis=1, keepdims=True)
    B_c = B_r - B_r.mean(axis=1, keepdims=True)
    A_n = A_c / (np.linalg.norm(A_c, axis=1, keepdims=True) + 1e-12)
    B_n = B_c / (np.linalg.norm(B_c, axis=1, keepdims=True) + 1e-12)

    return A_n @ B_n.T   # not symmetric yet; symmetrized in rsa_at_center


# %% load beta maps
print("Loading beta maps ...")
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

# %% stack betas

if len(unique_tts) != N_COND:
    print(f"WARNING: expected {N_COND} trial types, found {len(unique_tts)}: {unique_tts}")

beta_imgs_4d = image.concat_imgs(subject_betas)
data_4d      = beta_imgs_4d.get_fdata()
N_trials     = data_4d.shape[3]

# %% flatten beta (n_trials, v_mask)

mask_vol  = whole_brain_mask.get_fdata().astype(bool)
mask_flat = mask_vol.ravel(order="C")
idx       = np.flatnonzero(mask_flat)

X_flat   = data_4d.reshape(-1, N_trials).T
X_masked = X_flat[:, idx]

print(f"Brain voxels in mask: {idx.size}  |  Trials loaded: {N_trials}")

# %% split-half (odd/even)

unique_runs = sorted(set(subject_runs))
odd_runs    = [r for r in unique_runs if r % 2 == 1]
even_runs   = [r for r in unique_runs if r % 2 == 0]

odd_mask  = np.isin(runs_arr, odd_runs)
even_mask = np.isin(runs_arr, even_runs)

X_odd  = X_masked[odd_mask]
X_even = X_masked[even_mask]
y_odd  = trial_types_arr[odd_mask]
y_even = trial_types_arr[even_mask]

print(f"Odd-run trials: {odd_mask.sum()}  |  Even-run trials: {even_mask.sum()}")

# average pattern per trial type in each half, (N_COND, V_mask)
avg_odd_full  = np.array([X_odd [y_odd  == tt].mean(axis=0) for tt in unique_tts],
                          dtype=np.float32)
avg_even_full = np.array([X_even[y_even == tt].mean(axis=0) for tt in unique_tts],
                          dtype=np.float32)

print("Pre-averaged pattern shapes:", avg_odd_full.shape, avg_even_full.shape)

# %% model RSM vectors

X_model_dict = {name: vec_upper(M, include_diag=True)
                for name, M in zip(MODEL_NAMES, MODEL_MATS)}
n_pairs = len(next(iter(X_model_dict.values())))
print(f"RSM vector length (upper tri + diag): {n_pairs}")

# %% build searchlight neighborhoods

print("Building searchlight neighborhoods ...")
centers_mask, neighbors_mask = get_volume_searchlight(
    mask=mask_vol,
    radius=RADIUS_VOX,
    threshold=0.5,   # keep centers with >= half the sphere in the brain (default 1.0 drops the outer ~12 mm of brain)
)

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
    if c_chan != -1 and len(n_chan) >= N_COND:
        centers.append(c_chan)
        neighbors.append(n_chan)

centers = np.array(centers, dtype=int)
print(f"Valid searchlight centers: {len(centers)}")

# %% per-center rsa function

def rsa_at_center(neigh_idx):
    """Model betas (absolute, z-scored) for one sphere, or NaNs if it fails."""
    n_models = len(MODEL_NAMES)
    nan_out  = (np.full(n_models, np.nan, dtype=np.float32),
                np.full(n_models, np.nan, dtype=np.float32))

    A = avg_odd_full [:, neigh_idx]
    B = avg_even_full[:, neigh_idx]

    if A.shape[1] == 0:
        return nan_out

    rsm = spearman_corrmat(A, B)
    rsm = (rsm + rsm.T) / 2
    y_vec = vec_upper(rsm, include_diag=True)

    if np.any(np.isnan(y_vec)) or np.nanstd(y_vec) < 1e-12:
        return nan_out

    betas_abs_d = fit_linear_betas(X_model_dict, y_vec, standardize=False)
    betas_z_d   = fit_linear_betas(X_model_dict, y_vec, standardize=True)

    betas_abs = np.array([betas_abs_d[n] for n in MODEL_NAMES], dtype=np.float32)
    betas_z   = np.array([betas_z_d  [n] for n in MODEL_NAMES], dtype=np.float32)

    return betas_abs, betas_z


# %% run the searchlight

print(f"Running searchlight RSA across {len(centers)} centers "
      f"(n_jobs={N_JOBS}) ...")
start = time.time()

results = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(rsa_at_center)(neigh_idx) for neigh_idx in neighbors
)

elapsed = time.time() - start
print(f"Searchlight complete in {elapsed:.1f} s")

betas_abs_all = np.array([r[0] for r in results])
betas_z_all   = np.array([r[1] for r in results])

# %% save one map per model and standardization
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
        print(f"Saved -> {out_path}")

print("All outputs written. Done.")
# %%
