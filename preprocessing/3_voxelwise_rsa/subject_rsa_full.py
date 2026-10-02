# %% Import modules
import warnings
import sys
import os
import pickle
import numpy as np
import nibabel as nib
from nilearn import image, plotting
from nilearn.plotting import plot_stat_map, plot_roi
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from templateflow import api as tf
from scipy.spatial.distance import mahalanobis
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm
from itertools import combinations

sys.path.insert(0, "/home/sk2949/project/github/mrdecode/analysis/fmri/glm_mvpa")

from utils.utils import load_data_for_analysis, get_roi_mask

# %% define paths and crucial variables

# Set paths and params
project_dir = "/home/sk2949/project/mrdecode/"
mri_dir = os.path.join(project_dir, "fmriprep_aroma_volume/output/")
rsa_dir = os.path.join(project_dir, "outputs/rsa/")
space_list = ["T1w", "MNI152NLin2009cAsym"]
tr = 1.058

# getting my functional mask so I can use that information to register masks and stuff
mask_img = os.path.join(mri_dir, f'sub-mrd1011/anat/sub-mrd1011_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
functional_mask_img = nib.load(os.path.join(mri_dir, f'sub-mrd1011/func/sub-mrd1011_task-switch_run-001_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'))
mask_img = nib.load(tf.get("MNI152NLin2009cAsym", desc="brain", resolution=2, suffix= "mask", extension='nii.gz'))
# getting affine and shape information from arbitrary subject since they are all registered to the same template
functional_affine = functional_mask_img.affine
functional_fdata = functional_mask_img.get_fdata()
functional_shape = functional_fdata.shape

bg_img = nib.load(tf.get("MNI152NLin2009cAsym", desc="brain", resolution=2.5, suffix= "T1w", extension='nii.gz'))
bg_img = image.resample_img(bg_img, target_affine=functional_affine, target_shape=functional_shape, interpolation='linear')

# --- MASK bg_img with functional brain mask ---
mask_data = functional_mask_img.get_fdata().astype(bool)
bg_data = bg_img.get_fdata().astype(float)

bg_data[~mask_data] = np.nan   # outside brain → NaN (transparent)
bg_img_masked = nib.Nifti1Image(bg_data, bg_img.affine)

# Command line arguments
if len(sys.argv) > 2:
    roi = sys.argv[1]
    space = space_list[int(sys.argv[2])]
    run_censored_volume_threshold = int(sys.argv[3])
    subj_remaining_runs_threshold = int(sys.argv[4])
    event_of_interest = sys.argv[5]
    atlas = sys.argv[6]
else:
    raise ValueError("Please specify parameters.")

print(f"event of interest: {event_of_interest}")
print(f"roi: {roi}")


# %% helper functions

def vec_upper(rdm, include_diag=True):
    k = 0 if include_diag else 1
    i, j = np.triu_indices_from(rdm, k=k)
    return rdm[i, j].astype(float)

def zero_diag(M):
    M = M.copy()
    np.fill_diagonal(M, 0)
    return M

def zscore_1d(x, eps=1e-12):
    return (x - x.mean()) / (x.std(ddof=1) + eps)

def fit_linear_betas(X_dict, y_vec, standardize=False):
    """
    X_dict: {name: 1D vector} aligned to y_vec
    Returns dict {name: beta} in the order of X_dict keys.
    """
    names = list(X_dict.keys())
    X = np.column_stack([X_dict[n] for n in names]).astype(float)
    y = y_vec.astype(float)

    if standardize:
        X = np.column_stack([zscore_1d(X[:, j]) for j in range(X.shape[1])])
        y = zscore_1d(y)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)
    betas = dict(zip(names, lr.coef_.tolist()))
    return betas

# ----- cross-validated R^2 for variance partitioning -----
def variance_partitioning(predictor_dict, y_vec):
    names = list(predictor_dict.keys())
    X_all = np.column_stack([predictor_dict[n] for n in names])
    y = y_vec.astype(float)

    model_full = LinearRegression(fit_intercept=False).fit(X_all, y)
    full_r2 = model_full.score(X_all, y)

    unique = {}
    for i, n in enumerate(names):
        keep = [j for j in range(len(names)) if j != i]
        X_drop = X_all[:, keep]
        model_drop = LinearRegression(fit_intercept=False).fit(X_drop, y)
        r2_drop = model_drop.score(X_drop, y)
        unique[n] = max(0.0, full_r2 - r2_drop)

    shared_total = max(0.0, full_r2 - sum(unique.values()))
    return {"full_r2": full_r2, "unique_r2": unique, "shared_total_r2": shared_total}

def get_avg_betas_for_run(
        run_idx, beta_maps_list, trial_type_labels_list,
        region_mask, trial_types, save_dir=None):

    # extract trial betas for this run, masked
    trial_beta_maps_run = [
        beta_maps_list[i].get_fdata()[region_mask]
        for i, m in enumerate(run_idx) if m
    ]
    trial_labels_run = [trial_type_labels_list[i] for i, m in enumerate(run_idx) if m]

    # dictionary for trial type → list of beta vectors
    trial_type_betas = {i: [] for i in range(1, trial_types + 1)}

    for trial_idx, beta_vec in enumerate(trial_beta_maps_run):
        trial_type = trial_labels_run[trial_idx]
        trial_type_betas[trial_type].append(beta_vec)

    # Average for each trial type
    avg_betas_run = {
        t: np.mean(vecs, axis=0) if len(vecs) > 0 else None
        for t, vecs in trial_type_betas.items()
    }

    print("Averaged betas:", {k: (None if v is None else v.shape) for k, v in avg_betas_run.items()})

    return avg_betas_run

def vector_to_volume(avg_vec, region_mask, ref_img):
    """Insert masked vector back into 3D brain volume."""
    vol = np.full(bg_img.shape, np.nan)   # fill with NaNs instead of zeros
    vol[region_mask] = avg_vec            # insert actual beta values
    return nib.Nifti1Image(vol, bg_img.affine)

def plot_avg_beta_brain(avg_vec, region_mask, ref_img, trial_type, run_label, outdir=None):
    
    if run_label == "all":
        print("plotting whole brain beta map.")
        display = plot_stat_map(
            avg_vec,
            bg_img=bg_img,
            title=f"{run_label} — Trial type {trial_type}",
            display_mode="ortho",
            black_bg=False,
            alpha=0.7
        )
    else:
        img = vector_to_volume(avg_vec, region_mask, ref_img)

        display = plot_stat_map(
            img,
            bg_img=bg_img,
            title=f"{run_label} — Trial type {trial_type}",
            display_mode="ortho",
            black_bg=False,
            alpha=0.7
        )
    
    # SAVE FIGURE IF REQUESTED
    if outdir is not None:
        if run_label is not "all":
            outfile = os.path.join(rsa_dir, f"{outdir}/{subject}_{roi}_{run_label}_trialtype_{trial_type}.png")
        else:
            outfile = os.path.join(rsa_dir, f"{outdir}/{subject}_wholebrain_{run_label}_trialtype_{trial_type}.png")
        display.savefig(outfile)
        print(f"Saved plot to {outfile}")
    
    display.close()

def calculate_trialtype_to_trialtype_corr(beta_maps_list, trial_type_labels_list, subject_runs, brain_region):
    
    unique_runs = sorted(set(subject_runs))
    odd_runs = [r for r in unique_runs if r % 2 == 1]
    even_runs = [r for r in unique_runs if r % 2 == 0]

    run_idx_odd = np.isin(subject_runs, odd_runs)
    run_idx_even = np.isin(subject_runs, even_runs)
    
    print(run_idx_odd)
    print(run_idx_even)

    region_mask = get_roi_mask(roi=brain_region, atlas=atlas,
                               affine=functional_affine, shape=functional_shape, save_dir=rsa_dir)

    trial_type_number = len(np.unique(trial_type_labels_list))
    print("number of unique trial types: ", trial_type_number)
    
    avg_betas_odd = get_avg_betas_for_run(run_idx_odd, beta_maps_list, trial_type_labels_list,
                                         region_mask, trial_type_number)
    print("avg betas odd: ", avg_betas_odd)

    ref_img = beta_maps_list[0]
    
    outdir = "beta_plots"

    for tt, avg_vec in avg_betas_odd.items():
        if avg_vec is not None:
            plot_avg_beta_brain(avg_vec, region_mask, ref_img,
                                tt, "odd_run", outdir=outdir)
            plot_avg_beta_brain(beta_maps_list[0], bg_img, ref_img,
                                tt, "all", outdir=outdir)
    
    avg_betas_even = get_avg_betas_for_run(run_idx_even, beta_maps_list, trial_type_labels_list, 
                                           region_mask, trial_type_number)
    print("avg betas even: ", avg_betas_even)
    
    correlation_matrix = np.zeros((trial_types, trial_types))

    for i in range(1, trial_types + 1):
        for j in range(1, trial_types + 1):
            print(i, j)
            print(avg_betas_odd[i], avg_betas_even[j])
            corr, _ = spearmanr(avg_betas_odd[i].ravel(), avg_betas_even[j].ravel())
            print(corr)
            correlation_matrix[i-1, j-1] += corr
            correlation_matrix[j-1, i-1] += corr

    correlation_matrix = correlation_matrix/2
    
    output_path = rsa_dir + f'{subject}_{roi}_correlation_matrix_full_{event_of_interest}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(correlation_matrix, f)

    return correlation_matrix

def calculate_trialtype_to_trialtype_crossnobis(beta_maps_list, trial_type_labels_list, subject_runs, brain_region):
    unique_runs = sorted(set(subject_runs))
    odd_runs = [r for r in unique_runs if r % 2 == 1]
    even_runs = [r for r in unique_runs if r % 2 == 0]

    run_idx_odd = np.isin(subject_runs, odd_runs)
    run_idx_even = np.isin(subject_runs, even_runs)

    region_mask = get_roi_mask(roi=brain_region, atlas=atlas,
                               affine=functional_affine, shape=functional_shape, save_dir=rsa_dir)

    avg_betas_odd = get_avg_betas_for_run(run_idx_odd, beta_maps_list, trial_type_labels_list, region_mask)
    avg_betas_even = get_avg_betas_for_run(run_idx_even, beta_maps_list, trial_type_labels_list, region_mask)

    # Build pattern and label lists
    all_patterns = []
    all_stimuli = []
    all_partitions = []

    for trial_type in range(1, trial_types + 1):
        all_patterns.append(avg_betas_odd[trial_type])
        all_stimuli.append(f'trialtype_{trial_type}')
        all_partitions.append(0)

    for trial_type in range(1, trial_types + 1):
        all_patterns.append(avg_betas_even[trial_type])
        all_stimuli.append(f'trialtype_{trial_type}')
        all_partitions.append(1)

    patterns = np.stack(all_patterns)  # shape: [2*n_conditions, n_voxels]
    print(patterns)

    # Wrap as rsatoolbox Dataset
    ds = Dataset(
        data=patterns,
        obs_descriptors={
            'stimulus': all_stimuli,
            'partition': all_partitions
        },
        descriptors={'roi': brain_region}
    )

    # Calculate crossnobis RDM
    rdm = calc_rdm(ds, method='crossnobis', descriptor='stimulus', cv_descriptor='partition')

    # Convert to square numpy array
    crossnobis_matrix = rdm.get_matrices()[0]

    output_path = os.path.join(rsa_dir, f'{subject}_{roi}_crossnobis_matrix_{event_of_interest}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(crossnobis_matrix, f)

    return crossnobis_matrix

def flatten_upper_tri(matrix):
    # triu_indices = np.triu_indices_from(matrix, k=1)
    # print(matrix[triu_indices])
    matrix = matrix.flatten()
    
    return matrix

def perform_regression_on_corr_matrix(subject_corr_matrix, task_rule, goal_location, optimal_movement, trial_type):
    y = flatten_upper_tri(subject_corr_matrix)
    X = np.column_stack([
        flatten_upper_tri(task_rule),
        flatten_upper_tri(goal_location),
        flatten_upper_tri(optimal_movement),
        flatten_upper_tri(trial_type)
    ])

    model = LinearRegression()
    model.fit(X, y)
    betas = model.coef_
    
    print(f"Regression Coefficients for Sub {subject}, ROI {roi}:")
    print(betas)

    with open(rsa_dir + f'{subject}_{roi}_regression_coefficients_{event_of_interest}_full.pkl', 'wb') as f:
        pickle.dump(betas, f)

def perform_correlation_on_corr_matrix(subject_corr_matrix, task_rule, goal_location, optimal_movement, trial_type):
    y = flatten_upper_tri(subject_corr_matrix)

    corr_task_rule, _ = spearmanr(y, flatten_upper_tri(task_rule))
    corr_goal_location, _ = spearmanr(y, flatten_upper_tri(goal_location))
    corr_optimal_movement, _ = spearmanr(y, flatten_upper_tri(optimal_movement))
    corr_trial_type, _ = spearmanr(y, flatten_upper_tri(trial_type))
    
    print(f"Correlation Coefficients for Sub {subject}, ROI {roi}:")
    print(f"Task Rule Correlation: {corr_task_rule}")
    print(f"Goal Location Correlation: {corr_goal_location}")
    print(f"Trial Type Correlation: {corr_trial_type}")

    with open(rsa_dir + f'{subject}_{roi}_correlation_coefficients_{event_of_interest}_full.pkl', 'wb') as f:
        pickle.dump({
            'corr_task_rule': corr_task_rule,
            'corr_goal_location': corr_goal_location,
            'corr_optimal_movement': corr_optimal_movement,
            'trial_type': corr_trial_type
        }, f)

# Generate model RSMs
task_rule = np.zeros((8, 8))
trial_type = np.eye(8)
goal_location = np.eye(8)
optimal_movement = np.eye(8)

# Task rule matrix
task_rule[:4, :4] = 1  # Top-left block for 'Rot' trials
task_rule[4:, 4:] = 1  # Bottom-right block for 'Ref' trials

# Manually specify the ones for goal location matrix (apart from the diagonal)
goal_location[0, 4] = 1
goal_location[1, 5] = 1
goal_location[2, 6] = 1
goal_location[3, 7] = 1

goal_location[4, 0] = 1
goal_location[5, 1] = 1
goal_location[6, 2] = 1
goal_location[7, 3] = 1

# Manually specify the ones for optimal movement matrix (apart from the diagonal)
optimal_movement[0, 4] = 1
optimal_movement[2, 6] = 1

optimal_movement[4, 0] = 1
optimal_movement[6, 2] = 1


subject_list = ["mrd1011", "mrd1012", "mrd1013", "mrd1014", "mrd1017", "mrd1018", "mrd1019",
                "mrd1020", "mrd1021", "mrd1022", "mrd1023", "mrd1025", "mrd1026",
                "mrd1027", "mrd1028", "mrd1031", "mrd1033", "mrd1034", "mrd1035",
                "mrd1036", "mrd1037", "mrd1039", "mrd1040", "mrd1041", "mrd1043", "mrd1044",
                "mrd1029"]


for subject in subject_list:
    # Load subject data
    subject_betas, subject_labels, subject_runs, subject_critical_labels, subject_trial_types, whole_brain_mask, gm_mask = load_data_for_analysis(
        subj_name=subject,
        space=space,
        run_censored_volume_threshold=run_censored_volume_threshold,
        subj_remaining_runs_threshold=subj_remaining_runs_threshold,
        normalization="run_z",
        event_of_interest=event_of_interest,
        filter_trials="none",
        censor_label="no_censoring" # may need to toggle
    )

    trial_types = len(np.unique(subject_trial_types))
    functional_affine = subject_betas[0].affine
    functional_shape = subject_betas[0].shape
    subject_runs = subject_runs

    # Run correlation analysis
    subject_corr_matrix = calculate_trialtype_to_trialtype_corr(
        subject_betas,
        subject_trial_types,
        subject_runs,
        brain_region=roi
    )
    
    # subject_corr_matrix_crossnobis = calculate_trialtype_to_trialtype_crossnobis(beta_maps_list=subject_betas, 
    #                                                                              trial_type_labels_list=subject_trial_types, 
    #                                                                              subject_runs=subject_runs, 
    #                                                                              brain_region=roi)

    # --- Within-taskrule correlations ---
    # Rot trials (0–3)
    rot_block = subject_corr_matrix[:4, :4]
    # Ref trials (4–7)
    ref_block = subject_corr_matrix[4:, 4:]

    # Extract upper triangles (without diagonal)
    rot_within = rot_block[np.triu_indices(4, k=1)]
    ref_within = ref_block[np.triu_indices(4, k=1)]

    # Combine and average
    within_corr = np.nanmean(np.concatenate([rot_within, ref_within]))

    # --- Between-taskrule correlations ---
    # Between Rot (0–3) and Ref (4–7)

    within_corr = (subject_corr_matrix[0,1] + subject_corr_matrix[0,2] +
                      subject_corr_matrix[0,3] + subject_corr_matrix[1,2] +
                      subject_corr_matrix[1,3] + subject_corr_matrix[2,3] +
                      subject_corr_matrix[4,5] + subject_corr_matrix[4,6] +
                      subject_corr_matrix[4,7] + subject_corr_matrix[5,6] +
                      subject_corr_matrix[5,7] + subject_corr_matrix[6,7])/12

    between_corr = (subject_corr_matrix[0,4] + subject_corr_matrix[0,5] +
                      subject_corr_matrix[0,6] + subject_corr_matrix[0,7] +
                      subject_corr_matrix[1,4] + subject_corr_matrix[1,5] +
                      subject_corr_matrix[1,6] + subject_corr_matrix[1,7] +
                      subject_corr_matrix[2,4] + subject_corr_matrix[2,5] +
                      subject_corr_matrix[2,6] + subject_corr_matrix[2,7] +
                   subject_corr_matrix[3,4] + subject_corr_matrix[3,5] +
                   subject_corr_matrix[3,6] + subject_corr_matrix[3,7])/16

    print(f"Within-taskrule correlation: {within_corr:.3f}")
    print(f"Between-taskrule correlation: {between_corr:.3f}")

    wb_corr_list = [within_corr, between_corr]
    print("within/between correlation: ")
    print(wb_corr_list)

    with open(rsa_dir + f'{subject}_{roi}_wb_corr_{event_of_interest}_full.pkl', 'wb') as f:
            pickle.dump(wb_corr_list, f)

    # Plot matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(subject_corr_matrix, annot=True, cmap='viridis', square=True, cbar=True)
    plt.title(f'Trial-Type Correlation Matrix')
    plt.xlabel('Trial Types')
    plt.ylabel('Trial Types')
    plt.savefig(rsa_dir + f'{subject}_{roi}_subject_corr_matrix_{event_of_interest}_full.png', dpi=100)

    # Run analyses
    perform_regression_on_corr_matrix(subject_corr_matrix, task_rule, goal_location, optimal_movement, trial_type)
    perform_correlation_on_corr_matrix(subject_corr_matrix, task_rule, goal_location, optimal_movement, trial_type)
    
    # =====  (1) ABSOLUTE (as-is) regression  =====
    # Keep your matrices exactly as defined earlier (including diagonal).
    # Vectorize INCLUDING diagonal so the reliability signal is preserved.
    y_abs = vec_upper(subject_corr_matrix, include_diag=True)
    X_abs = {
        "task_rule":        vec_upper(task_rule, include_diag=True),
        "goal_location":    vec_upper(goal_location, include_diag=True),
        "optimal_movement": vec_upper(optimal_movement, include_diag=True),
        "trial_type":       vec_upper(trial_type, include_diag=True),
    }
    betas_abs = fit_linear_betas(X_abs, y_abs, standardize=False)
    with open(os.path.join(rsa_dir, f"{subject}_{roi}_betas_absolute_{event_of_interest}.pkl"), "wb") as f:
        pickle.dump(betas_abs, f)

    # =====  (2) Z-SCORED regression  =====
    # Same predictors and y, but standardized so β magnitudes are comparable within this regression.
    betas_z = fit_linear_betas(X_abs, y_abs, standardize=True)
    with open(os.path.join(rsa_dir, f"{subject}_{roi}_betas_zscored_{event_of_interest}.pkl"), "wb") as f:
        pickle.dump(betas_z, f)

    # =====  (3) VARIANCE PARTITIONING (CV R^2) x =====
    # Model diagonal explicitly with a diag mask; zero diagonal on structural RDMs to avoid collinearity with diag.
    D = np.eye(8)  # diagonal mask (captures reliability / same-type)
    vp_mats = {
        "task_rule":        zero_diag(task_rule),
        "goal_location":    zero_diag(goal_location),
        "optimal_movement": zero_diag(optimal_movement),
        "diag_mask":        D,
    }
    # Vectorize INCLUDING diagonal (since diag_mask carries diag; others have diag=0)
    y_vp = vec_upper(subject_corr_matrix, include_diag=True)
    preds_vp = {name: vec_upper(M, include_diag=True) for name, M in vp_mats.items()}

    vp_results = variance_partitioning(preds_vp, y_vp)
    with open(os.path.join(rsa_dir, f"{subject}_{roi}_variance_partitioning_{event_of_interest}.pkl"), "wb") as f:
        pickle.dump(vp_results, f)

    # (Optional) print a compact summary
    print(f"[{subject} | {roi}] Abs betas: {betas_abs}")
    print(f"[{subject} | {roi}] Z betas  : {betas_z}")
    print(f"[{subject} | {roi}] VP       : full={vp_results['full_r2']:.3f}, "
          f"unique={vp_results['unique_r2']}, shared={vp_results['shared_total_r2']:.3f}")

    
    print(f"{subject} calculation complete.")

print("Script completed successfully.")
# %%

