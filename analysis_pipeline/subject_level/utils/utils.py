"""Shared helpers: first-level design, loading trial-wise betas, ROI masks.
Paths are the lab cluster's."""
import numpy as np
import os
import nibabel as nib
import pandas as pd
import pickle
from nilearn import image
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.interfaces import fmriprep
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from templateflow import api as tf

project_dir = "/home/sk2949/project/"
github_dir = "/home/sk2949/project/github/mrdecode/"
behav_dir = github_dir + "task/data/"
glm_output_dir = project_dir + "mrdecode/outputs/glm_volume_all/"
mri_dir = project_dir + "mrdecode/fmriprep_aroma/output/"

def gen_trialwise_behav(subject_list):
    """Combine scanning-phase trial events across subjects and save CSVs."""
    subject_dfs = []
    for sub in subject_list:
        subject_directory = behav_dir + sub + "/"
        files = [f for f in os.listdir(subject_directory)
                 if f.endswith(".csv") and "_trial_events_" in f and "_switch_" in f]
        dfs = []
        for file in files:
            df = pd.read_csv(os.path.join(subject_directory, file))
            dfs.append(df[df["phase"] == "switch"])
        subject_trialwise_data = pd.concat(dfs, ignore_index=True)
        subject_trialwise_data.to_csv(behav_dir + f"{sub}/trialwise_data.csv")
        subject_dfs.append(subject_trialwise_data)

    final_data = pd.concat(subject_dfs, ignore_index=True)
    final_data.to_csv(behav_dir + "trialwise_all.csv")
    return final_data


def gen_event_regressors(trialwise_onset, run_num, event_of_interest, subject):
    """One LSA regressor per trial, hold onset to trial end; aborted trials dropped."""
    assert event_of_interest == "all", event_of_interest
    tw = trialwise_onset[trialwise_onset["run"] == run_num]
    tw = tw[~tw["abort"].isin(["too_slow", "movement_during_hold_time"])].copy()

    lsa_regressors = pd.DataFrame({
        "trial_type": tw["manipulation_type"].astype(str) + "_all",
        "onset":      pd.to_numeric(tw["run_onset_hold"], errors="coerce"),
        "duration":   pd.to_numeric(tw["end_time"] - tw["run_onset_hold"], errors="coerce"),
        "critical":   tw["which_target"].isin([1, 3]),
        "goal_location": tw["which_target"],
    })
    lsa_regressors = lsa_regressors.dropna(subset=["onset", "duration"])
    lsa_regressors = lsa_regressors[lsa_regressors["duration"] >= 0].reset_index(drop=True)

    # unique names so each trial gets its own column
    lsa_regressors["trial_type"] = (
        lsa_regressors["trial_type"].astype(str) + "__" +
        (lsa_regressors.index + 1).astype(str).str.zfill(2)
    )
    return None, lsa_regressors


def make_design_matrix(n_scans, tr, other_regressors, confounds, lsa_regressors=None):
    """LSA regressors (SPM HRF) plus confounds; drift is handled by the fMRIPrep cosines."""
    assert other_regressors is None and lsa_regressors is not None
    frame_times = np.arange(n_scans) * tr
    lsa_design = make_first_level_design_matrix(frame_times, lsa_regressors,
                                                hrf_model="spm", drift_model=None)
    return pd.concat([lsa_design.reset_index(drop=True),
                      confounds.reset_index(drop=True)], axis=1)


def get_roi_mask(roi, atlas, affine, shape, save_dir=None):
    """Boolean Juelich ROI mask (e.g. "PMd_left_volume") on the data's grid; save_dir is unused."""
    assert atlas == "juelich", atlas
    atlas_img = nib.load(os.path.join(
        project_dir, f"github/mrdecode/analysis/fmri/atlas/juelich_atlas/{roi}.nii.gz"))
    resampled = image.resample_img(atlas_img, target_affine=affine, target_shape=shape,
                                   interpolation="nearest", force_resample=True)
    return resampled.get_fdata().astype(bool)


def load_data_for_analysis(
    subj_name,
    space,
    run_censored_volume_threshold,
    subj_remaining_runs_threshold,
    normalization,
    event_of_interest,
    filter_trials,
    censor_label
):
    """Load a subject's trial-wise betas and labels, z-scored per voxel within each run."""

    assert censor_label in ["censoring", "no_censoring"]
    assert normalization == "run_z", normalization
    assert filter_trials == "none", filter_trials

    # load pickles
    with open(os.path.join(glm_output_dir, f"{subj_name}_beta_maps_{space}_{event_of_interest}_{censor_label}.pkl"), "rb") as f:
        beta_maps_list = pickle.load(f)  # [run][trial]

    with open(os.path.join(glm_output_dir, f"{subj_name}_labels.pkl"), "rb") as f:
        labels_list = pickle.load(f)

    with open(os.path.join(glm_output_dir, f"{subj_name}_critical_labels.pkl"), "rb") as f:
        critical_labels_list = pickle.load(f)

    with open(os.path.join(glm_output_dir, f"{subj_name}_trial_type_labels.pkl"), "rb") as f:
        trial_type_labels_list = pickle.load(f)

    # motion-based run exclusion (off for no_censoring)
    subj_dir = os.path.join(glm_output_dir, subj_name)
    censored = pd.read_csv(os.path.join(subj_dir, f"{subj_name}_censored_volumes.csv"))

    filtered_runs = censored.loc[censored["censored_volumes"] > run_censored_volume_threshold, "run"].tolist()
    if censor_label == "no_censoring":
        filtered_runs = []

    # flatten runs -> trials
    subject_betas = []
    subject_labels = []
    subject_runs = []
    subject_critical_labels = []
    subject_trial_types = []

    for m, run_beta_list in enumerate(beta_maps_list):
        run_num = m + 1
        if run_num in filtered_runs:
            print(f"[run {run_num}] excluded for motion ({len(critical_labels_list[m])} trials).")
            continue

        critical_flags = list(critical_labels_list[m])

        for k, beta in enumerate(run_beta_list):
            subject_betas.append(beta)
            subject_runs.append(run_num)

            lab = labels_list[m][k]
            if isinstance(lab, str):
                lab_low = lab.lower()
                if "mirror" in lab_low:
                    lab = "mirror"
                elif "rotation" in lab_low:
                    lab = "rotation"
            subject_labels.append(lab)

            subject_critical_labels.append(bool(critical_flags[k]))

            subject_trial_types.append(trial_type_labels_list[m][k])

    print("number of trials loaded:", len(subject_betas))

    unique_runs = sorted(set(subject_runs))
    print("available runs after filtering:", len(unique_runs), unique_runs)
    if len(unique_runs) < subj_remaining_runs_threshold:
        raise ValueError("available runs after motion rejection is below threshold.")

    # reference grid from the first included run
    ref_run = subject_runs[0]
    functional_mask_img = nib.load(os.path.join(
        mri_dir,
        f"sub-{subj_name}/func/sub-{subj_name}_task-switch_run-{ref_run:03}_space-{space}_desc-brain_mask.nii.gz"
    ))
    functional_affine = functional_mask_img.affine
    functional_shape = functional_mask_img.shape

    # whole-brain and GM masks on that grid
    wb = tf.get(space, desc="brain", resolution=2, suffix="mask", extension="nii.gz")
    wb = image.load_img(wb)
    wb_res = image.resample_img(
        wb,
        target_affine=functional_affine,
        target_shape=functional_shape,
        interpolation="nearest",
        force_resample=True,
    )
    wb_data = wb_res.get_fdata()
    wb_data = (wb_data > 0).astype(np.uint8)
    whole_brain_mask = nib.Nifti1Image(wb_data, wb_res.affine)

    gm = tf.get(space, label="GM", resolution=2, suffix="probseg", extension="nii.gz")
    gm = image.load_img(gm)
    gm_res = image.resample_img(
        gm,
        target_affine=functional_affine,
        target_shape=functional_shape,
        interpolation="nearest",
        force_resample=True,
    )
    gm_mask = image.math_img("i1 > 0.2", i1=gm_res)

    # assumes betas are already on the functional grid
    brain_vox = whole_brain_mask.get_fdata().astype(bool)

    X = np.vstack([b.get_fdata()[brain_vox] for b in subject_betas]).astype(np.float32)

    # z-score each voxel across trials within each run
    runs_arr = np.asarray(subject_runs)
    Xn = X.copy()
    for r in np.unique(runs_arr):
        idx = np.where(runs_arr == r)[0]
        Xn[idx] = StandardScaler(with_mean=True, with_std=True).fit_transform(X[idx])

    # back to NIfTI
    norm_betas = []
    for i, beta in enumerate(subject_betas):
        vol = np.zeros(functional_shape, dtype=np.float32)
        vol[brain_vox] = Xn[i]
        norm_betas.append(nib.Nifti1Image(vol, beta.affine, beta.header))

    subject_betas = norm_betas

    return subject_betas, subject_labels, subject_runs, subject_critical_labels, subject_trial_types, whole_brain_mask, gm_mask


def load_func_nii(sub, space, unique_runs, surface):
    """Preprocessed BOLD runs with their fMRIPrep confounds and scrubbing masks."""
    assert not surface, "every reported analysis is in volume space"
    base = Path(mri_dir)
    func_nii_list, confounds_list, masks_list = [], [], []
    for run in range(int(unique_runs)):
        vol = base / f"sub-{sub}/func/sub-{sub}_task-switch_run-{run+1:03d}_space-{space}_desc-preproc_bold.nii.gz"
        func_nii_list.append(image.load_img(str(vol)))
        confounds, mask = fmriprep.load_confounds(
            str(vol),
            strategy=("motion", "high_pass", "wm_csf", "compcor", "scrub"),
            scrub=5,
            fd_threshold=0.5,
            std_dvars_threshold=1.5,
        )
        confounds_list.append(confounds)
        masks_list.append(mask)
    return func_nii_list, confounds_list, masks_list
