import numpy as np
import os
import scipy.io
import nibabel as nib
import pandas as pd
import pickle

from nilearn import image, surface, plotting, datasets, input_data
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_epi_mask
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.interfaces import fmriprep
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn.plotting import plot_event
from nilearn.glm import first_level
from nilearn.image import smooth_img, math_img, get_data
from nilearn.datasets import load_fsaverage, load_mni152_template
from nilearn.surface import SurfaceImage, PolyMesh, PolyData

from pathlib import Path

import ants
import nibabel as nib
from nibabel.orientations import axcodes2ornt, aff2axcodes, io_orientation, inv_ornt_aff, apply_orientation

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit
from copy import deepcopy
from templateflow import api as tf

import matplotlib.pyplot as plt


# define paths and subjects to process
project_dir = "/home/sk2949/project/"
github_dir = "/home/sk2949/project/github/mrdecode/"
analysis_dir = github_dir + "analysis/"
behav_dir = github_dir + "task/data/"
glm_output_dir = project_dir + "mrdecode/outputs/glm_volume_all/"
mri_dir = project_dir + "mrdecode/fmriprep_aroma/output/"

# generating trialwise df to create one giant onset df
def gen_trialwise_behav(subject_list):
    '''
    given a list of subjects, load all onset files and save as one big dataframe.
    '''
    
    subject_dfs =[]
    
    for sub in subject_list:
        
        subject_directory = behav_dir + sub + "/"
        
        files = os.listdir(subject_directory)
        
        files = [file for file in files if file.endswith('.csv') and '_trial_events_' in file and '_switch_' in file]
        
        #print(files)
        
        dfs =[]
        # mrd1002 started over twice. Account for that
        if sub == "mrd1002":
            files = ["mrd1002_trial_events_04012024150741.csv", "mrd6002_trial_events_04012024155333.csv", "mrd7002_trial_events_04012024161044.csv"]
            for file_num in range(len(files)):
                file = files[file_num]
                print(file)
                df = pd.read_csv(os.path.join(subject_directory, file))
                if file_num > 0:
                    df['run'] = df['run'].transform(lambda x: x + 4 + ((file_num-1)*2))
                    df['block'] = df['block'].transform(lambda x: x + (df.loc[:,'run'].unique()[0]-1)*2)
                    df['trial'] = df['block'].transform(lambda x: x + (df.loc[:,'run'].unique()[0]-1)*32)
                    df.loc[:,'subject'] = sub
                # Append DataFrame to the list
                df = df[df['phase'] == "switch"]
                dfs.append(df)
            
        else:
            for file in files:
                # Load CSV file into a DataFrame
                df = pd.read_csv(os.path.join(subject_directory, file))
                # accounting for starting over after run 4 in mrd1003's session
                if sub == "mrd1003" and '04022024102034' in file:
                    print('yes')
                    df['run'] = df['run'].transform(lambda x: x + 4)
                    print(df.loc[:,'run'].unique())
                    df['block'] = df['block'].transform(lambda x: x + (df.loc[:,'run'].unique()[0]-1)*2)
                    df['trial'] = df['block'].transform(lambda x: x + (df.loc[:,'run'].unique()[0]-1)*32)
                # Append DataFrame to the list
                df = df[df['phase'] == "switch"] 
                dfs.append(df)
        
        # concatenate all dataframes into one df
        subject_trialwise_data = pd.concat(dfs, ignore_index=True)
        
        file_name = behav_dir + f'{sub}/trialwise_data.csv'
        
        subject_trialwise_data.to_csv(file_name)
        
        print(f"trialwise data saved for {file_name}.")
        
        subject_dfs.append(subject_trialwise_data)
        
    final_data = pd.concat(subject_dfs, ignore_index=True)
    
    file_name = behav_dir + 'trialwise_all.csv'
    
    final_data.to_csv(file_name)
    
    print(f"saved as {file_name}.")
    
    return final_data


# polydata to 1d vector
def _pd_to_vec(pd: PolyData) -> np.ndarray:
    parts = []
    for hemi in ("left", "right"):
        if hemi in pd.parts and pd.parts[hemi] is not None:
            a = np.asarray(pd.parts[hemi])
            if a.ndim == 2:     # (n_vertices, n_features) -> take first feature
                a = a[:, 0]
            parts.append(a.astype(np.float32))
    if not parts:
        raise ValueError("PolyData has no left/right arrays.")
    return np.concatenate(parts, axis=0)

def _vec_to_pd(vec: np.ndarray, mesh) -> PolyData:
    left_n  = mesh.parts["left"].n_vertices  if "left"  in mesh.parts else 0
    right_n = mesh.parts["right"].n_vertices if "right" in mesh.parts else 0
    vec = np.asarray(vec, dtype=np.float32).ravel()
    if vec.size != left_n + right_n:
        raise ValueError(f"Vector length {vec.size} != mesh verts {left_n + right_n}")
    left  = vec[:left_n]   if left_n  else None
    right = vec[left_n:]   if right_n else None
    return PolyData(left=left, right=right)

# ---------- SurfaceImage-only normalization ----------
def normalize_surfaceimages(
    images, runs, labels,
    mode="run",           # "run" or "condition"
    method="zscore",      # "zscore" or "center"
    eps=1e-8, verbose=True
):
    """
    Per-vertex normalization; returns a new list[SurfaceImage] on the same fsnative mesh.
    Prints group-wise (avg over vertices) PRE/POST mean & std if verbose=True.
    """
    X = np.vstack([_pd_to_vec(img.data) for img in images])
    runs   = np.asarray(runs)
    labels = np.asarray(labels, dtype=object)

    if mode == "run":
        keys   = sorted(np.unique(runs))
        groups = {g: np.where(runs == g)[0] for g in keys}
        gname  = lambda g: f"run={g}"
    elif mode == "condition":
        keys   = sorted({(int(r), str(l)) for r, l in zip(runs, labels)}, key=lambda t: (t[0], t[1]))
        groups = {g: np.where((runs == g[0]) & (labels == g[1]))[0] for g in keys}
        gname  = lambda g: f"run={g[0]}, label={g[1]}"
    else:
        raise ValueError("mode must be 'run' or 'condition'")

    Xn = np.empty_like(X)
    for g, idx in groups.items():
        mu = X[idx].mean(0, keepdims=True)
        sd = X[idx].std(0,  keepdims=True)
        if method == "zscore":
            Xn[idx] = (X[idx] - mu) / (sd + eps)  # eps avoids divide-by-zero
        elif method == "center":
            Xn[idx] = X[idx] - mu
        else:
            raise ValueError("method must be 'zscore' or 'center'")

        if verbose:
            pre_mean = float(mu.mean()); pre_sd = float(sd.mean())
            post_mu  = float(Xn[idx].mean(0).mean())
            post_sd  = float(Xn[idx].std(0).mean())
            print(f"[{gname(g)}] PRE  mean(avg verts)={pre_mean:.6f}, std(avg verts)={pre_sd:.6f}")
            print(f"[{gname(g)}] POST mean(avg verts)={post_mu:.6f}, std(avg verts)={post_sd:.6f}")

    # re-wrap as SurfaceImages on original meshes
    return [SurfaceImage(mesh=img.mesh, data=_vec_to_pd(Xn[i], img.mesh))
            for i, img in enumerate(images)]

# ---------- loader that RETURNS SurfaceImages ----------
def load_surface_fsnative(
    subj_name,
    glm_output_dir,
    event_of_interest,
    censor_label,
    contrast = "beta",                             # beta map? z map?
    run_censored_volume_threshold=70,     # "censoring" | "no_censoring"
    subj_remaining_runs_threshold=6,
    normalization=None,                   # None | "run" | "condition"
    normalization_method="center",        # "zscore" | "center"
    critical_only=False,
    verbose=True,
    ):
    """
    Return:
      subject_imgs             list[SurfaceImage] (possibly normalized)
      subject_labels           list[str]   (mirror/rotation mapping applied)
      subject_runs             list[int]
      subject_critical_labels  list[bool]
      subject_trial_types      list[int]   (remapped if critical_only=True)
      vertex_mask              np.ndarray[bool] (finite & non-constant; computed pre-norm)
    """
    assert censor_label in ("censoring", "no_censoring")

    # ---- Load saved objects ----
    with open(os.path.join(glm_output_dir, f'{subj_name}_{contrast}_maps_fsnative_{event_of_interest}_{censor_label}.pkl'), 'rb') as f:
        beta_maps_list = pickle.load(f)  # list[runs][trials] of SurfaceImage
    with open(os.path.join(glm_output_dir, f'{subj_name}_labels.pkl'), 'rb') as f:
        labels_list = pickle.load(f)
    with open(os.path.join(glm_output_dir, f'{subj_name}_critical_labels.pkl'), 'rb') as f:
        critical_labels_list = pickle.load(f)
    with open(os.path.join(glm_output_dir, f'{subj_name}_trial_type_labels.pkl'), 'rb') as f:
        trial_type_labels_list = pickle.load(f)

    # ---- Motion-based run filtering ----
    subj_dir = os.path.join(glm_output_dir, subj_name)
    censored = pd.read_csv(os.path.join(subj_dir, f"{subj_name}_censored_volumes.csv"))
    filtered_runs = censored[censored["censored_volumes"] > run_censored_volume_threshold]["run"].tolist()
    if censor_label == "no_censoring":
        filtered_runs = []

    subject_imgs, subject_labels = [], []
    subject_runs, subject_critical_labels, subject_trial_types = [], [], []

    for m, run_beta_list in enumerate(beta_maps_list):  # m = run index (0-based)
        run_id = m + 1
        if run_id in filtered_runs:
            if verbose:
                print(f"[run {run_id}] excluded for motion ({len(critical_labels_list[m])} trials).")
            continue

        crit_flags = list(critical_labels_list[m])
        for k, beta_img in enumerate(run_beta_list):
            if critical_only and not crit_flags[k]:
                continue

            # keep as SurfaceImage
            subject_imgs.append(beta_img)
            subject_labels.append(labels_list[m][k])
            subject_runs.append(run_id)
            subject_critical_labels.append(bool(crit_flags[k]))

            # trial-type mapping
            tt = trial_type_labels_list[m][k]
            if critical_only:
                if   tt == 1: trial_type = 1
                elif tt == 3: trial_type = 2
                elif tt == 5: trial_type = 3
                elif tt == 7: trial_type = 4
                else:
                    if verbose: print("Warning! unexpected trial_type value:", tt)
                    trial_type = tt
            else:
                trial_type = tt
            subject_trial_types.append(int(trial_type))

    if verbose:
        print("number of trials loaded:", len(subject_imgs))

    # normalize textual labels (mirror/rotation)
    subject_labels = ["mirror" if "mirror" in s else "rotation" if "rotation" in s else s
                      for s in subject_labels]

    unique_runs = sorted(set(subject_runs))
    if verbose:
        print("available runs after filtering:", len(unique_runs), unique_runs)
    if len(unique_runs) < subj_remaining_runs_threshold:
        raise ValueError("available runs after motion rejection is below threshold.")

    # ---- Build matrix (pre-norm) for vertex mask ----
    X = np.vstack([_pd_to_vec(img.data) for img in subject_imgs])
    # sanity: consistent vertex length
    if not np.all([v.size == X.shape[1] for v in [X[0]]]):
        raise ValueError("Inconsistent vertex lengths across trials.")
    finite_all = np.isfinite(X).all(axis=0)
    nonconst   = X.std(axis=0) > 0
    vertex_mask = finite_all & nonconst
    if verbose:
        print(f"vertex_mask kept {int(vertex_mask.sum())}/{X.shape[1]} vertices")

    # ---- Optional normalization: stays SurfaceImage throughout ----
    if normalization in ("run", "condition"):
        if verbose:
            print(f"normalizing: mode='{normalization}', method='{normalization_method}'")
        subject_imgs = normalize_surfaceimages(
            images=subject_imgs,
            runs=subject_runs,
            labels=subject_labels,
            mode=normalization,
            method=normalization_method,
            verbose=verbose
        )
        if verbose:
            print("Normalization completed.")

    return (
        subject_imgs,             # list[SurfaceImage] (possibly normalized)
        subject_labels,           # list[str]
        subject_runs,             # list[int]
        subject_critical_labels,  # list[bool]
        subject_trial_types,      # list[int]
        vertex_mask               # np.ndarray[bool]
    )

def gen_event_regressors(trialwise_onset, run_num, event_of_interest, subject):
    import pandas as pd
    import numpy as np

    tw = trialwise_onset[trialwise_onset['run'] == run_num]
    tw = tw[~tw['abort'].isin(['too_slow', 'movement_during_hold_time'])].copy()

    delay_onsets = tw['run_onset_delay']  # (instead of run_onset_move)

    # If we're running the "all" GLM, avoid movement entirely
    if event_of_interest == "all":
        lsa_regressors = pd.DataFrame({
            "trial_type": tw['manipulation_type'].astype(str) + "_all",
            "onset":      pd.to_numeric(tw['run_onset_hold'], errors="coerce"),
            "duration":   pd.to_numeric(tw['end_time'] - tw['run_onset_hold'], errors="coerce"),
            "critical":   tw['which_target'].isin([1, 3]),
            "goal_location": tw['which_target'],
        })

        # Final sanitation to guarantee nilearn-friendly events
        lsa_regressors = lsa_regressors.dropna(subset=["onset", "duration"])
        lsa_regressors = lsa_regressors[lsa_regressors["duration"] >= 0].reset_index(drop=True)

        # Make LSA names unique
        lsa_regressors["trial_type"] = (
            lsa_regressors["trial_type"].astype(str) + "__" +
            (lsa_regressors.index + 1).astype(str).str.zfill(2)
        )

        other_regressors = None
        return other_regressors, lsa_regressors



def gen_event_regressors_localizer(trialwise_onset, run_num, event_of_interest, subject):
    
    '''
    Get event regressors (both irrelevant spike regressors and relevant lsa regressors)
    for use in the design matrix later on. Specifically for localizer
    
    Inputs:
    trialwise_onset: the onset file from get_trialwise_behav().
    run_num: which run number?
    event_of_interest: which event will have the honor of being the one made into an lsa regressor (right now either plan or error)
    
    Outputs:
    spike_regressor (df): a stick regressor with onsets and durations for the other events during the trial that are not relevant but need to be controlled for
    lsa_regressors (df): regressors for each trial, for one event (defined via 'event of interest')
    '''
    
    trialwise_onset = trialwise_onset[trialwise_onset['run'] == run_num]
    trialwise_onset_noabort = trialwise_onset[~trialwise_onset['abort'].isin(['too_slow', 'movement_during_hold_time'])]

    iti_onsets = trialwise_onset_noabort['run_onset_fixation']
    trialwise_onset_noabort.loc[:,'iti_duration'] = trialwise_onset_noabort['iti']
    trialwise_onset_noabort.loc[:,'is_critical'] = trialwise_onset['which_target'].isin([1, 3])
    iti_durations = trialwise_onset_noabort['iti_duration']
    iti_conditions = ['iti']*len(iti_onsets)

    # some variables shared among different phases of a trial
    conditions = trialwise_onset_noabort['manipulation_type']
    print(conditions)
    critical = trialwise_onset_noabort['is_critical']
    
    # information specific to different phases of a trial
    plan_onsets = trialwise_onset_noabort['run_onset_hold']
    trialwise_onset_noabort.loc[:,'plan_duration'] = trialwise_onset_noabort.apply(lambda row: row['run_onset_move'] - row['run_onset_hold'], axis=1)
    plan_durations = trialwise_onset_noabort['plan_duration']
    goal_location = trialwise_onset_noabort['which_target']
    print("conditions length:")
    print(len(conditions))
    
    # doing something different for movement regressor
    move_df = pd.read_csv(os.path.join(analysis_dir, "trialwise_movement_time.csv"))
    move_subj_df = move_df[move_df['subject']==subject]
    move_run_df = move_subj_df[move_subj_df['run']==run_num]
    move_onsets = move_run_df['move_onset']
    move_durations = move_run_df['rt']
    print("move df length: ")
    print(len(move_durations))
    
    delay_onsets = trialwise_onset_noabort['run_onset_move']
    trialwise_onset_noabort.loc[:,'delay_duration'] = trialwise_onset_noabort.apply(lambda row: row['run_onset_feedback'] - row['run_onset_delay'], axis=1)
    delay_durations = trialwise_onset_noabort['delay_duration']
    
    error_onsets = trialwise_onset_noabort['run_onset_feedback']
    trialwise_onset_noabort.loc[:,'error_duration'] = trialwise_onset_noabort.apply(lambda row: row['end_time'] - row['run_onset_feedback'], axis=1)
    error_durations = trialwise_onset_noabort['error_duration']

    # setting onsets and durations for all of the trial
    all_onsets = trialwise_onset_noabort['run_onset_hold']
    plan_durations_all = plan_durations.copy().reset_index(drop=True)
    move_durations_all = move_durations.copy().reset_index(drop=True)
    error_durations_all = error_durations.copy().reset_index(drop=True)
    
    print("plan durations: ")
    print(plan_durations_all)
    print("move durations: ")
    print(move_durations_all)
    print("error durations: ")
    print(error_durations_all)
    
    all_durations = [plan_durations_all[i] + move_durations_all[i] + error_durations_all[i] for i, duration in enumerate(plan_durations_all)]
    
    # now define events
    iti_events = pd.DataFrame(
    {
        "trial_type": conditions + '_iti',
        "onset": iti_onsets,
        "duration": iti_durations,
        "critical": critical,
        "goal_location": goal_location
    }
    )

    plan_events = pd.DataFrame(
    {
        "trial_type": conditions,
        "onset": plan_onsets,
        "duration": plan_durations,
        "critical": critical,
        "goal_location": goal_location
    }
    )
    
    plan_events_localizer = pd.DataFrame(
    {
        "onset": plan_onsets,
        "duration": plan_durations,
        "trial_type": ["plan"]*len(plan_onsets)
    }
    )

    move_events = pd.DataFrame(
    {
        "trial_type": conditions.reset_index(drop=True) + '_move',
        "onset": move_onsets.reset_index(drop=True),
        "duration": move_durations.reset_index(drop=True),
        "critical": critical.reset_index(drop=True),
        "goal_location": goal_location.reset_index(drop=True),
    }
    )
    
    
    move_events_localizer = pd.DataFrame(
    {
        "onset": move_onsets.reset_index(drop=True),
        "duration": move_durations.reset_index(drop=True),
        "trial_type": ["move"]*len(move_onsets.reset_index(drop=True))
    }
    )


    delay_events = pd.DataFrame(
    {
        "trial_type": conditions + '_delay',
        "onset": delay_onsets,
        "duration": delay_durations,
        "critical": critical,
        "goal_location": goal_location
    }
    )

    error_events = pd.DataFrame(
        {
            "trial_type": conditions + '_error',
            "onset": error_onsets,
            "duration": error_durations,
            "critical": critical,
            "goal_location": goal_location
        }
    )
    
    error_events_localizer = pd.DataFrame(
        {
            "onset": error_onsets,
            "duration": error_durations,
            "trial_type": ["error"]*len(error_onsets)
        }
    )
    
    all_events_localizer = pd.DataFrame(
        {
            "trial_type": ["all"]*len(error_onsets),
            "onset": plan_onsets,
            "duration": all_durations
        }
    )
    
    if event_of_interest == "plan":
        other_regressors = pd.concat([move_events, error_events], axis=0)
        lsa_regressors = plan_events.copy()
            
    elif event_of_interest == "error":
        other_regressors = pd.concat([plan_events, move_events], axis=0)
        lsa_regressors = error_events.copy()
    
    elif event_of_interest == "iti":
        other_regressors = pd.concat([plan_events, move_events, error_events], axis=0)
        lsa_regressors = iti_events.copy()
        
    elif event_of_interest == "all":
        lsa_regressors = all_events_localizer.copy()
        other_regressors = None
        print("all regressors: ")
        print(lsa_regressors)
        
    # rename conditions for lsa regressor so it is trial-unique
    # conditions = lsa_regressors["trial_type"].unique()
    # condition_counter = {c: 0 for c in conditions}
    # trial_counter = 0
    # for i_trial, trial in lsa_regressors.iterrows():
    #     trial_condition = trial["trial_type"]
    #     # condition_counter[trial_condition] += 1
    #     # We use a unique delimiter here (``__``) that shouldn't be in the
    #     # original condition names
    #     # also, using trial counter during the run overall instead of condition-specific counter
    #     trial_name = f"{trial_condition}__{trial_counter+1:02d}"
    #     trial_counter +=1
    #     lsa_regressors.loc[i_trial, "trial_type"] = trial_name
    
    
    if event_of_interest == "all":
        regressors = pd.concat([all_events_localizer], axis=0)
    else:
        regressors = pd.concat([plan_events_localizer, move_events_localizer, error_events_localizer], axis=0)
    
    return regressors

def make_design_matrix(n_scans, tr, other_regressors, confounds, lsa_regressors=None):
    '''
    making design matrix for first-level (trial-wise) glm.
    '''
    # Create a full time frame index
    frame_times = np.arange(n_scans) * tr
    frame_times_df = pd.DataFrame(index=frame_times)

    # Generate the design matrices and align them to the full time frame
    lsa_design = make_first_level_design_matrix(frame_times, lsa_regressors, hrf_model='spm', drift_model=None)
    
    if other_regressors is None:
        # If no other regressors, just merge lsa_design with confounds
        # Use a join to handle potential missing time points
        design_matrix = frame_times_df.join(lsa_design, how='left')
        design_matrix = design_matrix.join(confounds, how='left')
    else:
        # Drop the 'constant' column from other_design to avoid duplicates
        other_design = make_first_level_design_matrix(frame_times, other_regressors, hrf_model='spm', drift_model=None).drop(columns=['constant'])
        
        # Join all dataframes to the full time frame index
        design_matrix = frame_times_df.join(lsa_design, how='left')
        design_matrix = design_matrix.join(other_design, how='left')
        design_matrix = design_matrix.join(confounds, how='left')
        
    # Fill any NaN values that might have been created by the joins with 0
    # This assumes missing regressors should be treated as zeros
    design_matrix = design_matrix.fillna(0)

    return design_matrix


def make_design_matrix(n_scans, tr,  other_regressors, confounds, lsa_regressors=None):
    '''
    making design matrix for first-level (trial-wise) glm.
    '''
    if lsa_regressors is None:
        frame_times = np.arange(n_scans) * tr
        other_design = make_first_level_design_matrix(frame_times, other_regressors, hrf_model='spm', drift_model=None)
        print(other_design)
        design_matrix = pd.concat([other_design.reset_index(drop=True), confounds.reset_index(drop=True)], axis=1)
    else:
        frame_times = np.arange(n_scans) * tr
        lsa_design = make_first_level_design_matrix(frame_times, lsa_regressors, hrf_model='spm', drift_model=None)
     
    if other_regressors is None:
        design_matrix = pd.concat([lsa_design.reset_index(drop=True), confounds.reset_index(drop=True)], axis=1)
    else:
        other_design = make_first_level_design_matrix(frame_times, other_regressors, hrf_model='spm', drift_model=None).drop(columns=['constant'])
        design_matrix = pd.concat([lsa_design.reset_index(drop=True), other_design.reset_index(drop=True), confounds.reset_index(drop=True)], axis=1)
    
    return design_matrix


def normalize_nii(beta_map, subject, smoothing, img=False, transformation_type="SyN", radiological=False):
    '''
    Normalizing given beta map with the transformation information generated during preprocessing.
    
    (deprecated; may need to emulate fmriprep registration methods more closely for reliable registration. Registration is reliant on fmriprep now)
    
    Inputs:x
    beta_map (nii): beta maps that must have been produced from first-level glm
    subject (str): subject number (e.g. "mrd1002")
    
    Output:
    normalized_map_nii (nii): beta map that has been normalized
    '''
    
    # Providing image?
    if img:
        # mni_template_nii = image.load_img(f'/home/sk2949/project/mrdecode/fmriprep_multspace/output/sub-{subject}/anat/sub-{subject}_space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz')
        # mni_template_nii = image.load_img(f'/home/sk2949/project/mrdecode/fmriprep_multspace/output/sub-{subject}/anat/sub-{subject}_desc-preproc_T1w.nii.gz')
        mni_template_nii = img
    else:
        # this causes resampling. Make sure you are doing what you are comfortable with
        mni_template_nii = load_mni152_template(resolution=1)
    
    # Convert the MNI template NIfTI to ANTs image
    mni_template = ants.from_nibabel(mni_template_nii)
    
    print(f"fixed image voxel size: {mni_template_nii.header.get_zooms()}")
    print(mni_template_nii.affine)
    print(mni_template.get_orientation())
    
    image_nifti = beta_map
    
    # Convert the image NIfTI to ANTs image
    
    # if in radiological space, switch affine 
#     affine = image_nifti.affine
    
#     if radiological: # switch x-axis orientation if in radiological space
#         affine[0, :] = -affine[0, :]
    
#     image_nifti = nib.Nifti1Image(image_nifti.get_fdata().astype(np.float32),
#                                  affine)
    
    image_ants = ants.from_nibabel(image_nifti)
    # now reorient
    image_ants = image_ants.reorient_image2(orientation=mni_template.get_orientation())
    
    print(f"moving image voxel size: {image_nifti.header.get_zooms()}")
    print(image_nifti.affine)
    print(image_ants.get_orientation())
    
    # Center the moving image to the fixed image
    # moving_centered = ants.resample_image_to_target(image=image_ants, target=mni_template, force_resample=True)
    
    # Perform an initial affine registration
    # initial_registration = ants.registration(
    #     fixed=mni_template, 
    #     moving=image_ants, 
    #     type_of_transform="Affine",  
    #     verbose=False
    # )
    
    # initial_transformed = ants.apply_transforms(
    #     fixed=mni_template, 
    #     moving=image_ants, 
    #     transformlist=initial_registration['fwdtransforms']
    # )
        
    # Perform the SyN registration with adjusted parameters
    registration = ants.registration(
        fixed=mni_template, 
        moving=image_ants, 
        type_of_transform=transformation_type, 
        # syn_metric='meansquares',
        #grad_step=0.1,  # gradient step
        #flow_sigma=3,   # regularization parameters
        #total_sigma=0,  # total sigma for smoothness
        #sampling_strategy='Random', 
        #sampling_percentage=0.2,  
        verbose=False
    )

    transformed_image = ants.apply_transforms(
        fixed=mni_template, 
        moving=image_ants, 
        transformlist=registration['fwdtransforms']
    )
    
    
    # create nii file
    normalized_map_nii = ants.to_nibabel(transformed_image)
    
    print(normalized_map_nii.affine)
    
    if smoothing:
        normalized_map_nii = smooth_img(normalized_map_nii, fwhm=6)
    
    return normalized_map_nii, mni_template_nii

def get_roi_mask(roi, atlas, affine, shape, save_dir=None):
    """
    Get mask for an ROI and save a brain plot of the ROI.

    Parameters
    ----------
    roi : str
        ROI label (HMAT, AAL, Harvard-Oxford, or 'cluster*').
    affine : np.ndarray
        Target affine for resampling.
    shape : tuple
        Target shape for resampling.
    save_dir : str or None
        Directory to save ROI plots. If None, defaults to
        os.path.join(project_dir, 'figures', 'rois').

    Returns
    -------
    boolean_mask : np.ndarray (bool)
        Boolean mask of the ROI in the target space.
    """
    print(roi)

    if save_dir is None:
        save_dir = os.path.join(project_dir, "figures", "rois")
    os.makedirs(save_dir, exist_ok=True)

    subject = "mrd1011"
    # getting my functional mask so I can use that information to register masks and stuff
    mask_img = os.path.join(mri_dir, f'sub-{subject}/anat/sub-{subject}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
    functional_mask_img = nib.load(os.path.join(mri_dir, f'sub-{subject}/func/sub-{subject}_task-switch_run-001_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'))
    # getting affine and shape information from arbitrary subject since they are all registered to the same template
    functional_affine = functional_mask_img.affine
    functional_fdata = functional_mask_img.get_fdata()
    functional_shape = functional_fdata.shape
    
    bg_img = nib.load(tf.get("MNI152NLin2009cAsym", desc="brain", resolution=2.5, suffix= "T1w", extension='nii.gz'))
    bg_img = image.resample_img(bg_img, target_affine=functional_affine, target_shape=functional_shape, interpolation='linear')
    
    
    # juelich atlas
    if atlas == "juelich":
        atlas_img = nib.load(
            os.path.join(project_dir,
                         f"github/mrdecode/analysis/fmri/atlas/juelich_atlas/{roi}.nii.gz")
        )
        resampled_atlas_img = image.resample_img(
            atlas_img, target_affine=affine, target_shape=shape,
            interpolation="nearest", force_resample=True
        )
        atlas_data = resampled_atlas_img.get_fdata()
        boolean_mask = atlas_data.astype(bool) # we'll be indexing data with the region mask so the mask has gotta be boolean
    
    
    # --------------------------------------------------
    # 1) HMAT atlases (already one ROI per file)
    # --------------------------------------------------
    elif "HMAT" in roi:
        atlas_img = nib.load(
            os.path.join(project_dir,
                         f"github/mrdecode/analysis/fmri/atlas/HMAT_website/{roi}.nii")
        )
        resampled_atlas_img = image.resample_img(
            atlas_img, target_affine=affine, target_shape=shape,
            interpolation="nearest"
        )
        atlas_data = resampled_atlas_img.get_fdata()
        boolean_mask = atlas_data.astype(bool)
        #bg_img = resampled_atlas_img

    # --------------------------------------------------
    # 2) Functional localizer clusters (already in target space)
    # --------------------------------------------------
    elif "cluster" in roi:
        cluster_img = nib.load(os.path.join(glm_localizer_dir, f"{roi}.nii.gz"))
        cluster_data = cluster_img.get_fdata()
        boolean_mask = cluster_data.astype(bool)
        bg_img = cluster_img

    # --------------------------------------------------
    # 3) Hippocampus via AAL
    # --------------------------------------------------
    elif "Hippocampus" in roi:
        atlas = datasets.fetch_atlas_aal(
            version="SPM12", data_dir=None, url=None, resume=True, verbose=1
        )
        atlas_labels = [
            lbl.decode("utf-8") if isinstance(lbl, bytes) else lbl
            for lbl in atlas["labels"]
        ]

        atlas_img = image.load_img(atlas["maps"])
        resampled_atlas_img = image.resample_img(
            atlas_img, target_affine=affine, target_shape=shape,
            interpolation="nearest"
        )
        atlas_data = resampled_atlas_img.get_fdata()

        unique_values = np.unique(atlas_data)
        print("unique values (AAL):", unique_values)

        # Map AAL code -> label using actually present values
        code_to_label = dict(zip(unique_values[1:], atlas_labels))

        region_label = roi
        # Find the integer code for this label
        try:
            region_index = next(
                code for code, lbl in code_to_label.items() if lbl == region_label
            )
        except StopIteration:
            raise ValueError(f"Label '{region_label}' not found in AAL atlas.")

        print("AAL region index:", region_index)

        boolean_mask = (atlas_data == region_index)
        bg_img = resampled_atlas_img

    # --------------------------------------------------
    # 4) Everything else: Harvard-Oxford cortical (symmetric split)
    # --------------------------------------------------
    else:
        atlas = datasets.fetch_atlas_harvard_oxford(
            "cort-maxprob-thr25-2mm", symmetric_split=True
        )

        atlas_labels = [
            lbl.decode("utf-8") if isinstance(lbl, bytes) else lbl
            for lbl in atlas["labels"]
        ]

        atlas_img = image.load_img(atlas["maps"])
        resampled_atlas_img = image.resample_img(
            atlas_img, target_affine=affine, target_shape=shape,
            interpolation="nearest"
        )
        atlas_data = resampled_atlas_img.get_fdata()

        unique_values = np.unique(atlas_data)
        print("unique values (HO):", unique_values)

        region_label = roi
        try:
            # In HO, label index corresponds directly to the integer in the volume
            region_index = atlas_labels.index(region_label)
        except ValueError:
            raise ValueError(f"Label '{region_label}' not found in Harvard-Oxford atlas.")

        print("HO region index:", region_index)

        boolean_mask = (atlas_data == region_index)
        bg_img = resampled_atlas_img

    # --------------------------------------------------
    # Plot & save ROI overlay
    # --------------------------------------------------
    roi_img = nib.Nifti1Image(boolean_mask.astype(np.uint8), affine)


    if bg_img is None:
        bg_img = roi_img  # fallback, should rarely happen

    display = plotting.plot_roi(
        roi_img,
        bg_img=bg_img,
        display_mode="ortho",
        title=roi,
        cmap="autumn",  # soft-ish single color; tweak if you like
        alpha=0.7,
    )

    out_path = os.path.join(save_dir, f"{roi}_mask.png")
    display.savefig(out_path, dpi=150, bbox_inches="tight")
    #plotting.close(display)
    print(f"Saved ROI plot to: {out_path}")

    return boolean_mask

import os
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image

from sklearn.preprocessing import StandardScaler
from templateflow import api as tf


def load_data_for_analysis(
    subj_name,
    space,
    run_censored_volume_threshold,
    subj_remaining_runs_threshold,
    normalization,              # "none" | "run_z" | "runcond_z" | "runcond_center"
    event_of_interest,
    filter_trials,
    censor_label               # "censoring" | "no_censoring"
    # glm_output_dir,
    # mri_dir,
):
    """
    Volume-only loader:
    - loads beta maps (Nifti1Image) + labels from pickles
    - optionally filters runs by motion (unless censor_label == 'no_censoring')
    - optionally filters out trials (should explicitly note "none" if not)
    - optional normalization in volume-space (applied to whole-brain masked vectors,
      then rewrapped into NIfTI)
    Returns:
      subject_betas (list of Nifti1Image),
      subject_labels (list),
      subject_runs (list),
      subject_critical_labels (list),
      subject_trial_types (list),
      whole_brain_mask (Nifti1Image),
      gm_mask (Nifti1Image)
    """

    assert censor_label in ["censoring", "no_censoring"]
    if normalization is None:
        normalization = "none"
    assert normalization in ["none", "run_z", "runcond_z", "runcond_center"]
    assert filter_trials in ["critical", "uncritical", "none"]

    # -------------------------
    # Load saved objects
    # -------------------------
    with open(os.path.join(glm_output_dir, f"{subj_name}_beta_maps_{space}_{event_of_interest}_{censor_label}.pkl"), "rb") as f:
        beta_maps_list = pickle.load(f)  # list[runs][trials] -> Nifti1Image

    with open(os.path.join(glm_output_dir, f"{subj_name}_labels.pkl"), "rb") as f:
        labels_list = pickle.load(f)

    with open(os.path.join(glm_output_dir, f"{subj_name}_critical_labels.pkl"), "rb") as f:
        critical_labels_list = pickle.load(f)

    with open(os.path.join(glm_output_dir, f"{subj_name}_trial_type_labels.pkl"), "rb") as f:
        trial_type_labels_list = pickle.load(f)

    # -------------------------
    # Motion-based run filtering
    # -------------------------
    subj_dir = os.path.join(glm_output_dir, subj_name)
    censored = pd.read_csv(os.path.join(subj_dir, f"{subj_name}_censored_volumes.csv"))

    filtered_runs = censored.loc[censored["censored_volumes"] > run_censored_volume_threshold, "run"].tolist()
    if censor_label == "no_censoring":
        filtered_runs = []

    # -------------------------
    # Flatten runs -> trials
    # -------------------------
    subject_betas = []
    subject_labels = []
    subject_runs = []
    subject_critical_labels = []
    subject_trial_types = []

    for m, run_beta_list in enumerate(beta_maps_list):  # m is 0-based run index
        run_num = m + 1
        if run_num in filtered_runs:
            print(f"[run {run_num}] excluded for motion ({len(critical_labels_list[m])} trials).")
            continue

        critical_flags = list(critical_labels_list[m])

        for k, beta in enumerate(run_beta_list):
            if filter_trials == "critical" and not critical_flags[k]: # if only using critical trials, and is not a critical trial, skip!
                continue
            if filter_trials == "uncritical" and critical_flags[k]: # if using only uncritical trials, and is a critical trial, skip!
                continue

            subject_betas.append(beta)
            subject_runs.append(run_num)

            lab = labels_list[m][k]
            # map label strings -> mirror/rotation if they contain those substrings
            if isinstance(lab, str):
                lab_low = lab.lower()
                if "mirror" in lab_low:
                    lab = "mirror"
                elif "rotation" in lab_low:
                    lab = "rotation"
            subject_labels.append(lab)

            subject_critical_labels.append(bool(critical_flags[k]))

            # trial-type mapping (your previous logic)
            tt = trial_type_labels_list[m][k]
            if filter_trials == "critical":
                if tt == 1: trial_type = 1
                elif tt == 3: trial_type = 2
                elif tt == 5: trial_type = 3
                elif tt == 7: trial_type = 4
                else:
                    print("Warning! unexpected trial_type value:", tt)
                    trial_type = tt
                subject_trial_types.append(trial_type)
            elif filter_trials == "uncritical":
                if tt == 2: trial_type = 1
                elif tt == 4: trial_type = 2
                elif tt == 6: trial_type = 3
                elif tt == 8: trial_type = 4
                else:
                    print("Warning! unexpected trial_type value:", tt)
                    trial_type = tt
                subject_trial_types.append(trial_type)
            else:
                subject_trial_types.append(tt)

    print("number of trials loaded:", len(subject_betas))

    unique_runs = sorted(set(subject_runs))
    print("available runs after filtering:", len(unique_runs), unique_runs)
    if len(unique_runs) < subj_remaining_runs_threshold:
        raise ValueError("available runs after motion rejection is below threshold.")

    # -------------------------
    # Build a functional reference (affine/shape)
    # -------------------------
    # Use first included run to pick a mask file
    ref_run = subject_runs[0]
    functional_mask_img = nib.load(os.path.join(
        mri_dir,
        f"sub-{subj_name}/func/sub-{subj_name}_task-switch_run-{ref_run:03}_space-{space}_desc-brain_mask.nii.gz"
    ))
    functional_affine = functional_mask_img.affine
    functional_shape = functional_mask_img.shape

    # -------------------------
    # Whole-brain mask in same grid
    # -------------------------
    # Use TemplateFlow MNI mask and resample to functional grid
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

    # GM mask (optional; still returned)
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

    # -------------------------
    # Vectorize betas in wb mask
    # -------------------------
    brain_vox = whole_brain_mask.get_fdata().astype(bool)

    # NOTE: this assumes each beta is already in the same grid as functional_affine/shape
    # If not, you should resample betas here before indexing.
    X = np.vstack([b.get_fdata()[brain_vox] for b in subject_betas]).astype(np.float32)

    runs_arr = np.asarray(subject_runs)
    y_arr = np.asarray(subject_labels)

    # -------------------------
    # Normalization (volume)
    # -------------------------
    if normalization == "none":
        Xn = X
        print("Normalization skipped.")
    else:
        print(f"normalizing volume betas: mode='{normalization}'")
        Xn = X.copy()  # safe: initialized

        if normalization == "run_z":
            for r in np.unique(runs_arr):
                idx = np.where(runs_arr == r)[0]
                if idx.size == 0:
                    continue
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xn[idx] = scaler.fit_transform(X[idx])

        elif normalization == "runcond_z":
            for r in np.unique(runs_arr):
                for c in np.unique(y_arr):
                    idx = np.where((runs_arr == r) & (y_arr == c))[0]
                    if idx.size == 0:
                        continue
                    scaler = StandardScaler(with_mean=True, with_std=True)
                    Xn[idx] = scaler.fit_transform(X[idx])

        elif normalization == "runcond_center":
            for r in np.unique(runs_arr):
                for c in np.unique(y_arr):
                    idx = np.where((runs_arr == r) & (y_arr == c))[0]
                    if idx.size == 0:
                        continue
                    mu = X[idx].mean(axis=0)
                    Xn[idx] = X[idx] - mu

        else:
            raise ValueError(f"Unknown normalization mode: {normalization}")

        print("Normalization completed.")

    # -------------------------
    # Re-wrap normalized vectors into NIfTI images
    # -------------------------
    norm_betas = []
    for i, beta in enumerate(subject_betas):
        vol = np.zeros(functional_shape, dtype=np.float32)
        vol[brain_vox] = Xn[i]
        # keep original affine/header from beta
        norm_betas.append(nib.Nifti1Image(vol, beta.affine, beta.header))

    subject_betas = norm_betas

    return subject_betas, subject_labels, subject_runs, subject_critical_labels, subject_trial_types, whole_brain_mask, gm_mask



# def load_data_for_analysis(subj_name, space, run_censored_volume_threshold,
#                            subj_remaining_runs_threshold, normalization,
#                            event_of_interest, critiåcal_only, censor_label):
#     """
#     Load beta maps + labels for second-level analyses.
#     Works for both VOLUME (Nifti) and SURFACE (SurfaceImage) data if your pickles
#     contain those objects accordingly.
#     """
#     assert censor_label in ["censoring", "no_censoring"]

#     # ---- Load saved objects ----
#     with open(os.path.join(glm_output_dir, f'{subj_name}_beta_maps_{space}_{event_of_interest}_{censor_label}.pkl'), 'rb') as f:
#         beta_maps_list = pickle.load(f)
#     with open(os.path.join(glm_output_dir, f'{subj_name}_labels.pkl'), 'rb') as f:
#         labels_list = pickle.load(f)
#     with open(os.path.join(glm_output_dir, f'{subj_name}_critical_labels.pkl'), 'rb') as f:
#         critical_labels_list = pickle.load(f)
#     with open(os.path.join(glm_output_dir, f'{subj_name}_trial_type_labels.pkl'), 'rb') as f:
#         trial_type_labels_list = pickle.load(f)

#     # ---- Decide SURFACE vs VOLUME from first item ----
#     # We expect: beta_maps_list[run][trial] is either a Nifti1Image (volume) or a SurfaceImage (surface)
#     first_map = beta_maps_list[0][0]
#     is_surface = hasattr(first_map, "mesh") and hasattr(first_map, "shape") and len(first_map.shape) == 1

#     # ---- Motion-based run filtering ----
#     subj_dir = os.path.join(glm_output_dir, subj_name)
#     censored = pd.read_csv(os.path.join(subj_dir, f"{subj_name}_censored_volumes.csv"))
#     filtered_runs = censored[censored['censored_volumes'] > run_censored_volume_threshold]['run'].tolist()
#     if censor_label == "no_censoring":
#         filtered_runs = []

#     subject_betas = []
#     subject_labels = []
#     subject_runs = []
#     subject_critical_labels = []
#     subject_trial_types = []

#     for m, run_beta_list in enumerate(beta_maps_list):  # m = run index
#         if (m + 1) in filtered_runs:
#             print(f"[run {m+1}] excluded for motion ({len(critical_labels_list[m])} trials).")
#             continue

#         critical_flags = list(critical_labels_list[m])  # ensure list indexing
#         for k, beta in enumerate(run_beta_list):
#             if critical_only and not critical_flags[k]:
#                 continue

#             subject_betas.append(beta)
#             subject_labels.append(labels_list[m][k])
#             subject_runs.append(m + 1)
#             subject_critical_labels.append(critical_flags[k])

#             # trial-type mapping
#             tt = trial_type_labels_list[m][k]
#             if critical_only:
#                 # only using critical trials from each condition
#                 if tt == 1: trial_type = 1
#                 elif tt == 3: trial_type = 2
#                 elif tt == 5: trial_type = 3
#                 elif tt == 7: trial_type = 4
#                 else: 
#                     print("Warning! unexpected trial_type value:", tt)
#                     trial_type = tt
#                 subject_trial_types.append(trial_type)
#             else:
#                 subject_trial_types.append(tt)

#     print("number of trials loaded:", len(subject_betas))
#     subject_labels = ["mirror" if "mirror" in s else "rotation" if "rotation" in s else s for s in subject_labels]
#     unique_runs = sorted(set(subject_runs))
#     print("available runs after filtering:", len(unique_runs), unique_runs)
#     if len(unique_runs) < subj_remaining_runs_threshold:
#         raise ValueError("available runs after motion rejection is below threshold.")

#     # =========================
#     # SURFACE BRANCH
#     # =========================
#     if is_surface:
#         # Build a vertex mask (exclude medial wall) using fsaverage sulc as a simple, robust mask
#         from nilearn import datasets, surface as nlsurf

#         # infer mesh from your GLM export (you used fsaverage5 in plotting; adjust if needed)
#         fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
#         # sulc maps are length n_vertices per hemi; build True/False mask:
#         # use finite sulc as "in-cortex". For combined hemi data, concatenate masks.
#         lh_mask = np.isfinite(nlsurf.load_surf_data(fsavg.sulc_left))
#         rh_mask = np.isfinite(nlsurf.load_surf_data(fsavg.sulc_right))
#         # Heuristic: many SurfaceImages store data concatenated L+R. Try to infer lengths:
#         nvert_l = lh_mask.size
#         nvert_r = rh_mask.size
#         nvert_total = nvert_l + nvert_r

#         # Convert each SurfaceImage to a 1D numpy vector, and apply mask later (caller can decide).
#         def surf_to_vec(img):
#             # Nilearn SurfaceImage stores a 1D signal per vertex; get as ndarray
#             # Try robust accessors:
#             try:
#                 x = np.asarray(img)  # works in newer nilearn
#             except Exception:
#                 try:
#                     x = img.get_fdata()
#                 except Exception:
#                     # last resort:
#                     x = np.array(img.data)
#             x = np.ravel(x).astype(np.float32)
#             if x.shape[0] not in (nvert_l, nvert_r, nvert_total):
#                 # fall back: just return flattened
#                 return x
#             return x

#         subject_vectors = [surf_to_vec(b) for b in subject_betas]

#         # Optional normalization on surface: z-score across trials per vertex (run-wise or condition-wise)
#         if normalization:
#             print(f"normalizing surface betas: mode = '{normalization}'")
#             runs = np.asarray(subject_runs)
#             labels = np.asarray(subject_labels)

#             X = np.vstack(subject_vectors)  # shape: n_trials × n_vertices
#             if normalization == "run":
#                 Xn = np.empty_like(X)
#                 for r in np.unique(runs):
#                     idx = (runs == r)
#                     scaler = StandardScaler(with_mean=True, with_std=True)
#                     Xn[idx] = scaler.fit_transform(X[idx])
#                     np.nan_to_num(Xn[idx], copy=False)
#                 subject_vectors = [row for row in Xn]
#             elif normalization == "condition":
#                 Xn = np.empty_like(X)
#                 for r in np.unique(runs):
#                     for c in np.unique(labels):
#                         idx = (runs == r) & (labels == c)
#                         if idx.size == 0: 
#                             continue
#                         scaler = StandardScaler(with_mean=True, with_std=True)
#                         Xn[idx] = scaler.fit_transform(X[idx])
#                         np.nan_to_num(Xn[idx], copy=False)
#                 subject_vectors = [row for row in Xn]
#             else:
#                 raise ValueError(f"Unknown normalization mode: {normalization}")
#             print("Normalization completed.")
#         else:
#             print("Normalization skipped.")

#         # Build a combined vertex mask (L then R). If your data are interleaved per hemi, adapt this.
#         vertex_mask = np.r_[lh_mask, rh_mask] if subject_vectors[0].shape[0] == nvert_total else np.ones_like(subject_vectors[0], dtype=bool)

#         # Return None for volume masks to keep signature compatible, plus the vertex mask for downstream use
#         whole_brain_mask = None
#         gm_mask = None

#         return subject_vectors, subject_labels, subject_runs, subject_critical_labels, subject_trial_types, whole_brain_mask, gm_mask, vertex_mask

#     # =============================
#     # VOLUME BRANCH
#     # =============================
#     print("Detected VOLUME data; constructing brain masks...")
#     functional_mask_img = nib.load(os.path.join(
#         mri_dir, f'sub-{subj_name}/func/sub-{subj_name}_task-switch_run-{subject_runs[0]:03}_space-{space}_desc-brain_mask.nii.gz'
#     ))
#     functional_affine = functional_mask_img.affine
#     functional_shape = functional_mask_img.shape

#     mask_img = nib.load(tf.get(space, desc="brain", resolution=2, suffix="mask", extension='nii.gz'))
#     mask_img = image.resample_img(mask_img, target_affine=functional_affine, target_shape=functional_shape, interpolation='nearest')
#     mask_data = mask_img.get_fdata()
#     mask_data[mask_data > 0] = 1
#     whole_brain_mask = nib.Nifti1Image(mask_data, mask_img.affine)

#     mni_gm = tf.get(space, label='GM', resolution=2, suffix='probseg', extension='nii.gz')
#     mni_gm_res = image.resample_img(mni_gm, target_affine=functional_affine, target_shape=functional_shape, interpolation='nearest')
#     gm_mask = image.math_img('i1 > 0.2', i1=image.load_img(mni_gm_res))

#     brain_voxels = whole_brain_mask.get_fdata() > 0
#     masked_beta_data = [beta.get_fdata()[brain_voxels] for beta in subject_betas]  # not used later but preserved
    
#     # ---- NORMALIZATION FOR VOLUME (add here) ----
#     runs   = np.asarray(subject_runs)
#     labels = np.asarray(subject_labels)

#     X = np.vstack(masked_beta_data).astype(np.float32)  # n_trials x n_voxels(in mask)

#     if normalization:
#         print(f"normalizing volume betas: mode = '{normalization}'")
#         Xn = np.empty_like(X)
#         if normalization == "run":
#             for r in np.unique(runs):
#                 idx = np.where(runs == r)[0]
#                 print(idx)
#                 if not np.any(idx):
#                     continue
#                 scaler = StandardScaler(with_mean=True, with_std=True)
#                 Xn[idx] = scaler.fit_transform(X[idx])
#                 #np.nan_to_num(Xn[idx], copy=False)
#         elif normalization == "condition":
#             for r in np.unique(runs):
#                 print("run: ", r)
#                 for c in np.unique(labels):
#                     print("condition: ", c)
#                     idx = np.where((runs == r) & (labels == c))[0]
#                     print("id: ", idx)
#                     if not np.any(idx):
#                         continue
#                     scaler = StandardScaler(with_mean=True, with_std=True)
#                     Xn[idx] = scaler.fit_transform(X[idx])
#                     # np.nan_to_num(Xn[idx], copy=False)
#                     # Sanity check: summarize across voxels
#                     mean_after = np.mean(Xn[idx], axis=0)
#                     std_after  = np.std(Xn[idx], axis=0)
#                     print(f"  Condition {c}: mean(mean_voxels)={np.mean(mean_after):.4f}, mean(std_voxels)={np.mean(std_after):.4f}")
#         else:
#             raise ValueError(f"Unknown normalization mode: {normalization}")
#         print("Normalization completed.")
#     else:
#         Xn = X  # no normalization

#     # Re-wrap vectors back into NIfTI volumes so your return type matches current API
#     norm_betas = []
#     for i, beta in enumerate(subject_betas):
#         vol = np.zeros(whole_brain_mask.shape, dtype=np.float32)
#         vol[brain_voxels] = Xn[i]
#         norm_betas.append(nib.Nifti1Image(vol, beta.affine, beta.header))
#     subject_betas = norm_betas

#     return subject_betas, subject_labels, subject_runs, subject_critical_labels, subject_trial_types, whole_brain_mask, gm_mask


def load_atlas(atlas_label):
    '''
    Load Atlas-Related Data, ensures that orientation is neurological.
    
    atlas_label (str): label of the atlas of interest
    
    Returns:
    atlas_data: numpy array corresponding to the atlas, reoriented to neurological orientation
    atlas_label: list of str, list of the names of the brain regions.
    '''
    
    # make sure atlas label is within reasonable bounds
    assert atlas_label in ["aal", "har-oxf"]
        
    # now, load relevant data
    if atlas_label == "aal":
        atlas = datasets.fetch_atlas_aal(version='SPM12')
    elif atlas_label == "har_oxf":
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    
    atlas_img = atlas['maps']
    atlas_labels = atlas['labels']
        
    # Load image
    img = nib.load(atlas_img)

    # Step 1: Get current orientation of axes (e.g., ('L', 'P', 'S'))
    current_ornt = io_orientation(img.affine)
    current_axcodes = aff2axcodes(img.affine)
    print("Current orientation:", current_axcodes)

    # Step 2: Desired orientation (neurological = RAS)
    target_axcodes = ('R', 'A', 'S')
    target_ornt = axcodes2ornt(target_axcodes)

    # Step 3: Get the transform from current to target orientation
    ornt_trans = nib.orientations.ornt_transform(current_ornt, target_ornt)

    # Step 4: Apply the orientation transform to the data
    data = img.get_fdata()
    reoriented_data = apply_orientation(data, ornt_trans)

    # Step 5: Update affine to match new orientation
    new_affine = img.affine @ inv_ornt_aff(ornt_trans, img.shape)

    # Step 6: Save the reoriented image
    reoriented_img = nib.Nifti1Image(reoriented_data, new_affine, img.header)
    print("New orientation:", aff2axcodes(reoriented_img.affine))
    
    atlas_img = reoriented_img

    
    return atlas_data, atlas_labels, new_affine

def load_func_nii(sub, space, unique_runs, surface):
    base = Path(mri_dir)  # e.g., "/.../fmriprep_aroma/output"
    func_nii_list, confounds_list, masks_list = [], [], []

    # runs to skip confounds/mask for (0-based run index)
    bad = {"mrd1006": {4}, "mrd1008": {0}}

    for run in range(int(unique_runs)):
        run_tag = f"{run+1:03d}"
        vol = base / f"sub-{sub}/func/sub-{sub}_task-switch_run-{run_tag}_space-{space}_desc-preproc_bold.nii.gz"

        if surface:
            L_mesh = base / f"sub-{sub}/anat/sub-{sub}_hemi-L_midthickness.surf.gii"
            R_mesh = base / f"sub-{sub}/anat/sub-{sub}_hemi-R_midthickness.surf.gii"
            L_gii  = base / f"sub-{sub}/func/sub-{sub}_task-switch_run-{run_tag}_hemi-L_space-{space}_bold.func.gii"
            R_gii  = base / f"sub-{sub}/func/sub-{sub}_task-switch_run-{run_tag}_hemi-R_space-{space}_bold.func.gii"

            func_nii_list.append(
                SurfaceImage(
                    mesh={"left": str(L_mesh), "right": str(R_mesh)},
                    data={"left": str(L_gii), "right": str(R_gii)},
                )
            )
            conf_key = [str(L_gii), str(R_gii)]  # GIFTI pair is valid
        else:
            func_nii_list.append(image.load_img(str(vol)))
            conf_key = str(vol)

        if sub in bad and run in bad[sub]:
            confounds, mask = None, None
        else:
            confounds, mask = fmriprep.load_confounds(
                conf_key,
                strategy=("motion", "high_pass", "wm_csf", "compcor", "scrub"),
                scrub=5,
                fd_threshold=0.5,
                std_dvars_threshold=1.5,
            )

        confounds_list.append(confounds)
        masks_list.append(mask)

    return func_nii_list, confounds_list, masks_list

