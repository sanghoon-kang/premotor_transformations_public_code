# Trial-wise (LSA) first-level GLM for one subject -> trial-wise beta maps.
# Usage: python generate_beta_maps.py <subj> <space_idx> <event> <censor_label>
import warnings
import sys 
if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import os 
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

from nilearn import image
from nilearn.glm import first_level
from nilearn.reporting import make_glm_report
from nilearn.surface import PolyMesh


import pickle

from templateflow import api as tf

sys.path.insert(0, "/home/sk2949/project/github/mrdecode/analysis/fmri/glm_mvpa")

from utils.utils import gen_trialwise_behav, load_func_nii, gen_event_regressors, make_design_matrix

# paths
project_dir = "/home/sk2949/project/mrdecode/"
mri_dir = project_dir + "fmriprep_aroma/output/"
github_dir = "/home/sk2949/project/github/mrdecode/"
behav_dir = github_dir + "task/data/"

tr = 1.058
space_list = ["T1w", "MNI152NLin2009cAsym", "fsnative", "fsaverage"]

if len(sys.argv) > 3:
    subj_name = sys.argv[1]
    print("Received subject name: ", subj_name)
    space = int(sys.argv[2])
    print("Received space option: ", space_list[space])
    event_of_interest = sys.argv[3]
    print("Received event: ", event_of_interest)
    censor_label = sys.argv[4]
    assert censor_label in ["censoring", "no_censoring"]
    print("Received censoring option: ", censor_label)
else:
    print("Please specify subject name, space option (0 for native space and 1 for mni space) and event of interest")

space_idx = int(sys.argv[2])
space = space_list[space_idx] 

surface = space in ["fsnative", "fsaverage"]

    
if censor_label == "censoring":
    censor = True
elif censor_label == "no_censoring":
    censor = False
    
glm_output_dir = project_dir + f"outputs/glm_volume_{event_of_interest}_mask_revision/"

subject_list = [subj_name]
subject = subj_name
# trial onsets from the behavioral files
trialwise_onset = gen_trialwise_behav(subject_list=subject_list)

beta_maps = []
z_maps =[]
labels = []
runs = []

censored_volumes_data = []

subj_output_dir = glm_output_dir+subj_name+"/"
Path(subj_output_dir).mkdir(parents=True, exist_ok=True)

# onsets for this subject
subject_onset = trialwise_onset.loc[trialwise_onset['subject']==subject]

if subject == "mrd1029":
    subject_onset = trialwise_onset.loc[trialwise_onset['subject']=="mrd1026"]

if subject == "mrd1011":
        subject_onset = trialwise_onset.loc[trialwise_onset['subject']=="mrd"]

run_num = len(subject_onset['run'].unique())
if subject in ["mrd1006", "mrd1008"]:
    run_num = 10
print(f"loading data and confounds for subject {subject}")

func_nii_list, confounds_list, masks_list = load_func_nii(sub=subject, space=space, unique_runs=run_num, surface=surface)

print("done loading.")

# reference grid from one subject (all are on the same template)
functional_example_img = nib.load(os.path.join(mri_dir, f'sub-mrd1011/func/sub-mrd1011_task-switch_run-001_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'))
functional_affine = functional_example_img.affine
functional_fdata = functional_example_img.get_fdata()
functional_shape = functional_fdata.shape


mask_img = nib.load(tf.get("MNI152NLin2009cAsym", desc="brain", resolution=2.5, suffix= "mask", extension='nii.gz'))
# resample the template mask to the functional grid
mask_img = image.resample_img(mask_img, target_affine=functional_affine, target_shape=functional_shape, interpolation='nearest')
mask_data = mask_img.get_fdata()

# per-run outputs
subject_beta_maps = []
subject_z_maps = []
subject_labels = []
subject_critical_labels = []
subject_runs = []
subject_trial_type_labels = []

if subject == "mrd1002":
    run_num = 6

# scan runs (a few subjects have exceptions)
run_list = [i+1 for i in range(run_num)]
if subject == "mrd1002":
    run_list = [2,3,4,5,6,8]
elif subject == "mrd1006":
    run_list = [1,2,3,4,6,7,8,9,10]
elif subject == "mrd1008":
    run_list = [2,3,4,5,6,7,8,9,10]


# behavior runs, which can differ from the scan runs
run_onset_list = run_list
if subject == "mrd1002":
    run_onset_list = [1,2,3,4,5,7]
elif subject == "mrd1006":
    run_onset_list = [1,2,3,4,5,6,7,8,9]
elif subject == "mrd1008":
    run_onset_list = [1,2,3,4,5,6,7,8,9]

print(f"using scans from following runs: {run_list} \n using onsets from the following runs: {run_onset_list}")

if space in ["fsnative", "fsaverage"]:
    surface_mesh_left = mri_dir + f"sub-{subject}/anat/sub-{subject}_hemi-L_midthickness.surf.gii"
    surface_mesh_right = mri_dir + f"sub-{subject}/anat/sub-{subject}_hemi-R_midthickness.surf.gii"
    subject_mesh = PolyMesh(left=surface_mesh_left, right=surface_mesh_right)

for i in range(len(run_list)):

    run = (run_list[i]-1)
    run_behav = (run_onset_list[i]-1)

    print(f"first-level glm for subject {subject}, run {run+1}, for {event_of_interest}")

    run_onset = subject_onset.loc[subject_onset['run']==run_behav+1]

    other_regressors, lsa_regressors = gen_event_regressors(trialwise_onset=run_onset, run_num=run_behav+1, event_of_interest=event_of_interest, subject=subject)
    print("lsa regressors: ")
    print(lsa_regressors)
    print("other regressors: ")
    print(other_regressors)
    n_scans = func_nii_list[run].shape[-1]
    confounds = confounds_list[run].filter(['a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 'a_comp_cor_05',
                                                          'rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z', 
                                                          'cosine00', 'cosine01', 'cosine02', 'cosine03', 'cosine04'])

    design_matrix = make_design_matrix(n_scans=n_scans, tr=tr, other_regressors=other_regressors, lsa_regressors=lsa_regressors, confounds=confounds)
    print("design matrix: ")
    print(design_matrix)
    design_matrix.to_csv(subj_output_dir + f"first_level_design_matrix_{subject}_run{run}_{event_of_interest}.csv")
    
    sample_masks = masks_list[run]
    print("sample mask:")
    print(sample_masks)

    if sample_masks is not None:
        # drop the initial volumes
        sample_masks = sample_masks[sample_masks > 5]

        print(f"After scrubbing, {sample_masks.shape[0]} out of {confounds.shape[0]} volumes remain")

        censored_volumes_data.append({
            'subject': subject,
            'run': run + 1,
            'pre-scrubbing': confounds.shape[0],
            'post-scrubbing': sample_masks.shape[0],
            'censored_volumes': confounds.shape[0]-sample_masks.shape[0],
            'mask': sample_masks.tolist()
        })
    else:
        print(f"Warning: sample_masks is None for subject {subject}, run {run+1}. Skipping this run for censored volume processing.")

    print("other regressors:")
    print(other_regressors)
    print("lsa regressors:")
    print(lsa_regressors)

    glm = first_level.FirstLevelModel(t_r=tr,
        noise_model="ar1",
        standardize=True, # within run
        hrf_model="spm",
        drift_model=None, # drift is in the fMRIPrep cosine confounds
        mask_img=mask_img
        )
    
    if censor:
        glm = glm.fit(func_nii_list[run], design_matrices=design_matrix, sample_masks=sample_masks)
    else:
        print ("no censoring/scrubbing due to explicit censor setting.")
        glm = glm.fit(func_nii_list[run], design_matrices=design_matrix, sample_masks=None)
        
    print("glm fit.")

    contrast_matrix = np.eye(design_matrix.shape[1])
    print(contrast_matrix)
    basic_contrast = contrast_matrix[0:5]

    # HTML GLM report
    report = make_glm_report(
        glm,
        contrasts=basic_contrast,
        bg_img=None
    )

    report.save_as_html(subj_output_dir+f"{subject}_run_{run+1}_{space}_{event_of_interest}.html")
    print(f"glm report for first trial saved at {subj_output_dir}.")

    skip = False

    run_beta_maps = []
    run_z_maps = []
    run_labels = []
    run_runs = []
    run_critical_labels = []
    trialwise_conditions = lsa_regressors["trial_type"].unique()
    run_critical_labels = lsa_regressors["critical"]
    
    # trial-wise betas
    for condition in trialwise_conditions:

        print(condition)

        run_runs.append(run+1)
        trial_number = condition.split("__")[1]
        beta_map_nii_dir = subj_output_dir+f'{subject}_run{run+1}_trial{trial_number}_{space}_{event_of_interest}.nii'

        trial_z_map = glm.compute_contrast(condition, output_type="z_score")
        trial_beta_map = glm.compute_contrast(condition, output_type="effect_size")

        trial_condition_name = condition.split("__")[0]

        run_labels.append(trial_condition_name)

        run_beta_maps.append(trial_beta_map)
        run_z_maps.append(trial_z_map)


    # trial type = task x goal location (mirror goals shifted by 4)

    run_goal_location_labels = lsa_regressors["goal_location"].reset_index(drop=True)

    updated_goal_location_labels = [
        goal + 4 if "mirror" in manipulation else goal
        for manipulation, goal in zip(run_labels, run_goal_location_labels)
    ]

    subject_trial_type_labels.append(updated_goal_location_labels)
    subject_labels.append(run_labels)
    subject_beta_maps.append(run_beta_maps)
    subject_z_maps.append(run_z_maps)
    subject_runs.append(run_runs)
    subject_critical_labels.append(run_critical_labels)

    print(f"beta maps and labels extracted for subject {subject} (in {space} space, no smoothing).")

    censored_volumes_df = pd.DataFrame(censored_volumes_data)
    censored_volumes_df.to_csv(subj_output_dir + f"{subject}_censored_volumes.csv", index=False)
    
    
with open(glm_output_dir + f'{subject}_labels.pkl', 'wb') as f:
    pickle.dump(subject_labels, f)
with open(glm_output_dir + f'{subject}_beta_maps_{space}_{event_of_interest}_{censor_label}.pkl', 'wb') as f:
    pickle.dump(subject_beta_maps, f)
with open(glm_output_dir + f'{subject}_z_maps_{space}_{event_of_interest}_{censor_label}.pkl', 'wb') as f:
    pickle.dump(subject_z_maps, f)
with open(glm_output_dir + f'{subject}_runs.pkl', 'wb') as f:
    pickle.dump(subject_runs, f)    
with open(glm_output_dir + f'{subject}_critical_labels.pkl', 'wb') as f:
    pickle.dump(subject_critical_labels, f)
with open(glm_output_dir + f'{subject}_trial_type_labels.pkl', 'wb') as f:
    pickle.dump(subject_trial_type_labels, f)

print(f"beta maps and labels saved as external objects for subject {subject} in {space}.")

print("all beta maps and labels extracted.")