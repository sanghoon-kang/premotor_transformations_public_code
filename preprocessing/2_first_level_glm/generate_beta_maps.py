import warnings
import sys 
if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import os 
import nibabel as nib
import numpy as np
import pandas as pd
import math
from pathlib import Path
import matplotlib.pyplot as plt

from nilearn import image, datasets, plotting, input_data
from nilearn.interfaces import fmriprep 
from nilearn.maskers import NiftiMasker
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn.plotting import plot_event, plot_stat_map
from nilearn.glm import first_level
from nilearn.reporting import make_glm_report
from nilearn.surface import SurfaceImage, PolyMesh, PolyData

from scipy import stats
import scipy.io
from sklearn.svm import LinearSVC
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler

import pickle

from IPython.display import display, HTML

from templateflow import api as tf

sys.path.insert(0, "/home/sk2949/project/github/mrdecode/analysis/fmri/glm_mvpa")

from utils.utils import gen_trialwise_behav, load_func_nii, gen_event_regressors, make_design_matrix

# define paths and subjects to process
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
# function to load behavioral data and combine everything so we have onsets
trialwise_onset = gen_trialwise_behav(subject_list=subject_list)

beta_maps = []
z_maps =[]
labels = []
runs = []

# Define the data structure to record censored volumes
censored_volumes_data = []

# making subject-specific directories for beta maps niis and contrast plots
subj_output_dir = glm_output_dir+subj_name+"/"
Path(subj_output_dir).mkdir(parents=True, exist_ok=True)

# load onset file, nii and confounds list
subject_onset = trialwise_onset.loc[trialwise_onset['subject']==subject]

if subject == "mrd1029":
    subject_onset = trialwise_onset.loc[trialwise_onset['subject']=="mrd1026"]

if subject == "mrd1011":
        subject_onset = trialwise_onset.loc[trialwise_onset['subject']=="mrd"]

run_num = len(subject_onset['run'].unique()) # this will mostly be 9
if subject in ["mrd1006", "mrd1008"]:
    run_num = 10
print(f"loading data and confounds for subject {subject}")

# native space
func_nii_list, confounds_list, masks_list = load_func_nii(sub=subject, space=space, unique_runs=run_num, surface=surface)

print("done loading.")

# getting my functional mask so I can use that information to register masks and stuff
functional_example_img = nib.load(os.path.join(mri_dir, f'sub-mrd1011/func/sub-mrd1011_task-switch_run-001_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'))
# getting affine and shape information from arbitrary subject since they are all registered to the same template
functional_affine = functional_example_img.affine
functional_fdata = functional_example_img.get_fdata()
functional_shape = functional_fdata.shape


bg_img = nib.load(tf.get("MNI152NLin2009cAsym", desc="brain", resolution=2.5, suffix= "T1w", extension='nii.gz'))
mask_img = nib.load(tf.get("MNI152NLin2009cAsym", desc="brain", resolution=2.5, suffix= "mask", extension='nii.gz'))
# above mask doesn't work because the affine matrix for your subject's data is a bit weird and does not adhere strictly.
mask_img = image.resample_img(mask_img, target_affine=functional_affine, target_shape=functional_shape, interpolation='nearest')
mask_data = mask_img.get_fdata()

def save_quad_surface_views(
    surf_img,
    subject,
    run=None,
    trial=None,
    map_type=None,
    out_dir=None,
    mesh="fsaverage5",
    threshold=None,
    dpi=300,
    title=None,
    add_one_colorbar=True,   # put a single colorbar on the top-left panel
):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(
        2, 2, subplot_kw={'projection': '3d'},
        figsize=(10, 8), constrained_layout=True
    )

    def panel(mesh_surf, hemi, view, ax, colorbar=False):
        return plotting.plot_surf_stat_map(
            mesh_surf,
            surf_img,               # pass the SurfaceImage directly
            hemi=hemi,
            view=view,
            threshold=threshold,
            colorbar=colorbar,      # let Nilearn make one if requested
            axes=ax, figure=fig
        )

    # Left hemi
    disp_ll = panel(mesh,  'left',  'lateral', axes[0, 0],
                    colorbar=bool(add_one_colorbar))
    axes[0, 0].set_title('Left Lateral')
    panel(mesh,  'left',  'medial',  axes[1, 0])
    axes[1, 0].set_title('Left Medial')

    # Right hemi
    panel(mesh, 'right', 'lateral', axes[0, 1])
    axes[0, 1].set_title('Right Lateral')
    panel(mesh, 'right', 'medial',  axes[1, 1])
    axes[1, 1].set_title('Right Medial')

    if title:
        fig.suptitle(title, y=1.02)

    bits = [subject]
    if run is not None:   bits.append(f"run{int(run):02d}")
    if trial is not None: bits.append(f"trial{int(trial):03d}")
    if map_type is not None: bits.append(f"{map_type}")
    out_path = os.path.join(out_dir, "_".join(bits) + "_surface_views.pdf")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path



# now, go through each run in this subject and extract beta maps and labels.
subject_beta_maps = []
subject_z_maps = []
subject_labels = []
subject_critical_labels = []
subject_runs = []
subject_trial_type_labels = []

if subject == "mrd1002":
    run_num = 6

# define run list for imaging data, note exceptions
run_list = [i+1 for i in range(run_num)]
if subject == "mrd1002":
    run_list = [2,3,4,5,6,8]
elif subject == "mrd1006":
    run_list = [1,2,3,4,6,7,8,9,10]
elif subject == "mrd1008":
    run_list = [2,3,4,5,6,7,8,9,10]


# define run list for behavior onset (usually matches=run_list but not always)
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

    # Create design matrix for this run
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

    # Check if sample_masks is not None before processing
    if sample_masks is not None:
        # Make sure you are discarding first five volumes
        sample_masks = sample_masks[sample_masks > 5]

        print(f"After scrubbing, {sample_masks.shape[0]} out of {confounds.shape[0]} volumes remain")

        # Record the censored volumes for this run
        censored_volumes_data.append({
            'subject': subject,
            'run': run + 1,  # store 1-indexed run
            'pre-scrubbing': confounds.shape[0],
            'post-scrubbing': sample_masks.shape[0],
            'censored_volumes': confounds.shape[0]-sample_masks.shape[0],
            'mask': sample_masks.tolist()  # save the mask as a list (this can help reconstruct the censored volumes)
        })
    else:
        # Log a message or warning if sample_masks is None
        print(f"Warning: sample_masks is None for subject {subject}, run {run+1}. Skipping this run for censored volume processing.")

    print("other regressors:")
    print(other_regressors)
    print("lsa regressors:")
    print(lsa_regressors)

    # Initialize the FirstLevelModel
    glm = first_level.FirstLevelModel(t_r=tr,
        noise_model="ar1",
        # should try standardize = False and True both?
        standardize=True, #standardize within run
        hrf_model="spm",
        drift_model=None, # not modeling drift here because I am using the drift model from fmriprep
        # high_pass=0.01, # cut frequency of the high-pass filter in Hz for the design matrices. 0.01 is default
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
    basic_contrast = contrast_matrix[0:5]  # Example: first trial

    # save report for first trial only (should include an image of design matrix)
    report = make_glm_report(
        glm,
        contrasts=basic_contrast,
        #contrasts=error_events_run1["trial_type"].tolist(),
        bg_img=None # default is the MNI template
    )

    report.save_as_html(subj_output_dir+f"{subject}_run_{run+1}_{space}_{event_of_interest}.html")
    print(f"glm report for first trial saved at {subj_output_dir}.")

    skip = False

    run_beta_maps = []
    run_z_maps = []
    # condition labels? (mirror/rotation) to be used for classification later
    run_labels = []
    run_runs = []
    run_critical_labels = []
    trialwise_conditions = lsa_regressors["trial_type"].unique()
    run_critical_labels = lsa_regressors["critical"]
    
    #trialwise loop
    for condition in trialwise_conditions:

        print(condition)

        run_runs.append(run+1)
        trial_number = condition.split("__")[1]
        beta_map_nii_dir = subj_output_dir+f'{subject}_run{run+1}_trial{trial_number}_{space}_{event_of_interest}.nii'

        trial_z_map = glm.compute_contrast(condition, output_type="z_score")
        trial_beta_map = glm.compute_contrast(condition, output_type="effect_size")
        # trial_beta_map_data = trial_beta_map.get_fdata()
        # print("beta map affine: ")
        # print(trial_beta_map.affine)
        # print("beta map shape: ")
        # print(trial_beta_map_data.shape)

        # save beta map - do we need this?
        # nib.save(trial_beta_map, beta_map_nii_dir)
        # print(f"beta map saved: {beta_map_nii_dir}")

#         if trial_number == 1:

#             # load/resample background image
#             zmap_path = save_quad_surface_views(
#                 surf_img=trial_z_map,
#                 subject=subject,
#                 run=run+1,
#                 trial=trial_number,
#                 map_type="z",
#                 out_dir=subj_output_dir,
#                 mesh=subject_mesh,
#                 threshold=0,
#                 title="Trial Z Map"
#                 )
#             print("z map saved to:", zmap_path)

#             betamap_path = save_quad_surface_views(
#                 surf_img=trial_beta_map,
#                 subject=subject,
#                 run=run+1,
#                 trial=trial_number,
#                 map_type="beta",
#                 out_dir=subj_output_dir,
#                 mesh=subject_mesh,
#                 threshold=0,
#                 title="Trial Beta Map"
#                 )
#             print("beta map saved to:", betamap_path)

        # Drop the trial number from the condition name to get the original name
        trial_condition_name = condition.split("__")[0]

        # make label list for this run (mirror vs. rotation)
        run_labels.append(trial_condition_name)

        # save the niis in a list
        run_beta_maps.append(trial_beta_map)
        run_z_maps.append(trial_z_map)


    # make trial type (task x goal location) labels for this run

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

    # Save the list of censored volumes for this subject to a CSV
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