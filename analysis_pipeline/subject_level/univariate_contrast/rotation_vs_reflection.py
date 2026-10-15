#!/usr/bin/env python3
# Whole-brain univariate Rotation vs Reflection contrast (Fig S2).

# %%
import os
import pickle

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from nilearn.glm.second_level import non_parametric_inference
from templateflow import api as tf

PROJECT = "/home/sk2949/project/mrdecode/"
SPACE = "MNI152NLin2009cAsym"
EVENT = "all"
CENSOR_LABEL = "no_censoring"
GLM_DIR = os.path.join(PROJECT, f"outputs/glm_volume_{EVENT}/")
OUT_DIR = os.path.join(GLM_DIR, f"second_level_rotation_minus_mirror_perm_{CENSOR_LABEL}")

ROT, REF = "rotation_all", "mirror_all"      # first-level trial labels
SMOOTH_FWHM = 6.0
N_PERM = 10000
ALPHA = 0.05

SUBJECTS = [
    "mrd1011", "mrd1012", "mrd1013", "mrd1014", "mrd1017", "mrd1018", "mrd1019",
    "mrd1020", "mrd1021", "mrd1022", "mrd1023", "mrd1025", "mrd1026", "mrd1027",
    "mrd1028", "mrd1029", "mrd1031", "mrd1033", "mrd1034", "mrd1035", "mrd1036",
    "mrd1037", "mrd1039", "mrd1040", "mrd1041", "mrd1043", "mrd1044",
]
os.makedirs(OUT_DIR, exist_ok=True)


# %% subject level: Rot - Ref per run, averaged over runs
def run_contrast(betas, labels):
    rot = [b for b, l in zip(betas, labels) if l == ROT]
    ref = [b for b, l in zip(betas, labels) if l == REF]
    return image.math_img("a - b", a=image.mean_img(rot), b=image.mean_img(ref))


contrasts = []
for s in SUBJECTS:
    with open(os.path.join(GLM_DIR, f"{s}_beta_maps_{SPACE}_{EVENT}_{CENSOR_LABEL}.pkl"), "rb") as f:
        betas = pickle.load(f)                 # list over runs
    with open(os.path.join(GLM_DIR, f"{s}_labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    contrasts.append(image.mean_img([run_contrast(b, l) for b, l in zip(betas, labels)]))


# %% group level: two-sided permutation test, cluster-forming p < .05, cluster-mass FWER < .05
brain = nib.load(tf.get(SPACE, desc="brain", resolution=2.5, suffix="mask", extension="nii.gz"))
design = pd.DataFrame({"intercept": np.ones(len(contrasts))})
FORMING_P = {"p05": 0.05}

for corr, thr in FORMING_P.items():
    fwer_path = os.path.join(OUT_DIR, f"{corr}_t_FWER.nii.gz")
    if not os.path.exists(fwer_path):
        res = non_parametric_inference(
            second_level_input=contrasts,
            design_matrix=design,
            second_level_contrast="intercept",
            mask=brain,
            smoothing_fwhm=SMOOTH_FWHM,
            model_intercept=False,                 # intercept is in the design
            n_perm=N_PERM,
            two_sided_test=True,
            threshold=thr,
            n_jobs=-1,
            random_state=0,
        )
        for k, v in res.items():
            v.to_filename(os.path.join(OUT_DIR, f"{corr}_{k}.nii.gz"))
        # voxels in clusters surviving FWER < .05
        logp = res["logp_max_mass"].get_fdata()
        t = res["t"]
        nib.Nifti1Image(t.get_fdata() * (logp > -np.log10(ALPHA)), t.affine,
                        t.header).to_filename(fwer_path)


# %% cluster report
from scipy import ndimage
from nilearn import datasets

ROI_DIR = "/home/sk2949/project/github/mrdecode/analysis/fmri/atlas/juelich_atlas"
ROIS = [f"{r}_{h}_volume" for r in ["M1", "PMd", "PMv", "DLPFC", "LOC", "S1", "SPL", "Hippocampus"]
        for h in ["left", "right"]]
HO_CORT = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", symmetric_split=True)
HO_SUB = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
SUB_KEEP = ("Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala",
            "Accumbens", "Brain-Stem")      # skip HO's cortex / white matter / ventricle labels


def cluster_report(name, fwer_path, logp_path):
    """Each surviving cluster per direction (t = Rotation minus Reflection)."""
    ref = nib.load(fwer_path)
    fwer, logp = ref.get_fdata(), nib.load(logp_path).get_fdata()
    rs = lambda img: image.resample_to_img(img, ref, interpolation="nearest").get_fdata()
    cort, sub = rs(HO_CORT["maps"]).astype(int), rs(HO_SUB["maps"]).astype(int)
    rois = {r: rs(f"{ROI_DIR}/{r}.nii.gz") > 0 for r in ROIS}
    print(f"\n{name}")
    for sign, direction in ((1, "Rot > Ref"), (-1, "Ref > Rot")):
        lab, n = ndimage.label(np.sign(fwer) == sign)
        for i in range(1, n + 1):
            m = lab == i
            k, p = int(m.sum()), 10 ** -logp[m].max()
            names = [HO_CORT["labels"][c] if c > 0 else
                     HO_SUB["labels"][s] if s > 0 and any(w in HO_SUB["labels"][s] for w in SUB_KEEP)
                     else "unlabeled (white matter / outside atlas)"
                     for c, s in zip(cort[m], sub[m])]
            u, cnt = np.unique(names, return_counts=True)
            regions = [f"{u[j]} {100 * cnt[j] / k:.0f}%" for j in np.argsort(-cnt)[:3]]
            hits = [f"{r} {int((m & msk).sum())} vox" for r, msk in rois.items() if (m & msk).any()]
            print(f"  {direction}: {k} vox, cluster p = {p:.3f}")
            print(f"    regions: {'; '.join(regions)}")
            print(f"    ROIs:    {', '.join(hits) or 'none'}")


for corr in FORMING_P:
    cluster_report(f"Rotation - Reflection, whole brain, cluster-forming {corr}",
                   f"{OUT_DIR}/{corr}_t_FWER.nii.gz", f"{OUT_DIR}/{corr}_logp_max_mass.nii.gz")
