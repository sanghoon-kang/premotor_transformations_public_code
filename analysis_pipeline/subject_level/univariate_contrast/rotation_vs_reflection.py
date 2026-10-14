#!/usr/bin/env python3
# Whole-brain univariate Rotation vs Reflection control, from the uncensored
# trial-wise beta maps. Group-level permutation test with FWER < .05.
# Statistics only: writes the t-maps, surviving clusters ({corr}_t_FWER.nii.gz,
# {corr}_clusters.csv) and prints the cluster report. Figure code is not included.

# %%
import os
import pickle

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from nilearn.glm.second_level import non_parametric_inference
from nilearn.reporting import get_clusters_table
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


# %% group level: one permutation test per correction and mask
# two-sided. Cluster-forming p < .05 is reported (Fig S6 = whole brain p05; same
# threshold as the searchlight); .001, .005 and .01 are sensitivity checks, plus TFCE.
# Whole brain (reported) and gray matter; finished tests are reused.
brain = nib.load(tf.get(SPACE, desc="brain", resolution=2.5, suffix="mask", extension="nii.gz"))
gm = image.resample_to_img(
    tf.get(SPACE, label="GM", resolution=2, suffix="probseg", extension="nii.gz"),
    brain, interpolation="nearest")
MASKS = {
    "gm": (image.math_img("(b > 0) & (g > 0.2)", b=brain, g=gm), OUT_DIR + "_gm"),
    "wholebrain": (brain, OUT_DIR),
}
design = pd.DataFrame({"intercept": np.ones(len(contrasts))})
CORRECTIONS = ["p001", "p005", "p01", "p05", "tfce"]
FORMING_P = {"p001": 0.001, "p005": 0.005, "p01": 0.01, "p05": 0.05}


for mode, (mask, out_dir) in MASKS.items():
    os.makedirs(out_dir, exist_ok=True)
    for corr in CORRECTIONS:
        fwer_path = os.path.join(out_dir, f"{corr}_t_FWER.nii.gz")
        if not os.path.exists(fwer_path):
            tfce = corr == "tfce"
            res = non_parametric_inference(
                second_level_input=contrasts,
                design_matrix=design,
                second_level_contrast="intercept",
                mask=mask,
                smoothing_fwhm=SMOOTH_FWHM,
                model_intercept=False,                 # intercept is in the design
                n_perm=N_PERM,
                two_sided_test=True,
                threshold=None if tfce else FORMING_P[corr],
                tfce=tfce,
                n_jobs=-1,
                random_state=0,
            )
            for k, v in res.items():
                v.to_filename(os.path.join(out_dir, f"{corr}_{k}.nii.gz"))
            # voxels in clusters surviving FWER < .05
            logp = res["logp_max_tfce" if tfce else "logp_max_mass"].get_fdata()
            t = res["t"]
            nib.Nifti1Image(t.get_fdata() * (logp > -np.log10(ALPHA)), t.affine,
                            t.header).to_filename(fwer_path)
        t_fwer = image.load_img(fwer_path)
        table = get_clusters_table(t_fwer, stat_threshold=0, cluster_threshold=0,
                                   min_distance=8, two_sided=True)
        if not table.empty:   # t is Rotation minus Reflection
            table["Direction"] = np.where(table["Peak Stat"] > 0, "Rot > Ref", "Ref > Rot")
        table.to_csv(os.path.join(out_dir, f"{corr}_clusters.csv"), index=False)


# %% cluster report: direction, size, cluster p, location (% of voxels per Harvard-Oxford
# region) and overlap with the paper's Julich ROIs. nilearn downloads Harvard-Oxford once;
# if a compute node has no internet, run the import/fetch lines once on a login node.
from scipy import ndimage
from nilearn import datasets

ROI_DIR = "/home/sk2949/project/github/mrdecode/analysis/fmri/atlas/juelich_atlas"
ROIS = [f"{r}_{h}_volume" for r in ["M1", "PMd", "PMv", "DLPFC", "LOC", "S1", "SPL", "Hippocampus"]
        for h in ["left", "right"]]
HO_CORT = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", symmetric_split=True)
HO_SUB = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
SUB_KEEP = ("Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala",
            "Accumbens", "Brain-Stem")      # skip HO's cortex / white matter / ventricle labels
_atlas_grid = {}


def atlases_on(ref):
    """Harvard-Oxford labels and Julich ROI masks on ref's grid (cached per grid)."""
    key = (ref.shape, ref.affine.tobytes())
    if key not in _atlas_grid:
        rs = lambda img: image.resample_to_img(img, ref, interpolation="nearest").get_fdata()
        rois = {r: rs(f"{ROI_DIR}/{r}.nii.gz") > 0 for r in ROIS
                if os.path.exists(f"{ROI_DIR}/{r}.nii.gz")}
        _atlas_grid[key] = (rs(HO_CORT["maps"]).astype(int), rs(HO_SUB["maps"]).astype(int), rois)
    return _atlas_grid[key]


def cluster_report(name, t_path, fwer_path, logp_path, pos_label, neg_label):
    """Print every surviving cluster; return the survivor mask."""
    ref = nib.load(t_path)
    fwer, logp = nib.load(fwer_path).get_fdata(), nib.load(logp_path).get_fdata()
    cort, sub, rois = atlases_on(ref)
    surv = fwer != 0
    print(f"\n{name}: {int(surv.sum())} voxels survive")
    for sign, direction in ((1, pos_label), (-1, neg_label)):
        lab, n = ndimage.label(surv & (np.sign(fwer) == sign))
        for i in range(1, n + 1):
            m = lab == i
            k, p = int(m.sum()), 10 ** -logp[m].max()
            peak = fwer[m][np.argmax(np.abs(fwer[m]))]
            names = [HO_CORT["labels"][c] if c > 0 else
                     HO_SUB["labels"][s] if s > 0 and any(w in HO_SUB["labels"][s] for w in SUB_KEEP)
                     else "unlabeled (white matter / outside atlas)"
                     for c, s in zip(cort[m], sub[m])]
            u, cnt = np.unique(names, return_counts=True)
            order = np.argsort(-cnt)
            regions = [f"{u[j]} {100 * cnt[j] / k:.0f}%" for j in order[:3]]
            if len(order) > 3:
                regions.append(f"other {100 * cnt[order[3:]].sum() / k:.0f}%")
            hits = [f"{r} {int((m & msk).sum())} vox ({100 * (m & msk).sum() / k:.0f}%)"
                    for r, msk in rois.items() if (m & msk).any()]
            print(f"  {direction}: {k} vox, peak t {peak:+.2f}, cluster p = {p:.3f}"
                  f"{'  (borderline)' if p > .03 else ''}")
            print(f"    regions: {'; '.join(regions)}")
            print(f"    ROIs:    {', '.join(hits) or 'none'}")
    return surv


def consistency(surv, keys, masks, corrs):
    """Gray matter vs whole brain, and stricter vs looser thresholds."""
    print("\nconsistency")
    for key in keys:
        for c in corrs:
            g, w = surv.get((key, "gm", c)), surv.get((key, "wholebrain", c))
            if g is not None and w is not None and g.any():
                print(f"  {key} {c}: {(g & w).sum() / g.sum():.0%} of gray-matter survivors "
                      f"also survive whole brain")
        for mode in masks:
            seq = [c for c in corrs if c != "tfce" and (key, mode, c) in surv]
            pairs = [f"{a}->{b} {(surv[(key, mode, a)] & surv[(key, mode, b)]).sum() / surv[(key, mode, a)].sum():.0%}"
                     for a, b in zip(seq, seq[1:]) if surv[(key, mode, a)].any()]
            if pairs:
                print(f"  {key} {mode}: stricter survivors kept at looser threshold: {', '.join(pairs)}")

SURV = {}
for mode, (mask, out_dir) in MASKS.items():
    for corr in CORRECTIONS:
        lp = "logp_max_tfce" if corr == "tfce" else "logp_max_mass"
        SURV[("rot-ref", mode, corr)] = cluster_report(
            f"Rotation - Reflection, {mode}, {corr}", f"{out_dir}/{corr}_t.nii.gz",
            f"{out_dir}/{corr}_t_FWER.nii.gz", f"{out_dir}/{corr}_{lp}.nii.gz",
            "Rot > Ref", "Ref > Rot")
consistency(SURV, ["rot-ref"], MASKS, CORRECTIONS)
