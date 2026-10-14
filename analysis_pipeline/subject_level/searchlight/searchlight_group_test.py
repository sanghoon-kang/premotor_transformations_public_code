# %% group searchlight tests (Figs 6 and S6): one-sided sign-flip permutations, cluster-mass FWER < .05,
# whole brain, cluster-forming p < .05 (reported) and p < .001 (reported in the text).
# Writes the surviving clusters to CLUSTER_DIR and prints each cluster's size, p, location and
# overlap with the Julich ROIs. Figure code is not included.
import numpy as np
from nilearn import image
import os
import nibabel as nib
import pandas as pd
from nilearn.glm.second_level import non_parametric_inference
from scipy import ndimage
from nilearn import datasets

# %% config
subject_list = ["mrd1011", "mrd1012", "mrd1013", "mrd1014", "mrd1017", "mrd1018", "mrd1019",
                "mrd1020", "mrd1021", "mrd1022", "mrd1023", "mrd1025", "mrd1026",
                "mrd1027", "mrd1028", "mrd1031", "mrd1033", "mrd1034", "mrd1035",
                "mrd1036", "mrd1037", "mrd1039", "mrd1040", "mrd1041", "mrd1043", "mrd1044",
                "mrd1029"]
data_dir        = "/home/sk2949/project/mrdecode/outputs/searchlight_rsa_intercept"
smooth          = True


# %% load subject maps
need = ["task_rule", "optimal_movement"]
subj = {m: [] for m in need}
valid = {m: None for m in need}     # voxels with a value in every participant
for s in subject_list:
    for m in need:
        img = image.load_img(f"{data_dir}/searchlight_rsa_{m}_absolute_{s}_all_no_censoring.nii.gz")
        ok = np.isfinite(img.get_fdata())
        valid[m] = ok if valid[m] is None else valid[m] & ok
        subj[m].append(image.smooth_img(img, fwhm=6) if smooth else img)
    print(f"loaded {s}")


# %% cluster correction (cached in CLUSTER_DIR)
CLUSTER_DIR = f"{data_dir}/cluster_test"
os.makedirs(CLUSTER_DIR, exist_ok=True)
FORMING_P = {"p05": 0.05, "p001": 0.001}
N_PERM, ALPHA = 10000, 0.05


def surviving(model, corr):
    tag = f"{model}_{corr}"
    out = f"{CLUSTER_DIR}/{tag}_t_FWER.nii.gz"
    if not os.path.exists(out):
        mask = nib.Nifti1Image(valid[model].astype(np.uint8), subj[model][0].affine)
        res = non_parametric_inference(
            second_level_input=subj[model],
            design_matrix=pd.DataFrame({"intercept": np.ones(len(subj[model]))}),
            second_level_contrast="intercept", mask=mask, smoothing_fwhm=None,
            model_intercept=False, n_perm=N_PERM, two_sided_test=False,
            threshold=FORMING_P[corr], n_jobs=-1, random_state=0)
        for k, v in res.items():
            v.to_filename(f"{CLUSTER_DIR}/{tag}_{k}.nii.gz")
        logp = res["logp_max_mass"].get_fdata()
        t = res["t"]
        surv = logp > -np.log10(ALPHA)
        nib.Nifti1Image(t.get_fdata() * surv, t.affine, t.header).to_filename(out)


# %% cluster report: size, cluster p, Harvard-Oxford regions (% of the cluster) and overlap
# with the paper's Julich ROIs. nilearn downloads Harvard-Oxford once; if a compute node has
# no internet, run the fetch lines once on a login node.
ROI_DIR = "/home/sk2949/project/github/mrdecode/analysis/fmri/atlas/juelich_atlas"
ROIS = [f"{r}_{h}_volume" for r in ["M1", "PMd", "PMv", "DLPFC", "LOC", "S1", "SPL", "Hippocampus"]
        for h in ["left", "right"]]
HO_CORT = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", symmetric_split=True)
HO_SUB = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
SUB_KEEP = ("Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala",
            "Accumbens", "Brain-Stem")      # skip HO's cortex / white matter / ventricle labels
UNLABELED = "unlabeled (white matter / outside atlas)"


def atlases_on(ref):
    """Per-voxel Harvard-Oxford region names and Julich ROI masks on ref's grid."""
    rs = lambda img: image.resample_to_img(img, ref, interpolation="nearest").get_fdata()
    cort, sub = rs(HO_CORT["maps"]).astype(int), rs(HO_SUB["maps"]).astype(int)
    cort_names = np.array(HO_CORT["labels"], dtype=object)
    sub_names = np.array([l if any(w in l for w in SUB_KEEP) else UNLABELED
                          for l in HO_SUB["labels"]], dtype=object)
    names = np.where(cort > 0, cort_names[cort], np.where(sub > 0, sub_names[sub], UNLABELED))
    rois = {r: rs(f"{ROI_DIR}/{r}.nii.gz") > 0 for r in ROIS}
    return names, rois


def cluster_report(name, fwer_path, logp_path, min_pct=2):
    """Each surviving cluster: size, cluster p, regions covering >= min_pct of it, ROI overlap."""
    ref = nib.load(fwer_path)
    fwer, logp = ref.get_fdata(), nib.load(logp_path).get_fdata()
    names, rois = atlases_on(ref)
    lab, n = ndimage.label(fwer > 0)
    print(f"\n{name}: {n} cluster(s)")
    for i in range(1, n + 1):
        m = lab == i
        k, p = int(m.sum()), 10 ** -logp[m].max()
        print(f"  #{i}: {k} vox, peak t {fwer[m].max():.2f}, cluster p = {p:.3f}")
        u, cnt = np.unique(names[m], return_counts=True)
        for j in np.argsort(-cnt):
            if 100 * cnt[j] / k >= min_pct:
                print(f"    {100 * cnt[j] / k:4.0f}%  {u[j]}")
        hits = [f"{r} {int((m & msk).sum())} vox" for r, msk in rois.items() if (m & msk).any()]
        print(f"    ROIs: {', '.join(hits) or 'none'}")


for corr in FORMING_P:
    for m in need:
        surviving(m, corr)
        cluster_report(f"{m}, whole brain, cluster-forming {corr}",
                       f"{CLUSTER_DIR}/{m}_{corr}_t_FWER.nii.gz",
                       f"{CLUSTER_DIR}/{m}_{corr}_logp_max_mass.nii.gz")
