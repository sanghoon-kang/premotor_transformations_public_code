# %% group searchlight tests (Figs 6 and S5): one-sided sign-flip permutations, cluster-mass FWER < .05
# Statistics only: writes the surviving clusters to CLUSTER_DIR ({tag}_t_FWER.nii.gz, {tag}_clusters.csv,
# {tag}_cluster_regions.csv) and prints the cluster report. Reported: whole brain, cluster-forming
# p < .05; the other thresholds, TFCE and the gray-matter mask are sensitivity checks.
# Figures 6 and S5 were made from these outputs; the figure code is not included.
import numpy as np
from nilearn import image
import os
import nibabel as nib
import pandas as pd
from nilearn.glm.second_level import non_parametric_inference
from nilearn.reporting import get_clusters_table
from templateflow import api as tf

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
coverage = {m: 0 for m in need}     # number of participants with a value, per voxel
for s in subject_list:
    for m in need:
        img = image.load_img(f"{data_dir}/searchlight_rsa_{m}_absolute_{s}_all_no_censoring.nii.gz")
        ok = np.isfinite(img.get_fdata())
        valid[m] = ok if valid[m] is None else valid[m] & ok
        coverage[m] = coverage[m] + ok
        subj[m].append(image.smooth_img(img, fwhm=6) if smooth else img)
    print(f"loaded {s}")

# gray-matter mask for the group level (sensitivity check): TemplateFlow GM probability > 0.2,
# the same definition as load_data_for_analysis() in utils.py
_gm = image.resample_to_img(
    image.load_img(tf.get("MNI152NLin2009cAsym", label="GM", resolution=2,
                          suffix="probseg", extension=".nii.gz")),
    subj["task_rule"][0], interpolation="nearest")
GM = _gm.get_fdata() > 0.2
MASK_MODES = ["gm", "wholebrain"]


# %% cluster correction
# one-sided sign-flip permutations, FWER < .05, in the whole brain (reported) or within
# gray matter. Cluster-forming p < .05 is reported in the paper; .001, .005 and .01 are
# sensitivity checks, plus TFCE. Results are cached in CLUSTER_DIR.
CLUSTER_DIR = f"{data_dir}/cluster_test"
os.makedirs(CLUSTER_DIR, exist_ok=True)
CORRECTIONS = ["p001", "p005", "p01", "p05", "tfce"]
FORMING_P = {"p001": 0.001, "p005": 0.005, "p01": 0.01, "p05": 0.05}
N_PERM, ALPHA = 10000, 0.05


def surviving(model, corr, mask_mode="gm"):
    tag = f"{model}_gm_{corr}" if mask_mode == "gm" else f"{model}_{corr}"
    out = f"{CLUSTER_DIR}/{tag}_t_FWER.nii.gz"
    if not os.path.exists(out):
        tfce = corr == "tfce"
        test_mask = valid[model] & GM if mask_mode == "gm" else valid[model]
        mask = nib.Nifti1Image(test_mask.astype(np.uint8), subj[model][0].affine)
        res = non_parametric_inference(
            second_level_input=subj[model],
            design_matrix=pd.DataFrame({"intercept": np.ones(len(subj[model]))}),
            second_level_contrast="intercept", mask=mask, smoothing_fwhm=None,
            model_intercept=False, n_perm=N_PERM, two_sided_test=False,
            threshold=None if tfce else FORMING_P[corr], tfce=tfce, n_jobs=-1, random_state=0)
        for k, v in res.items():
            v.to_filename(f"{CLUSTER_DIR}/{tag}_{k}.nii.gz")
        logp = res["logp_max_tfce" if tfce else "logp_max_mass"].get_fdata()
        t = res["t"]
        surv = logp > -np.log10(ALPHA)
        nib.Nifti1Image(t.get_fdata() * surv, t.affine, t.header).to_filename(out)
        table = get_clusters_table(image.load_img(out), stat_threshold=0,
                                   cluster_threshold=0, min_distance=8)
        table.to_csv(f"{CLUSTER_DIR}/{tag}_clusters.csv", index=False)
    return image.load_img(out).get_fdata() != 0


# %% coverage: voxels with data in every participant (the tested voxels)
print("coverage: voxels with data in all participants:", int(valid["task_rule"].sum()),
      "| in at least one:", int((coverage["task_rule"] > 0).sum()))

# %% run all corrections, gray matter then whole brain
for mode in MASK_MODES:
    for corr in CORRECTIONS:
        for m in need:
            surviving(m, corr, mode)

# %% cluster report: direction, size, cluster p, location (% of the cluster in each Harvard-Oxford
# region, % of each region covered) and overlap with the paper's Julich ROIs; also saved as
# {tag}_cluster_regions.csv in CLUSTER_DIR. nilearn downloads Harvard-Oxford once;
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


UNLABELED = "unlabeled (white matter / outside atlas)"


def atlases_on(ref):
    """Per-voxel Harvard-Oxford region names, region sizes and Julich ROI masks on ref's grid
    (cached per grid)."""
    key = (ref.shape, ref.affine.tobytes())
    if key not in _atlas_grid:
        rs = lambda img: image.resample_to_img(img, ref, interpolation="nearest").get_fdata()
        cort, sub = rs(HO_CORT["maps"]).astype(int), rs(HO_SUB["maps"]).astype(int)
        cort_names = np.array(HO_CORT["labels"], dtype=object)
        sub_names = np.array([l if any(w in l for w in SUB_KEEP) else UNLABELED
                              for l in HO_SUB["labels"]], dtype=object)
        names = np.where(cort > 0, cort_names[cort], np.where(sub > 0, sub_names[sub], UNLABELED))
        u, cnt = np.unique(names, return_counts=True)
        rois = {r: rs(f"{ROI_DIR}/{r}.nii.gz") > 0 for r in ROIS
                if os.path.exists(f"{ROI_DIR}/{r}.nii.gz")}
        _atlas_grid[key] = (names, dict(zip(u, cnt)), rois)
    return _atlas_grid[key]


def cluster_report(name, t_path, fwer_path, logp_path, pos_label, neg_label, csv_path=None,
                   min_pct=2, max_regions=12):
    """Print every surviving cluster with the Harvard-Oxford regions it covers (% of the cluster,
    and % of each region the cluster covers) and its overlap with the Julich ROIs. Writes the
    full breakdown to csv_path. Returns the survivor mask."""
    ref = nib.load(t_path)
    fwer, logp = nib.load(fwer_path).get_fdata(), nib.load(logp_path).get_fdata()
    names, sizes, rois = atlases_on(ref)
    surv = fwer != 0
    rows = []
    print(f"\n{name}: {int(surv.sum())} voxels survive")
    for sign, direction in ((1, pos_label), (-1, neg_label)):
        lab, n = ndimage.label(surv & (np.sign(fwer) == sign))
        for i in range(1, n + 1):
            m = lab == i
            k, p = int(m.sum()), 10 ** -logp[m].max()
            peak = fwer[m][np.argmax(np.abs(fwer[m]))]
            base = dict(direction=direction, cluster=i, cluster_voxels=k,
                        cluster_p=round(p, 4), peak_t=round(float(peak), 2))
            print(f"  {direction} #{i}: {k} vox, peak t {peak:+.2f}, cluster p = {p:.3f}"
                  f"{'  (borderline)' if p > .03 else ''}")
            u, cnt = np.unique(names[m], return_counts=True)
            shown, shown_pct = 0, 0.0
            for j in np.argsort(-cnt):
                pc, pr = 100 * cnt[j] / k, 100 * cnt[j] / sizes[u[j]]
                rows.append({**base, "source": "Harvard-Oxford", "region": u[j], "voxels": int(cnt[j]),
                             "pct_of_cluster": round(pc, 1), "pct_of_region": round(pr, 1)})
                if pc >= min_pct and shown < max_regions:
                    print(f"    {pc:4.0f}%  {u[j]}  (covers {pr:.0f}% of region)")
                    shown, shown_pct = shown + 1, shown_pct + pc
            rest = 100 - shown_pct
            if rest >= 0.5:
                print(f"    {rest:4.0f}%  other regions (each < {min_pct}% of the cluster, or beyond the top {max_regions})")
            hits = []
            for r, msk in rois.items():
                ov = int((m & msk).sum())
                if ov:
                    rows.append({**base, "source": "Julich ROI", "region": r, "voxels": ov,
                                 "pct_of_cluster": round(100 * ov / k, 1),
                                 "pct_of_region": round(100 * ov / msk.sum(), 1)})
                    hits.append(f"{r} {ov} vox ({100 * ov / msk.sum():.0f}% of ROI)")
            print(f"    ROIs: {', '.join(hits) or 'none'}")
    if csv_path:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
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
for mode in MASK_MODES:
    for corr in CORRECTIONS:
        for m in need:
            tag = f"{m}_gm_{corr}" if mode == "gm" else f"{m}_{corr}"
            lp = "logp_max_tfce" if corr == "tfce" else "logp_max_mass"
            SURV[(m, mode, corr)] = cluster_report(
                f"{m}, {mode}, {corr}", f"{CLUSTER_DIR}/{tag}_t.nii.gz",
                f"{CLUSTER_DIR}/{tag}_t_FWER.nii.gz", f"{CLUSTER_DIR}/{tag}_{lp}.nii.gz",
                "coef > 0", "coef < 0", csv_path=f"{CLUSTER_DIR}/{tag}_cluster_regions.csv")
consistency(SURV, need, MASK_MODES, CORRECTIONS)
