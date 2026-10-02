#!/usr/bin/env python3
# RSA regression coefficients — Figure 3, and Figure S4 with SUPP = True.
# One-sample t vs zero per predictor per ROI, one-tailed, FDR across ROIs.


# %%
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config

SUPP = False                    # True -> S1, SPL, hippocampus (Fig S4)

# %%
rois = config.SUPP_ROIS if SUPP else config.MAIN_ROIS
df = pd.read_csv(os.path.join(config.DATA, "rsa_betas.csv"))

print("=" * 70)
print(f"RSA regression coefficients — {'Fig S4' if SUPP else 'Fig 3'}")
print("=" * 70)
print("one-sample t vs 0, one-tailed, FDR-corrected across ROIs\n")

for predictor in config.PREDICTORS:
    # FDR is corrected across ALL SIXTEEN ROIs at once — the ten main ones and
    # the six supplementary ones — because they form a single family. The
    # analysis pipeline groups these sixteen into one family (labelled
    # "cortical" there, though it includes the hippocampus) and a separate
    # "subcortical" one for striatal ROIs not reported here, and
    # corrects within family, so that testing further subcortical regions could
    # never alter the p-values reported for these. SUPP below chooses only
    # which rows are printed, never which tests are corrected together:
    # correcting the main ten on their own gives different q-values (left LOC
    # trial type: .005 rather than the reported .008).
    rows = []
    for roi in config.ALL_ROIS:
        v = df.loc[df.roi == roi, predictor].dropna().to_numpy()
        t, p = stats.ttest_1samp(v, 0.0, alternative="greater")
        rows.append(dict(roi=roi, n=len(v), M=v.mean(),
                         dz=v.mean() / v.std(ddof=1), t=t, p=p))
    full = pd.DataFrame(rows)
    full["p_fdr"] = multipletests(full.p, alpha=config.ALPHA,
                                  method="fdr_bh")[1]
    out = full[full.roi.isin(rois)].copy()
    out["sig"] = np.where(out.p_fdr < config.ALPHA, "*", "")

    print(f"--- {predictor} ---")
    for _, r in out.iterrows():
        print(f"  {r.roi:<9} t({int(r.n)-1}) = {r.t:5.2f}, "
              f"{config.fmt_p(r.p):<11} FDR {config.fmt_p(r.p_fdr):<11} "
              f"dz = {r.dz:5.2f}  {r.sig}")

    ns = out[out.p_fdr >= config.ALPHA]
    if len(ns):
        # For the "no other region showed this effect (ps > X)" sentences. The
        # claim covers every region in the set at once, so the bound is the
        # SMALLEST p, and it is quoted over the regions PRINTED here, not over
        # the whole correction family — "no other region" in the paper means
        # no other region of the ten a priori ones.
        def floor2(v):
            return f"{np.floor(v * 100) / 100:.2f}".replace("0.", ".", 1)
        print(f"  non-significant: ps_uncorrected > {floor2(ns.p.min())}, "
              f"ps_fdr > {floor2(ns.p_fdr.min())}")
    print()
