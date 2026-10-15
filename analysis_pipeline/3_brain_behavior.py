#!/usr/bin/env python3
# Rot/Ref coefficient vs mean absolute initial error across participants (Fig 4c-d).
import os

import pandas as pd
from scipy.stats import pearsonr, spearmanr

import config

behavior = pd.read_csv(os.path.join(config.DATA, "behavior.csv"))
betas = pd.read_csv(os.path.join(config.DATA, "rsa_betas.csv"))

# mean absolute initial error per participant (trial-type cells averaged first)
b = behavior.dropna(subset=["initial_heading_deg"])
cell = (b.assign(abs_err=b.initial_heading_deg.abs())
         .groupby(["subject", "trial", "manipulation_type", "critical"],
                  dropna=False)["abs_err"].mean())
error = cell.groupby("subject").mean().rename("error")

rows = []
for roi in ["L PMd", "R PMd", "L PMv", "R PMv", "L M1", "R M1"] + config.SUPP_ROIS:
    m = pd.concat([betas[betas.roi == roi].set_index("subject")["Rot/Ref"], error],
                  axis=1, join="inner").dropna()
    r, p = pearsonr(m["Rot/Ref"], m.error)
    rho, p_rho = spearmanr(m["Rot/Ref"], m.error)
    rows.append(dict(roi=roi, n=len(m), r=r, p=p, rho=rho, p_rho=p_rho))
print(pd.DataFrame(rows).to_string(index=False))
