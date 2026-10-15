#!/usr/bin/env python3
# Rot/Ref minus Move coefficient, region (premotor vs M1) x hemisphere repeated-measures ANOVA (Results, Fig 3).
import os

import pandas as pd
from statsmodels.stats.anova import AnovaRM

import config

betas = pd.read_csv(os.path.join(config.DATA, "rsa_betas.csv"))
betas["diff"] = betas["Rot/Ref"] - betas["Move"]
betas[["hemi", "region"]] = betas.roi.str.split(" ", expand=True)

for premotor in ("PMd", "PMv"):
    d = betas[betas.region.isin([premotor, "M1"])]
    print(premotor, "vs M1")
    print(AnovaRM(d, "diff", "subject", within=["region", "hemi"]).fit())
