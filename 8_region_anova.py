#!/usr/bin/env python3
# Premotor vs M1: are context and movement weighted differently? — Results,
# the paragraph after Figure 3.
#
# Difference-of-differences. The dependent variable is already a difference
# within each ROI (transformation context minus optimal movement), so the
# "ROI x coefficient-type interaction" the paper reports is the MAIN EFFECT of
# region on that difference — an interaction in the untransformed coefficients
# becomes a main effect once the difference is taken. The ROI x hemisphere
# interaction is the interaction term.
#
# Run separately for dorsal and ventral premotor, each against M1, as in the
# Methods.


# %%
import os

import pandas as pd
from statsmodels.stats.anova import AnovaRM

import config

betas = pd.read_csv(os.path.join(config.DATA, "rsa_betas.csv"))
betas["diff"] = betas["Rot/Ref"] - betas["Move"]
wide = betas.pivot(index="subject", columns="roi", values="diff")

print("=" * 70)
print("Context minus movement: 2 (region) x 2 (hemisphere) repeated-measures")
print("=" * 70)
print("DV = transformation-context coefficient - movement coefficient\n")

for premotor in ("PMd", "PMv"):
    long = pd.DataFrame([
        dict(subject=s, region=region, hemi=hemi, y=wide.loc[s, f"{hemi} {region}"])
        for s in wide.index
        for region in (premotor, "M1")
        for hemi in ("L", "R")
    ]).dropna()

    table = AnovaRM(long, "y", "subject", within=["region", "hemi"]).fit()
    a = table.anova_table

    print(f"--- {premotor} vs M1  (n = {long.subject.nunique()}) ---")
    for term, label in (("region", "region (= the difference-of-differences)"),
                        ("hemi", "hemisphere"),
                        ("region:hemi", "region x hemisphere")):
        f, p = a.loc[term, "F Value"], a.loc[term, "Pr > F"]
        df1, df2 = int(a.loc[term, "Num DF"]), int(a.loc[term, "Den DF"])
        print(f"  {label:<42} F({df1}, {df2}) = {f:5.2f}, {config.fmt_p(p)}")
    print()
