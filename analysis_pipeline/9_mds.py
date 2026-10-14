#!/usr/bin/env python3
# Classical MDS coordinates, Figs 4a-b and S4, with scikit-learn's ClassicalMDS as in the
# Methods. ClassicalMDS needs scikit-learn >= 1.8, which needs Python >= 3.11 (the Methods'
# separate Python version for MDS). Solutions are only defined up to rotation and reflection.


# %%
import os

import numpy as np
from sklearn.manifold import ClassicalMDS

import config

N_COMPONENTS = 2

print("Classical MDS on group-mean similarity matrices: Figs 4a-b, S4")
print("dissimilarity = 1 - r, diagonal set to 0; coordinates are arbitrary")
print("up to rotation and reflection\n")

for roi in config.ALL_ROIS:
    paths = [os.path.join(config.DATA, "rsm", f"{s}_{roi.replace(' ', '')}.npy")
             for s in config.SUBJECTS]
    mats = [np.load(p) for p in paths if os.path.exists(p)]
    group = np.mean(mats, axis=0)

    # symmetrize (a no-op on this data)
    group = 0.5 * (group + group.T)

    rdm = 1.0 - group
    np.fill_diagonal(rdm, 0.0)

    xy = ClassicalMDS(n_components=N_COMPONENTS, metric="precomputed").fit_transform(rdm)

    print(f"--- {roi}  (n = {len(mats)}) ---")
    for cond, (x, y) in zip(config.CONDITIONS, xy):
        print(f"    {cond:<5} {x:+.4f}  {y:+.4f}")
    print()
