#!/usr/bin/env python3
# Classical MDS of group-mean similarity matrices (Figs 4a-b and S4). Needs Python >= 3.11.
import os

import numpy as np
import pandas as pd
from sklearn.manifold import ClassicalMDS

import config

for roi in config.ALL_ROIS:
    group = np.mean([np.load(os.path.join(config.DATA, "rsm", f"{s}_{roi.replace(' ', '')}.npy"))
                     for s in config.SUBJECTS], axis=0)
    rdm = 1 - group
    np.fill_diagonal(rdm, 0)
    xy = ClassicalMDS(n_components=2, metric="precomputed").fit_transform(rdm)
    print(roi)
    print(pd.DataFrame(xy, index=config.CONDITIONS, columns=["dim1", "dim2"]), "\n")
