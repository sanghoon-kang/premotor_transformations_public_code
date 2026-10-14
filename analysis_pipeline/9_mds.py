#!/usr/bin/env python3
# Classical MDS coordinates, Figs 4a-b and S3 (same as sklearn ClassicalMDS).
# Solutions are only defined up to rotation and reflection.


# %%
import os

import numpy as np

import config

N_COMPONENTS = 2

print("=" * 70)
print("Classical MDS on group-mean similarity matrices: Figs 4a-b, S3")
print("=" * 70)
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

    n = rdm.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (rdm ** 2) @ J

    eigvals, eigvecs = np.linalg.eigh(B)       # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # clip negative eigenvalues (never fires on this data)
    L = np.clip(eigvals[:N_COMPONENTS], 0.0, None)
    xy = eigvecs[:, :N_COMPONENTS] * np.sqrt(L)

    # variance captured, over the positive eigenvalues
    pos = eigvals[eigvals > 0]
    captured = eigvals[:N_COMPONENTS].clip(0).sum() / pos.sum()

    print(f"--- {roi}  (n = {len(mats)}) ---")
    print(f"  variance captured by {N_COMPONENTS} dimensions: "
          f"{captured * 100:.1f}%")
    for cond, (x, y) in zip(config.CONDITIONS, xy):
        print(f"    {cond:<5} {x:+.4f}  {y:+.4f}")
    print()
