#!/usr/bin/env python3
# MDS coordinates — Figures 4a-b and S3.
#
# No statistic in the paper comes from these plots, so this script prints the
# coordinates and how much of the structure two dimensions capture, rather than
# drawing anything. The point is that the route from a correlation matrix to a
# 2-D picture involves choices the figures cannot show.
#
# This is CLASSICAL MDS (principal coordinates analysis), computed in closed
# form: double-centre the squared dissimilarities, B = -1/2 J D^2 J with
# J = I - 1/n, then take the eigenvectors of B with the largest eigenvalues,
# scaled by the square root of the eigenvalue.
#
# The Methods describe scikit-learn's ClassicalMDS (added in scikit-learn 1.8,
# which is why MDS ran in its own Python 3.11.5 environment). This closed-form
# NumPy version is numerically identical to it — coordinates agree to ~1e-14
# on every ROI once each axis's sign is aligned, and pairwise distances agree
# to ~1e-14 without any alignment — so it reproduces the same configuration
# without the dependency.
#
# Not to be confused with scikit-learn's plain MDS class, which is a different
# algorithm (SMACOF): iterative, randomly initialized, and liable to give a
# different configuration from one release to the next.
#
# One thing to know before comparing against the published figures: an MDS
# solution is only defined up to rotation and reflection. A configuration that
# looks mirrored or spun relative to Fig 4a has reproduced it, not contradicted
# it. The solutions are deliberately not rotated or aligned to each other.


# %%
import os

import numpy as np

import config

N_COMPONENTS = 2

print("=" * 70)
print("Classical MDS on group-mean similarity matrices — Figs 4a-b, S3")
print("=" * 70)
print("dissimilarity = 1 - r, diagonal set to 0; coordinates are arbitrary")
print("up to rotation and reflection\n")

for roi in config.ALL_ROIS:
    paths = [os.path.join(config.DATA, "rsm", f"{s}_{roi.replace(' ', '')}.npy")
             for s in config.SUBJECTS]
    mats = [np.load(p) for p in paths if os.path.exists(p)]
    group = np.mean(mats, axis=0)

    # Symmetrize. The stored matrices are already exactly symmetric, so on this
    # data this is a no-op; it is kept because classical MDS requires symmetry
    # and nothing upstream guarantees it survives a round-trip through another
    # format.
    group = 0.5 * (group + group.T)

    rdm = 1.0 - group
    np.fill_diagonal(rdm, 0.0)

    n = rdm.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (rdm ** 2) @ J

    eigvals, eigvecs = np.linalg.eigh(B)       # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Clipping guards against the trailing eigenvalues going slightly negative,
    # which classical MDS allows when the dissimilarities are not exactly
    # Euclidean. On this data none of them do, so this never fires; it is the
    # standard form of the algorithm and is kept as written.
    L = np.clip(eigvals[:N_COMPONENTS], 0.0, None)
    xy = eigvecs[:, :N_COMPONENTS] * np.sqrt(L)

    # With classical MDS the eigenvalues ARE the variance along each axis, so
    # unlike an iterative solution this gives a real answer to "how much is the
    # 2-D plot showing" — over the positive eigenvalues, which are the part of
    # the structure that is Euclidean at all.
    pos = eigvals[eigvals > 0]
    captured = eigvals[:N_COMPONENTS].clip(0).sum() / pos.sum()

    print(f"--- {roi}  (n = {len(mats)}) ---")
    print(f"  variance captured by {N_COMPONENTS} dimensions: "
          f"{captured * 100:.1f}%")
    for cond, (x, y) in zip(config.CONDITIONS, xy):
        print(f"    {cond:<5} {x:+.4f}  {y:+.4f}")
    print()
