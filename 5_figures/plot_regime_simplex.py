#!/usr/bin/env python3
"""Build the 3-regime decoder-norm simplex figure cluster-side (avoids fetching
per-latent data over a congested login node). Reads decnorm npz for seed 0 of each
regime, plots barycentric simplex of the sum-normalised (ProtBERT,IgBERT-unp,IgBERT-paired)
relative decoder norms, coloured by barycentric mix. Writes PDF+PNG to the job cwd."""

import numpy as np, os
import matplotlib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = f"{BASE}/results"
# tab10 palette to match house scheme: ProtBERT blue(0), IgBERT-unp orange(1), IgBERT-paired red(3)
tab = plt.get_cmap("tab10").colors
CPROT = np.array(tab[0])
CUNP = np.array(tab[1])
CPAIR = np.array(tab[3])


def bary_to_xy(r):  # r: (n,3) sum-normalised -> 2D simplex coords
    # vertices: ProtBERT top, IgBERT-unp bottom-left, IgBERT-paired bottom-right
    V = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0]])
    return r @ V


def color(r):  # barycentric mix of vertex colors
    c = r[:, 0:1] * CPROT + r[:, 1:2] * CUNP + r[:, 2:3] * CPAIR
    return np.clip(c, 0, 1)


rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
labels = ["Unpaired", "Paired", "Mixed"]
regs = ["unpaired", "paired", "mixed"]
for ax, reg, lab in zip(axes, regs, labels):
    z = np.load(D + "/decnorm_" + reg + "_small_s0_L20.npz")
    rel = z["rel"]
    freq = z["val_act_freq"]
    active = np.nonzero(freq > 1e-4)[0]
    n_active = active.size  # TRUE active-latent count (before subsampling)
    # subsample dense cloud for file size (keep all if <8000); n in title is the true count
    if active.size > 8000:
        active = rng.choice(active, 8000, replace=False)
    r = rel[active]
    xy = bary_to_xy(r)
    cols = color(r)
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c=cols, alpha=0.55, edgecolors="none", rasterized=True)
    # triangle edges
    V = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    ax.plot(V[:, 0], V[:, 1], color="0.3", lw=0.8)
    # dashed medians (vertex -> opposite edge midpoint)
    mids = [((0 + 1) / 2, (0 + 0) / 2), (0.75, np.sqrt(3) / 4), (0.25, np.sqrt(3) / 4)]
    verts = [(0.5, np.sqrt(3) / 2), (0, 0), (1, 0)]
    for v, mid in zip(verts, mids):
        ax.plot([v[0], mid[0]], [v[1], mid[1]], ls="--", color="0.6", lw=0.5)
    ax.text(0.5, np.sqrt(3) / 2 + 0.04, "ProtBERT", ha="center", va="bottom", fontsize=7)
    ax.text(-0.02, -0.03, "IgBERT-unp", ha="right", va="top", fontsize=7)
    ax.text(1.02, -0.03, "IgBERT-paired", ha="left", va="top", fontsize=7)
    ax.set_title(f"{lab}  (n={n_active:,} active)", fontsize=8, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.12, 1.0)
fig.suptitle(
    "Decoder-norm simplex by training regime (rebuild95, 95%-NR, size-matched)  \u2014  up to 8,000 active latents shown per panel",
    fontsize=8,
    y=0.99,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("regime_simplex_rebuild95.pdf", dpi=150)
fig.savefig("regime_simplex_rebuild95.png", dpi=150)
print("saved regime_simplex_rebuild95 pdf+png")
for reg in regs:
    z = np.load(D + "/decnorm_" + reg + "_small_s0_L20.npz")
    print(reg, "active", int((z["val_act_freq"] > 1e-4).sum()))
