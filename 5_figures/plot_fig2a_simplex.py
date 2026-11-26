#!/usr/bin/env python3
"""Fig 2a decoder-norm simplex on rebuild95 — styled to match the paper draft exactly.
Vertices: ProtBert (top, blue), IgBert-unpaired (bottom-left, orange), IgBert (bottom-right, red).
Points colored by barycentric blend of the sum-normalised (ProtBert,IgBert-unp,IgBert-paired) rel norms.
Renders the 3-panel Fig 2a AND each panel as a standalone file. Agg backend (cluster has no cairo).
"""

import numpy as np, os
import matplotlib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = f"{BASE}/results"
tab = plt.get_cmap("tab10").colors
CPROT = np.array(tab[0])
CUNP = np.array(tab[1])
CPAIR = np.array(tab[3])  # blue / orange / red


def bary_to_xy(r):
    V = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0]])
    return r @ V


def color(r):
    return np.clip(r[:, 0:1] * CPROT + r[:, 1:2] * CUNP + r[:, 2:3] * CPAIR, 0, 1)


def draw_panel(ax, reg, lab, rng, letter=None):
    z = np.load(D + "/decnorm_" + reg + "_small_s0_L20.npz")
    rel = z["rel"]
    freq = z["val_act_freq"]
    active = np.nonzero(freq > 1e-4)[0]
    n_active = active.size
    if active.size > 8000:
        active = rng.choice(active, 8000, replace=False)
    r = rel[active]
    xy = bary_to_xy(r)
    cols = color(r)
    ax.scatter(xy[:, 0], xy[:, 1], s=3, c=cols, alpha=0.6, edgecolors="none", rasterized=True)
    V = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    ax.plot(V[:, 0], V[:, 1], color="0.15", lw=1.2)
    verts = [(0.5, np.sqrt(3) / 2), (0, 0), (1, 0)]
    mids = [(0.5, 0.0), (0.75, np.sqrt(3) / 4), (0.25, np.sqrt(3) / 4)]
    for v, mid in zip(verts, mids):
        ax.plot([v[0], mid[0]], [v[1], mid[1]], ls="--", color="0.55", lw=0.7)
    ax.text(
        0.5,
        np.sqrt(3) / 2 + 0.05,
        "ProtBert",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=CPROT,
    )
    ax.text(
        -0.03,
        -0.04,
        "IgBert-unpaired",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=CUNP,
    )
    ax.text(
        1.03, -0.04, "IgBert", ha="right", va="top", fontsize=11, fontweight="bold", color=CPAIR
    )
    ax.set_title(f"{lab} (n = {n_active:,})", fontsize=12, pad=12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.14, 1.02)
    if letter:
        ax.text(
            -0.13, 1.0, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top"
        )
    return n_active


rng = np.random.default_rng(0)
regs = ["unpaired", "paired", "mixed"]
labels = ["unpaired", "paired", "mixed"]
# 3-panel composite
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
ns = {}
for i, (ax, reg, lab) in enumerate(zip(axes, regs, labels)):
    ns[reg] = draw_panel(ax, reg, lab, rng, letter="a" if i == 0 else None)
fig.tight_layout()
fig.savefig("fig2a_simplex_rebuild95.pdf", dpi=200)
fig.savefig("fig2a_simplex_rebuild95.png", dpi=200)
# standalone panels
for reg, lab in zip(regs, labels):
    f1, a1 = plt.subplots(figsize=(4.2, 4.2))
    draw_panel(a1, reg, lab, np.random.default_rng(0))
    f1.tight_layout()
    f1.savefig(f"fig2a_simplex_{reg}_rebuild95.pdf", dpi=200)
    f1.savefig(f"fig2a_simplex_{reg}_rebuild95.png", dpi=200)
    plt.close(f1)
print("saved fig2a composite + 3 standalone panels; n_active", ns)
