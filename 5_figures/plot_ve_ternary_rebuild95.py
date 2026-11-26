#!/usr/bin/env python3
"""Rebuild95 by-layer figures (3 regimes × layers 4/12/20/28), reusing paper_style.
 - ve_by_layer_rebuild95: VE vs layer, 3 regimes × 3 models, PER-PANEL y-axis,
   model palette gray/green/blue = ProtBERT/IgBERT-unpaired/IgBERT-paired (matches regime comparison).
 - ternary_grid_rebuild95: per-layer decoder-norm simplex, 3 regimes × 4 layers, MODEL-BLEND color
   (same orientation + palette as fig2a_simplex: ProtBERT top, IgBert-unpaired BL, IgBert BR).
Inputs: velayer_decnorm_all.npz (rel/cat/freq per regime×layer incl L20), ve_grid.json.
"""

import argparse, json, os, numpy as np, matplotlib

matplotlib.use("Agg")  # Agg honors per-artist rasterized=True -> compact vector-frame PDFs
import matplotlib.pyplot as plt
from paper_style import paper_style, hgrid, LEG

REG = ["unpaired", "paired", "mixed"]
LAY = [4, 12, 20, 28]


def model_cols(tab):  # gray, green, blue
    return ["#7f7f7f", tab[2], tab[0]]


def tern(rel):  # prot TOP, unp bottom-left, pair bottom-right (bary_to_xy, matches fig2a + labels)
    V = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0]])
    return (rel @ V)[:, 0], (rel @ V)[:, 1]


def blend(rel, CPROT, CUNP, CPAIR):
    return np.clip(rel[:, 0:1] * CPROT + rel[:, 1:2] * CUNP + rel[:, 2:3] * CPAIR, 0, 1)


def ve_fig(ve_grid, out, tab):
    mn = ["ProtBERT", "IgBERT-unpaired", "IgBERT-paired"]
    mc = model_cols(tab)
    mk = ["o", "s", "^"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))  # per-panel y-axis (no sharey)
    for ax, reg in zip(axes, REG):
        hgrid(ax)
        ys = []
        for mi in range(3):
            y = [ve_grid[reg][str(L)][mi] for L in LAY]
            ys += y
            ax.plot(LAY, y, marker=mk[mi], color=mc[mi], lw=1.8, ms=5, label=mn[mi], zorder=4)
        lo, hi = min(ys), max(ys)
        pad = max(0.005, (hi - lo) * 0.12)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(LAY)
        ax.set_xlabel("layer")
        ax.set_title(reg, fontweight="bold")
        ax.set_ylabel("Variance explained (val)")
    axes[2].legend(loc="best", **LEG)
    fig.suptitle(
        "Crosscoder variance explained by layer (rebuild95, 3 models × 3 regimes)",
        y=1.02,
        fontsize=9,
    )
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{out}/ve_by_layer_rebuild95.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def ternary_fig(Z, out, tab):
    CPROT = np.array(tab[0])
    CUNP = np.array(tab[1])
    CPAIR = np.array(tab[3])  # simplex palette blue/orange/red (fig2a-consistent)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.8))
    for i, rg in enumerate(REG):
        for j, L in enumerate(LAY):
            ax = axes[i, j]
            rel = Z[f"{rg}_L{L}_rel"]
            freq = Z[f"{rg}_L{L}_freq"]
            active = np.nonzero(freq > 1e-4)[0]
            if active.size > 8000:
                active = rng.choice(active, 8000, replace=False)
            r = rel[active]
            x, y = tern(r)
            cols = blend(r, CPROT, CUNP, CPAIR)
            V = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
            ax.plot(V[:, 0], V[:, 1], color="0.15", lw=1.0, zorder=1)
            for v, mid in zip(
                [(0.5, np.sqrt(3) / 2), (0, 0), (1, 0)],
                [(0.5, 0.0), (0.75, np.sqrt(3) / 4), (0.25, np.sqrt(3) / 4)],
            ):
                ax.plot([v[0], mid[0]], [v[1], mid[1]], ls="--", color="0.6", lw=0.6, zorder=2)
            ax.scatter(x, y, s=3, c=cols, alpha=0.55, edgecolors="none", rasterized=True, zorder=5)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_xlim(-0.15, 1.15)
            ax.set_ylim(-0.14, 1.02)
            if i == 0:
                ax.set_title(f"L{L}", fontsize=10, fontweight="bold")
            if j == 0:
                ax.text(
                    -0.10,
                    0.5,
                    rg,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=11,
                    fontweight="bold",
                )
            if i == 2:
                ax.text(
                    -0.02,
                    -0.03,
                    "IgB-unp",
                    ha="left",
                    va="top",
                    fontsize=5,
                    color=CUNP,
                    fontweight="bold",
                )
                ax.text(
                    1.02,
                    -0.03,
                    "IgB",
                    ha="right",
                    va="top",
                    fontsize=5,
                    color=CPAIR,
                    fontweight="bold",
                )
            if j == 0:
                ax.text(
                    0.5,
                    np.sqrt(3) / 2 + 0.02,
                    "ProtBERT",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    color=CPROT,
                    fontweight="bold",
                )
    fig.suptitle(
        "Relative decoder-norm triples by layer × regime (rebuild95; live latents, model-blend color; dict=32768, k=32)",
        fontsize=11,
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(f"{out}/ternary_grid_rebuild95.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}/ternary_grid_rebuild95.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tab = paper_style()
    ve_fig(json.load(open(f"{a.data}/ve_grid.json")), a.out, tab)
    ternary_fig(np.load(f"{a.data}/velayer_decnorm_all.npz"), a.out, tab)
    print("done")
