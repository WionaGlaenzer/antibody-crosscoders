#!/usr/bin/env python
"""Regenerate the feature-classification figures (single vs paired) from the
per-config classification CSVs written by the crosscoder classification step.

Inputs  (in RESULTS): classification_{single,paired}_L{6,12,20,28}.csv
        each has columns rel_prot, rel_unp, rel_pair, category, ...
Outputs (in OUTDIR):
    category_by_layer_both.{pdf,png}   line plot, % of active latents per category
    category_bars_both.{pdf,png}       grouped stacked bars (U vs P per layer)
    decoder_norm_dist.{pdf,png}        THRESHOLD-FREE rel_pair distributions

Usage:  python make_category_figures.py [--results DIR] [--outdir DIR]
"""

import argparse, os, numpy as np, pandas as pd
import matplotlib as mpl

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

LAYERS = [6, 12, 20, 28]
REGIMES = ["single", "paired"]


def load(results):
    """Return per-config category %s and the raw rel-norm rows."""
    cat_rows, norm_rows = [], []
    for reg in REGIMES:
        for L in LAYERS:
            f = os.path.join(results, f"classification_{reg}_L{L}.csv")
            d = pd.read_csv(f)
            n = len(d)
            vc = d["category"].value_counts()
            cat_rows.append(
                dict(
                    regime=reg,
                    layer=L,
                    n_active=n,
                    shared=100 * vc.get("shared-all", 0) / n,
                    gunp=100 * vc.get("gained-at-unpaired", 0) / n,
                    gpair=100 * vc.get("gained-at-paired", 0) / n,
                )
            )
            r = d[["rel_prot", "rel_unp", "rel_pair"]].copy()
            r["regime"], r["layer"] = reg, L
            norm_rows.append(r)
    return pd.DataFrame(cat_rows), pd.concat(norm_rows, ignore_index=True)


def style():
    tab = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.titlelocation": "left",
            "legend.fontsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    return tab


LEG = dict(
    frameon=True, fancybox=False, edgecolor="#666", framealpha=0.95, fontsize=7, borderpad=0.6
)


def hgrid(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#cccccc", lw=0.6, ls=(0, (4, 3)), zorder=0)


def fig_lines(cat, tab, outdir):
    C0, C1, C2 = tab[0], tab[1], tab[2]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3))
    for ax, reg, lab in [
        (axes[0], "single", "Single-chain (unpaired) regime"),
        (axes[1], "paired", "Paired regime"),
    ]:
        sub = cat[cat.regime == reg].sort_values("layer")
        x = np.arange(len(sub))
        hgrid(ax)
        for key, col, name in [
            ("shared", C0, "shared-all"),
            ("gunp", C1, "gained-at-unpaired"),
            ("gpair", C2, "gained-at-paired"),
        ]:
            ax.plot(x, sub[key], "-o", color=col, lw=2, ms=6, label=name, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in sub.layer])
        ax.set_xlabel("Layer")
        ax.set_ylim(-3, 90)
        ax.set_ylabel("% of active latents")
        ax.set_yticks(range(0, 91, 10))
        ax.set_title(lab)
    axes[1].legend(loc="upper right", **LEG)
    fig.suptitle(
        "Feature classification across layers — gained-at-paired ≈ 0 in both regimes",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(
            os.path.join(outdir, f"category_by_layer_both.{e}"), dpi=200, bbox_inches="tight"
        )
    plt.close(fig)


def fig_bars(cat, tab, outdir):
    C0, C1, C2 = tab[0], tab[1], tab[2]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    hgrid(ax)
    x = np.arange(len(LAYERS))
    w = 0.36
    for reg, off, hatch in [("single", -w / 2 - 0.02, None), ("paired", +w / 2 + 0.02, "//////")]:
        for li, L in enumerate(LAYERS):
            r = cat[(cat.regime == reg) & (cat.layer == L)].iloc[0]
            b = 0
            for val, col in [(r.shared, C0), (r.gunp, C1), (r.gpair, C2)]:
                ax.bar(
                    x[li] + off,
                    val,
                    w,
                    bottom=b,
                    color=col,
                    edgecolor="white",
                    lw=0.6,
                    hatch=hatch,
                    zorder=3,
                )
                b += val
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in LAYERS])
    ax.set_xlabel("Layer")
    ax.set_ylabel("% of active latents")
    ax.set_ylim(0, 105)
    for li in range(len(LAYERS)):
        ax.text(x[li] - w / 2 - 0.02, -4, "U", ha="center", va="top", fontsize=6.5, color="#555")
        ax.text(x[li] + w / 2 + 0.02, -4, "P", ha="center", va="top", fontsize=6.5, color="#555")
    cat_leg = [
        Patch(fc=C0, label="shared-all"),
        Patch(fc=C1, label="gained-at-unpaired"),
        Patch(fc=C2, label="gained-at-paired"),
    ]
    reg_leg = [
        Patch(fc="#ccc", label="U = unpaired"),
        Patch(fc="#ccc", hatch="//////", label="P = paired"),
    ]
    l1 = ax.legend(handles=cat_leg, loc="upper left", title="category", **LEG)
    ax.add_artist(l1)
    ax.legend(handles=reg_leg, loc="upper right", **LEG)
    fig.suptitle(
        "Feature classification by layer and regime — gained-at-paired ≈ 0 everywhere",
        y=1.0,
        fontsize=9.5,
    )
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"category_bars_both.{e}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_normdist(norms, tab, outdir):
    Csingle, Cpaired = tab[0], tab[3]
    bins = np.linspace(0, 1, 61)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, L in zip(axes, LAYERS):
        hgrid(ax)
        for reg, col in [("single", Csingle), ("paired", Cpaired)]:
            v = norms[(norms.regime == reg) & (norms.layer == L)]["rel_pair"].values
            ax.hist(
                v, bins=bins, density=True, histtype="step", color=col, lw=1.8, label=reg, zorder=3
            )
        ax.axvspan(0.6, 1.0, color="#c1121f", alpha=0.07, zorder=0)
        ax.axvline(1 / 3, color="#888", lw=0.7, ls=":", zorder=1)
        ax.set_title(f"L{L}")
        ax.set_xlabel("relative paired-decoder norm $r_{pair}$")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper right", **LEG)
    axes[3].annotate(
        "paired-specific\nwould live here",
        (0.8, axes[3].get_ylim()[1] * 0.5),
        ha="center",
        fontsize=6.5,
        color="#c1121f",
    )
    fig.suptitle(
        "Relative paired-decoder-norm distribution is tight near 1/3 — no mass in the paired-specific region (threshold-free)",
        y=1.03,
        fontsize=10,
    )
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"decoder_norm_dist.{e}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{BASE}/results")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    tab = style()
    cat, norms = load(a.results)
    fig_lines(cat, tab, a.outdir)
    fig_bars(cat, tab, a.outdir)
    fig_normdist(norms, tab, a.outdir)
    print("wrote 3 figures (x pdf+png) to", a.outdir)
    print(cat.to_string(index=False))
