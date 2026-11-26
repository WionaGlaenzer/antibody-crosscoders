#!/usr/bin/env python3
"""All-layer 2-model crosscoder overview (IgBERT unpaired vs paired, L1-30, rebuild95).
Two-panel Cairo figure:
  top   - variance explained vs layer, both models (gray/green/blue model palette convention).
  bottom- paired-specific latent count per layer as a STACKED bar: latent-scaling survivors
          (red) + flagged decoder-norm artifacts (grey). Latent scaling = Minder et al. 2504.02922
          diagnostic (nu_r/nu_e >= 0.5 -> artifact), computed in-training per layer.
Input: alllayer_overview.csv (layer, n_active, ve_igbert_unpaired, ve_igbert_paired,
       paired_only_raw, latent_scaling_survive, latent_scaling_flagged_artifact).
Usage: python plot_alllayer_overview.py --csv alllayer_overview.csv --out .
"""

import argparse, csv, numpy as np, matplotlib

matplotlib.use("cairo")
import matplotlib.pyplot as plt
from paper_style import paper_style, hgrid, LEG


def main(csv_path, out):
    L, na, veu, vep, raw, surv, flag = [], [], [], [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            L.append(int(row["layer"]))
            na.append(int(row["n_active"]))
            veu.append(float(row["ve_igbert_unpaired"]))
            vep.append(float(row["ve_igbert_paired"]))
            raw.append(int(row["paired_only_raw"]))
            surv.append(int(row["latent_scaling_survive"]))
            flag.append(int(row["latent_scaling_flagged_artifact"]))
    L = np.array(L)
    veu = np.array(veu)
    vep = np.array(vep)
    surv = np.array(surv)
    flag = np.array(flag)
    tab = paper_style()
    GREEN = tab[2]
    BLUE = tab[0]
    RED = tab[3]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.4, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.12)
    )
    hgrid(ax1)
    ax1.plot(L, veu, marker="o", ms=3, lw=1.5, color=GREEN, label="IgBERT-unpaired", zorder=4)
    ax1.plot(L, vep, marker="^", ms=3, lw=1.5, color=BLUE, label="IgBERT-paired", zorder=4)
    ax1.set_ylabel("Variance explained")
    ax1.set_ylim(0.95, 1.0)
    ax1.legend(loc="lower right", **LEG)
    ax1.set_title(
        "2-model crosscoder across all 30 layers (IgBERT unpaired vs paired)",
        fontsize=8,
        fontweight="bold",
    )
    hgrid(ax2)
    ax2.bar(L, surv, width=0.7, color=RED, label="survive latent scaling", zorder=3)
    ax2.bar(L, flag, width=0.7, bottom=surv, color="0.7", label="flagged artifact", zorder=3)
    ax2.set_ylabel("paired-\nspecific\nlatents", fontsize=6.5)
    ax2.set_xlabel("layer")
    ax2.set_ylim(0, max(6, (surv + flag).max() + 1))
    ax2.set_xticks(L)
    ax2.tick_params(axis="x", labelsize=5.5)
    ax2.legend(loc="upper right", ncol=2, **LEG)
    ax2.text(
        0.99,
        0.60,
        f"{int(surv.sum())} survive of {int((surv+flag).sum())} raw across all layers",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color="0.4",
    )
    for e in ("pdf", "png"):
        fig.savefig(f"{out}/fig_alllayer_overview.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="alllayer_overview.csv")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    main(a.csv, a.out)
    print("done")
