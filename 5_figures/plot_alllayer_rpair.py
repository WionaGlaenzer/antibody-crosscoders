#!/usr/bin/env python3
"""Reproducible plotter: all-layer 2-model relative paired-decoder-norm distribution,
smooth filled KDE curves colored by a layer gradient. Reuses the original twomodel_dist /
fig_normdist convention (r_pair = ||d_pair||/(||d_unp||+||d_pair||) over active latents;
balance line at 0.5; paired-specific region shaded >=0.6).

Input:  alllayer_decnorm_dist.npz  (from alllayer_decnorm_dist.py: one r_pair array per layer
        key 'L{N}', active latents of cc2_paired_alllayer_L{N}). figdata/fig_alllayer_rpair_*.csv
        are plain-text mirrors of the summary + plotted densities.
Output: fig_alllayer_rpair_gradient.{pdf,png}
"""

import numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde

z = np.load("alllayer_decnorm_dist.npz")
layers = sorted(int(k[1:]) for k in z.files)
cmap = plt.get_cmap("viridis")
norm = mpl.colors.Normalize(vmin=1, vmax=max(layers))
xs = np.linspace(0.30, 0.90, 400)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.axvspan(0.6, 1.0, color="#c1121f", alpha=0.06, zorder=0)
ax.axvline(0.5, color="#888", lw=0.8, ls=":", zorder=1)
for L in layers:
    d = gaussian_kde(z[f"L{L}"], bw_method=0.15)(xs)
    col = cmap(norm(L))
    ax.fill_between(xs, d, color=col, alpha=0.10, zorder=2, lw=0)
    ax.plot(xs, d, color=col, lw=1.2, alpha=0.9, zorder=3)
ax.set_xlim(0.30, 0.90)
ax.set_ylim(bottom=0)
ax.set_xlabel(
    r"relative paired-decoder norm  $r_{\mathrm{pair}}=\|d_{\mathrm{pair}}\|/(\|d_{\mathrm{unp}}\|+\|d_{\mathrm{pair}}\|)$"
)
ax.set_ylabel("density")
ax.set_title(
    "Relative paired-decoder-norm distribution is tight near balance at every layer", fontsize=7.5
)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.045)
cb.set_label("layer", fontsize=6.5)
cb.ax.tick_params(labelsize=5.5)
ax.annotate(
    "paired-specific\nwould live here",
    (0.73, ax.get_ylim()[1] * 0.55),
    ha="center",
    fontsize=6,
    color="#c1121f",
)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(f"fig_alllayer_rpair_gradient.{e}", dpi=200)
print("wrote fig_alllayer_rpair_gradient.{pdf,png} for layers", layers)


def scatter_rpair_vs_layer(npz="alllayer_decnorm_dist.npz", out="fig_alllayer_rpair_scatter"):
    """r_pair vs layer: jittered active-latent point cloud per layer + per-layer median.
    Scatter is rasterized so the vector PDF stays small."""
    import numpy as np, matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(npz)
    layers = sorted(int(k[1:]) for k in z.files)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.axhspan(0.6, 1.0, color="#c1121f", alpha=0.06, zorder=0)
    ax.axhline(0.5, color="#888", lw=0.8, ls=":", zorder=1)
    allx, ally, meds = [], [], []
    for L in layers:
        v = z[f"L{L}"]
        allx.append(L + rng.uniform(-0.28, 0.28, size=len(v)))
        ally.append(v)
        meds.append(np.median(v))
    sc = ax.scatter(
        np.concatenate(allx),
        np.concatenate(ally),
        s=1.4,
        color="0.55",
        alpha=0.10,
        edgecolors="none",
        zorder=2,
    )
    sc.set_rasterized(True)
    ax.plot(layers, meds, "-", color="#1f77b4", lw=1.3, zorder=4)
    ax.scatter(
        layers,
        meds,
        s=16,
        color="#1f77b4",
        edgecolors="white",
        lw=0.4,
        zorder=5,
        label="per-layer median",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel(r"relative paired-decoder norm  $r_{\mathrm{pair}}$")
    ax.set_ylim(0.30, 0.90)
    ax.set_xlim(0.3, max(layers) + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20, 25, layers[-1]])
    ax.set_title("Relative paired-decoder norm stays near balance across all layers", fontsize=7.5)
    ax.annotate(
        "paired-specific region (empty)",
        (np.mean(layers), 0.75),
        ha="center",
        fontsize=6,
        color="#c1121f",
    )
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=200)
    fig.savefig(f"{out}.pdf", dpi=150)
    print("wrote", out, "for layers", layers)


if __name__ == "__main__":
    scatter_rpair_vs_layer()
