#!/usr/bin/env python3
"""Reproducible plotter: crosscoder VE + fraction-of-active-latents by training regime,
rebuild95, 3-seed mean +/- s.d. Reuses the original regime_comparison layout/palette.
Reads figdata/regime_comparison_rebuild95.csv (regime, seed, n_active, frac_active_pct,
ve_prot_bert, ve_igbert_unpaired, ve_igbert_paired). Writes regime_comparison_rebuild95.{pdf,png}.
"""

import csv, numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DICT = 32768
regimes = ["unpaired", "paired", "mixed"]
rows = list(csv.DictReader(open("figdata/regime_comparison_rebuild95.csv")))


def agg(reg, key):
    return np.array([float(r[key]) for r in rows if r["regime"] == reg])


VEm = {
    r: [
        agg(r, "ve_prot_bert").mean(),
        agg(r, "ve_igbert_unpaired").mean(),
        agg(r, "ve_igbert_paired").mean(),
    ]
    for r in regimes
}
VEs = {
    r: [
        agg(r, "ve_prot_bert").std(ddof=1),
        agg(r, "ve_igbert_unpaired").std(ddof=1),
        agg(r, "ve_igbert_paired").std(ddof=1),
    ]
    for r in regimes
}
NAm = {r: agg(r, "n_active").mean() for r in regimes}
NAs = {r: agg(r, "n_active").std(ddof=1) for r in regimes}
C = ["#5E5E5E", "#0076BA", "#00AB8E"]
names = ["ProtBert", "IgBert-unpaired", "IgBert"]
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(3)
w = 0.26
axA.set_axisbelow(True)
axB.set_axisbelow(True)  # gridlines behind bars
for mi in range(3):
    axA.bar(
        x + (mi - 1) * w,
        [VEm[r][mi] for r in regimes],
        w,
        yerr=[VEs[r][mi] for r in regimes],
        capsize=2,
        color=C[mi],
        label=names[mi],
        error_kw=dict(lw=0.7, capthick=0.7),
        zorder=3,
    )
axA.set_xticks(x)
axA.set_xticklabels(regimes)
axA.set_xlabel("training regime")
axA.set_ylabel("Variance explained (val)")
axA.set_ylim(0.75, 1.0)
axA.legend(loc="lower left", fontsize=6)
axA.grid(axis="y", ls="--", alpha=0.35)
axA.text(-0.13, 1.02, "a", transform=axA.transAxes, fontsize=10, fontweight="bold")
fm = [NAm[r] / DICT * 100 for r in regimes]
fs = [NAs[r] / DICT * 100 for r in regimes]
axB.bar(
    x, fm, 0.55, yerr=fs, capsize=3, color="#3B6EA5", error_kw=dict(lw=0.8, capthick=0.8), zorder=3
)
for i, r in enumerate(regimes):
    axB.annotate(
        f"{fm[i]:.0f}%\n({NAm[r]:,.0f})", (i, fm[i] + fs[i]), ha="center", va="bottom", fontsize=6.3
    )
axB.set_xticks(x)
axB.set_xticklabels(regimes)
axB.set_xlabel("training regime")
axB.set_ylabel("Active latents (% of dictionary)")
axB.set_ylim(0, 100)
axB.grid(axis="y", ls="--", alpha=0.35)
axB.text(-0.13, 1.02, "b", transform=axB.transAxes, fontsize=10, fontweight="bold")
fig.suptitle(
    "Crosscoder reconstruction and dictionary usage by regime (layer 20, dict 32,768, k=32; rebuild95, 3-seed mean\u00b1s.d.)",
    fontsize=8,
    y=1.0,
)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(f"regime_comparison_rebuild95.{e}", dpi=200)
print("wrote regime_comparison_rebuild95.{pdf,png}")
