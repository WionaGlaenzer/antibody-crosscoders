#!/usr/bin/env python
"""Rebuild95 histogram/line figures in the ORIGINAL paper style.

Uses paper_style.py (framed LEG box legend, hgrid dashed y-grid, step-hist lw=1.8,
pdf.fonttype=42) so all rebuild95 density/line figures match make_category_figures.py.

Run:  python plot_rebuild95_restyled.py --data DATADIR --out OUTDIR
DATADIR must contain:
  mixed_firing_3seed_topk.npz, rpair_3arch.npz, r2_active.npz,
  regime_3seed.json, alllayer_decnorm_dist.npz, summary_L{n}.json (2-model alllayer)
"""

import argparse, os, json, numpy as np, matplotlib

matplotlib.use("cairo")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from paper_style import paper_style, hgrid, LEG


def fig2b(D, out, tab):
    z = np.load(f"{D}/mixed_firing_3seed_topk.npz")
    AB, GEN, ANY, CAT = z["ab_freq"], z["gen_freq"], z["any_freq"], z["cat"]
    eps = 1e-6
    CB = tab[0]
    CG = "0.6"
    CR = tab[3]
    groups = [("base-only", 1, CB), ("shared-all", 0, CG), ("gained-at-unpaired", 2, CR)]
    bins = np.linspace(-15, 15, 61)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    hgrid(ax)
    H = []
    for name, cc, col in groups:
        dens = []
        for s in range(3):
            m = (CAT[s] == cc) & (ANY[s] > 1e-4)
            d = np.log2((AB[s][m] + eps) / (GEN[s][m] + eps))
            d = d[np.isfinite(d)]
            h, _ = np.histogram(d, bins=bins, density=True)
            dens.append(h)
        dens = np.array(dens)
        mu = dens.mean(0)
        sd = dens.std(0)
        ax.fill_between(
            ctr,
            np.clip(mu - sd, 0, None),
            mu + sd,
            color=col,
            alpha=0.22,
            step="mid",
            lw=0,
            zorder=2,
        )
        ax.step(ctr, mu, where="mid", color=col, lw=1.8, zorder=3)
        H.append(Patch(facecolor="none", edgecolor=col, lw=1.8, label=name))
    ax.axvline(0, ls=(0, (4, 3)), color="#888", lw=0.7, zorder=1)
    ax.set_xlabel("log2(antibody firing / general firing)")
    ax.set_ylabel("Density")
    ax.set_xlim(-15, 15)
    ax.legend(handles=H, loc="upper right", **LEG)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{out}/fig2b_domain_density.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig2c(D, out, tab):
    z = np.load(f"{D}/mixed_firing_3seed_topk.npz")
    AB, GEN, ANY, CAT = z["ab_freq"], z["gen_freq"], z["any_freq"], z["cat"]
    FL = 0.0005
    RT = 32

    def pct(cc, s):
        m = (CAT[s] == cc) & (ANY[s] > 1e-4)
        n = m.sum()
        if n == 0:
            return 0.0, 0.0
        ab = AB[s][m]
        gen = GEN[s][m]
        return ((ab > FL) & (ab > RT * gen)).sum() / n * 100, (
            (gen > FL) & (gen > RT * ab)
        ).sum() / n * 100

    labels = ["base-only", "shared-all", "gained-unpaired"]
    codes = [1, 0, 2]
    AAm, AAs, GGm, GGs = [], [], [], []
    for cc in codes:
        aa = [pct(cc, s)[0] for s in range(3)]
        gg = [pct(cc, s)[1] for s in range(3)]
        AAm.append(np.mean(aa))
        AAs.append(np.std(aa))
        GGm.append(np.mean(gg))
        GGs.append(np.std(gg))
    AAm, AAs, GGm, GGs = map(np.array, (AAm, AAs, GGm, GGs))
    x = np.arange(3)
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    hgrid(ax)
    ax.bar(
        x - w / 2,
        GGm,
        w,
        yerr=GGs,
        color=tab[0],
        capsize=3,
        label="specific for other proteins",
        error_kw={"lw": 0.9},
        zorder=3,
    )
    ax.bar(
        x + w / 2,
        AAm,
        w,
        yerr=AAs,
        color=tab[3],
        capsize=3,
        label="antibody-specific",
        error_kw={"lw": 0.9},
        zorder=3,
    )
    for i in range(3):
        ax.text(i - w / 2, GGm[i] + GGs[i] + 2, f"{GGm[i]:.0f}", ha="center", fontsize=6.3)
        ax.text(i + w / 2, AAm[i] + AAs[i] + 2, f"{AAm[i]:.0f}", ha="center", fontsize=6.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage of latents")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper center", **LEG)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{out}/fig2c_specific_pct.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig2d(D, out, tab):
    Z = np.load(f"{D}/rpair_3arch.npz")
    r3 = Z["r3"]
    rd = Z["rd"]
    r2 = np.load(f"{D}/r2_active.npz")["r2"]
    series = [
        ("3-model crosscoder", r3, tab[0]),
        ("2-model crosscoder", r2, tab[1]),
        ("delta-crosscoder", rd, tab[3]),
    ]
    bins = np.linspace(0.40, 0.75, 71)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    hgrid(ax)
    H = []
    ymax = 0
    for name, v, col in series:
        h, _ = np.histogram(v, bins=bins, density=True)
        ax.step(ctr, h, where="mid", color=col, lw=1.8, zorder=3)
        H.append(Patch(facecolor="none", edgecolor=col, lw=1.8, label=name))
        ymax = max(ymax, h.max())
    ax.set_ylim(0, ymax * 1.08)
    ax.axvline(0.5, ls=(0, (4, 3)), color="#888", lw=0.8, zorder=1)
    ax.text(
        0.5,
        ax.get_ylim()[1] * 0.97,
        "balance (1/2)",
        ha="center",
        va="top",
        fontsize=6.5,
        color="#555",
    )
    ax.axvline(0.60, ls=":", color="#c1121f", lw=1.0, zorder=2)
    ax.set_xlim(0.40, 0.78)
    ax.annotate(
        "paired-specific\n(r$_{pair}$ > 0.6)",
        (0.615, ax.get_ylim()[1] * 0.5),
        ha="left",
        fontsize=6.5,
        color="#c1121f",
    )
    ax.set_xlabel(r"r$_{pair}$ = d$_{pair}$ / (d$_{unp}$ + d$_{pair}$)")
    ax.set_ylabel("Density")
    ax.legend(handles=H, loc="upper right", **LEG)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{out}/fig2d_rpair_density.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def regime_comp(D, out, tab):
    d = json.load(open(f"{D}/regime_3seed.json"))
    regimes = ["unpaired", "paired", "mixed"]
    DICT = 32768
    mn = ["ProtBERT", "IgBERT-unpaired", "IgBERT-paired"]
    mc = [tab[0], tab[1], tab[3]]
    VEm, VEs, NAm, NAs = {}, {}, {}, {}
    for reg in regimes:
        rows = d[reg]
        prot = np.array([r["ve_prot"] for r in rows])
        unp = np.array([r["ve_unp"] for r in rows])
        pair = np.array([r["ve_pair"] for r in rows])
        na = np.array([r["n_active"] for r in rows])
        VEm[reg] = [prot.mean(), unp.mean(), pair.mean()]
        VEs[reg] = [prot.std(ddof=1), unp.std(ddof=1), pair.std(ddof=1)]
        NAm[reg] = na.mean()
        NAs[reg] = na.std(ddof=1)
    x = np.arange(3)
    w = 0.26
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.8, 4.0))
    hgrid(axA)
    for mi in range(3):
        axA.bar(
            x + (mi - 1) * w,
            [VEm[r][mi] for r in regimes],
            w,
            yerr=[VEs[r][mi] for r in regimes],
            capsize=2,
            color=mc[mi],
            label=mn[mi],
            error_kw=dict(lw=0.7, capthick=0.7),
            zorder=3,
        )
    axA.set_xticks(x)
    axA.set_xticklabels(regimes)
    axA.set_xlabel("training regime")
    axA.set_ylabel("Variance explained (val)")
    axA.set_ylim(0.75, 1.0)
    axA.legend(loc="lower left", **LEG)
    axA.set_title("a")
    hgrid(axB)
    fm = [NAm[r] / DICT * 100 for r in regimes]
    fs = [NAs[r] / DICT * 100 for r in regimes]
    axB.bar(
        x,
        fm,
        0.55,
        yerr=fs,
        capsize=3,
        color="#3B6EA5",
        error_kw=dict(lw=0.8, capthick=0.8),
        zorder=3,
    )
    for i in range(3):
        axB.text(i, fm[i] + fs[i] + 2, f"{fm[i]:.0f}%", ha="center", fontsize=6.5)
    axB.set_xticks(x)
    axB.set_xticklabels(regimes)
    axB.set_xlabel("training regime")
    axB.set_ylabel("Active latents (% of dictionary)")
    axB.set_ylim(0, 100)
    axB.set_title("b")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{out}/regime_comparison_rebuild95.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def alllayer_ve(D, out, tab):
    sf = {}
    for f in os.listdir(D):
        if f.startswith("summary_L") and f.endswith(".json"):
            try:
                sf[int(f[len("summary_L") : -5])] = os.path.join(D, f)
            except ValueError:
                pass
    lays = sorted(sf)
    vu = []
    vp = []
    for L in lays:
        ve = json.load(open(sf[L]))["final_var_expl"]
        vu.append(ve["igbert_unpaired"])
        vp.append(ve["igbert_paired"])
    La = np.array(lays)
    vu = np.array(vu)
    vp = np.array(vp)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    hgrid(ax)
    for sel in [6, 12, 20, 28]:
        ax.axvline(sel, color="0.85", lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.plot(La, vu, "-o", color=tab[0], ms=3.2, lw=1.8, label="IgBERT-unpaired", zorder=4)
    ax.plot(La, vp, "-s", color=tab[2], ms=3.0, lw=1.8, label="IgBERT-paired", zorder=4)
    ax.set_xlabel("layer")
    ax.set_ylabel("variance explained (val)")
    ax.set_xlim(0.3, La.max() + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20, 25, int(La.max())])
    ax.set_title(
        "Crosscoder variance explained vs layer depth (2-model, rebuild95 paired-small)",
        fontsize=6.9,
    )
    ax.legend(loc="lower center", **LEG)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{out}/fig_alllayer_ve.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def alllayer_band(D, out, tab):
    z = np.load(f"{D}/alllayer_decnorm_dist.npz")
    layers = sorted(int(k[1:]) for k in z.files)
    La = np.array(layers)
    med = np.array([np.median(z[f"L{L}"]) for L in layers])
    p1 = np.array([np.percentile(z[f"L{L}"], 1) for L in layers])
    p99 = np.array([np.percentile(z[f"L{L}"], 99) for L in layers])
    p25 = np.array([np.percentile(z[f"L{L}"], 25) for L in layers])
    p75 = np.array([np.percentile(z[f"L{L}"], 75) for L in layers])
    mx = np.array([z[f"L{L}"].max() for L in layers])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axhspan(0.6, 1.0, color="#c1121f", alpha=0.06, zorder=0)
    ax.axhline(0.6, color="#c1121f", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0.5, color="#888", lw=0.8, ls=":", zorder=1)
    ax.fill_between(La, p1, p99, color=tab[0], alpha=0.15, zorder=2)
    ax.fill_between(La, p25, p75, color=tab[0], alpha=0.30, zorder=3)
    ax.plot(La, med, "-", color=tab[0], lw=1.8, zorder=5)
    ax.scatter(La, mx, s=16, marker="^", color=tab[3], edgecolors="white", lw=0.3, zorder=6)
    ax.set_xlabel("layer")
    ax.set_ylabel(r"relative paired-decoder norm  $r_{\mathrm{pair}}$")
    ax.set_ylim(0.30, 0.95)
    ax.set_xlim(0.3, La.max() + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20, 25, int(La.max())])
    ax.text(
        La.mean(),
        0.63,
        "paired-specific threshold (0.6)",
        ha="center",
        fontsize=5.8,
        color="#c1121f",
    )
    ax.set_title(
        "Paired-decoder-norm bulk stays near 1/2 at every layer; only rare tail latents exceed the 0.6 threshold",
        fontsize=6.6,
    )
    H = [
        Patch(fc=tab[0], alpha=0.15, label="1–99th percentile"),
        Patch(fc=tab[0], alpha=0.30, label="IQR (25–75th)"),
        Line2D([0], [0], color=tab[0], lw=1.8, label="median"),
        Line2D(
            [0], [0], marker="^", color=tab[3], lw=0, markeredgecolor="white", label="per-layer max"
        ),
    ]
    ax.legend(handles=H, loc="lower right", ncol=2, **LEG)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(f"{out}/fig_alllayer_rpair_band.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tab = paper_style()
    for fn in (fig2b, fig2c, fig2d, regime_comp, alllayer_ve, alllayer_band):
        try:
            fn(a.data, a.out, tab)
            print("ok", fn.__name__)
        except Exception as e:
            print("skip", fn.__name__, ":", str(e)[:80])
