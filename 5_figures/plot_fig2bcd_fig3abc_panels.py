#!/usr/bin/env python3
"""Reproducible local plotters for rebuild95 paper-draft panels.
Reads analysis npz in rb95/, writes individual panels to panels_rebuild95/.
Encode conventions (match original code):
  Fig 2b/2c (mixed_firing_3seed_topk.npz) -> use_threshold=False  (mixed_label.py)
  Fig 3a/b/c (fig3reuse_*, fig3c_region_firing.npz) -> use_threshold=True (concept_labeling.py)
  Fig 2d -> 3-WAY r_pair=d_pair/(d_prot+d_unp+d_pair), centred 1/3, paired-specific shaded 0.6-1.0
            (orig make_category_figures.fig_normdist)
Run in the `python` env after apply_figure_style().
"""

import numpy as np, os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl

try:
    apply_figure_style(grid=False, sizes=(8, 7, 6))  # if figure-style loaded
except NameError:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
tab = plt.rcParams["axes.prop_cycle"].by_key()["color"]
RB = "rb95"
OUT = "panels_rebuild95"
os.makedirs(OUT, exist_ok=True)
eps = 1e-6


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(fig)


# ---------- Fig 2b/2c : domain specificity (use_threshold=False) ----------
z = np.load(f"{RB}/mixed_firing_3seed_topk.npz")
AB, GEN, ANY, CAT = z["ab_freq"], z["gen_freq"], z["any_freq"], z["cat"]
CB, CG, CR = tab[0], "0.6", tab[3]
groups = [("base-only", 1, CB), ("shared-all", 0, CG), ("gained-at-unpaired", 2, CR)]
bins = np.linspace(-15, 15, 61)
ctr = 0.5 * (bins[:-1] + bins[1:])
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for nm, cc, col in groups:
    dens = [
        np.histogram(
            np.log2(
                (AB[s][(CAT[s] == cc) & (ANY[s] > 1e-4)] + eps)
                / (GEN[s][(CAT[s] == cc) & (ANY[s] > 1e-4)] + eps)
            ),
            bins=bins,
            density=True,
        )[0]
        for s in range(3)
    ]
    dens = np.array(dens)
    mu = dens.mean(0)
    sd = dens.std(0, ddof=1)
    ax.fill_between(
        ctr, np.clip(mu - sd, 0, None), mu + sd, color=col, alpha=0.25, step="mid", lw=0
    )
    ax.step(ctr, mu, where="mid", color=col, lw=1.7, label=nm)
ax.axvline(0, ls="--", color="0.3", lw=0.9)
ax.set_xlabel("log2(antibody firing / general firing)")
ax.set_ylabel("Density")
ax.set_title(
    "Domain specificity (mixed crosscoder, 3-seed mean ± s.d.)", fontsize=7, fontweight="bold"
)
ax.legend(frameon=False, fontsize=6)
ax.grid(axis="y", ls="--", alpha=0.35)
fig.tight_layout()
save(fig, "fig2b_domain_density")

FL, RT = 0.0005, 32


def pct(cc, s):
    m = (CAT[s] == cc) & (ANY[s] > 1e-4)
    t = int(m.sum())
    if not t:
        return 0.0, 0.0
    a, g = AB[s][m], GEN[s][m]
    return (
        100 * ((a >= FL) & (a >= RT * np.maximum(g, eps))).sum() / t,
        100 * ((g >= FL) & (g >= RT * np.maximum(a, eps))).sum() / t,
    )


labels = ["base-only", "shared-all", "gained-unpaired"]
codes = [1, 0, 2]
AAm = [np.mean([pct(cc, s)[0] for s in range(3)]) for cc in codes]
AAs = [np.std([pct(cc, s)[0] for s in range(3)], ddof=1) for cc in codes]
GGm = [np.mean([pct(cc, s)[1] for s in range(3)]) for cc in codes]
GGs = [np.std([pct(cc, s)[1] for s in range(3)], ddof=1) for cc in codes]
fig, ax = plt.subplots(figsize=(4.4, 3.4))
x = np.arange(3)
w = 0.38
ax.bar(
    x - w / 2,
    GGm,
    w,
    yerr=GGs,
    color=tab[0],
    capsize=3,
    label="specific for other proteins",
    error_kw={"lw": 0.9},
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
)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Percentage of latents (3-seed mean ± s.d.)")
ax.set_ylim(0, 105)
ax.set_title("Domain-specific fraction (0.05% freq, 32x ratio)", fontsize=7, fontweight="bold")
ax.legend(frameon=False, fontsize=6, loc="upper center")
ax.grid(axis="y", ls="--", alpha=0.35)
fig.tight_layout()
save(fig, "fig2c_specific_pct")

# ---------- Fig 2d : 3-way r_pair (fig_normdist convention) ----------
r3s = [
    np.load(f"{RB}/fig3reuse_s{s}.npz")["rel"][:, 2][
        np.load(f"{RB}/fig3reuse_s{s}.npz")["ab_freq"] > 1e-4
    ]
    for s in range(3)
]
bins = np.linspace(0, 1, 61)
ctr = 0.5 * (bins[:-1] + bins[1:])
fig, ax = plt.subplots(figsize=(4.8, 3.4))
d3 = np.array([np.histogram(r, bins=bins, density=True)[0] for r in r3s])
mu = d3.mean(0)
sd = d3.std(0, ddof=1)
ax.fill_between(ctr, np.clip(mu - sd, 0, None), mu + sd, color=tab[3], alpha=0.25, step="mid", lw=0)
ax.step(ctr, mu, where="mid", color=tab[3], lw=1.8, label="3-model crosscoder (3-seed mean ± s.d.)")
ax.axvspan(0.6, 1.0, color="#c1121f", alpha=0.07)
ax.axvline(1 / 3, color="#888", lw=0.8, ls=":")
ax.annotate(
    "paired-specific\nwould live here",
    (0.8, ax.get_ylim()[1] * 0.5),
    ha="center",
    fontsize=6.5,
    color="#c1121f",
)
ax.set_xlabel(r"relative paired-decoder norm $r_{pair}$")
ax.set_ylabel("Density")
ax.set_xlim(0, 1)
ax.set_title(
    "Paired-decoder-norm distribution: no mass in paired-specific region",
    fontsize=6.8,
    fontweight="bold",
)
ax.legend(frameon=False, fontsize=6, loc="upper right")
ax.grid(axis="y", ls="--", alpha=0.35)
fig.tight_layout()
save(fig, "fig2d_rpair_density")

# ---------- Fig 3a/b : region detectors (verified F1, use_threshold=True) ----------
S = [np.load(f"{RB}/fig3reuse_s{s}.npz", allow_pickle=True) for s in range(3)]
rn = [str(x) for x in S[0]["region_names"]]


def cnts(z):
    f1, fl, cat = z["f1max_wholetest"], z["region_floor_wholetest"], z["cat"]
    g = np.zeros(7, int)
    sh = np.zeros(7, int)
    for r in range(7):
        d = f1[:, r] > fl[r]
        g[r] = int((d & (cat == 2)).sum())
        sh[r] = int((d & (cat == 0)).sum())
    return g, sh


G = np.array([cnts(z)[0] for z in S])
Sh = np.array([cnts(z)[1] for z in S])
fig, ax = plt.subplots(figsize=(4.8, 3.4))
x = np.arange(7)
w = 0.4
ax.bar(
    x - w / 2,
    Sh.mean(0),
    w,
    yerr=Sh.std(0, ddof=1),
    color="0.55",
    capsize=2,
    label="shared",
    error_kw={"lw": 0.8},
)
ax.bar(
    x + w / 2,
    G.mean(0),
    w,
    yerr=G.std(0, ddof=1),
    color=tab[3],
    capsize=2,
    label="gained-at-unpaired",
    error_kw={"lw": 0.8},
)
ax.set_xticks(x)
ax.set_xticklabels(rn)
ax.set_ylabel("# detector latents (3-seed mean ± s.d.)")
ax.set_title("Region detectors by category", fontsize=7.5, fontweight="bold")
ax.legend(frameon=False, fontsize=6)
ax.grid(axis="y", ls="--", alpha=0.35)
fig.tight_layout()
save(fig, "fig3a_region_counts")
z = S[0]
f1, fl = z["f1max_wholetest"], z["region_floor_wholetest"]
fig, ax = plt.subplots(figsize=(5.0, 3.4))
rng = np.random.default_rng(0)
for r in range(7):
    top = np.sort(f1[:, r])[::-1][:100]
    xr = np.full(100, r) + rng.uniform(-0.28, 0.28, 100)
    ax.scatter(xr, top, s=4, color=tab[0], alpha=0.35, edgecolors="none")
    ax.plot([r - 0.4, r + 0.4], [fl[r], fl[r]], color=tab[3], lw=1.4)
ax.set_xticks(range(7))
ax.set_xticklabels(rn)
ax.set_ylabel("best-latent F1 (top 100 latents)")
ax.set_title("Region detector F1 distribution (top 100/region)", fontsize=7, fontweight="bold")
ax.plot([], [], color=tab[3], lw=1.4, label="F1 floor")
ax.legend(frameon=False, fontsize=6)
ax.set_ylim(0, 1)
ax.grid(axis="y", ls="--", alpha=0.35)
fig.tight_layout()
save(fig, "fig3b_region_f1_scatter")

# ---------- Fig 3c : region-detector firing ratio (mixed model, use_threshold=True) ----------
z = np.load(f"{RB}/fig3c_region_firing.npz", allow_pickle=True)
f1, fl, ab, gen = z["f1max"], z["floor"], z["ab_freq"], z["gen_freq"]
det = np.zeros(f1.shape[0], bool)
for r in range(7):
    det |= f1[:, r] > fl[r]
active = (ab + gen) > 1e-4
ratio = np.log2((ab + eps) / (gen + eps))
bins = np.linspace(-15, 15, 61)
ctr = 0.5 * (bins[:-1] + bins[1:])
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ha, _ = np.histogram(ratio[active], bins=bins, density=True)
ax.step(ctr, ha, where="mid", color="0.6", lw=1.4, label="all active latents")
ax.fill_between(ctr, 0, ha, step="mid", color="0.6", alpha=0.15, lw=0)
hd, _ = np.histogram(ratio[det & active], bins=bins, density=True)
ax.step(
    ctr,
    hd,
    where="mid",
    color=tab[3],
    lw=1.7,
    label=f"region detectors (n={int((det&active).sum())})",
)
ax.axvline(0, ls="--", color="0.3", lw=0.9)
ax.set_xlabel("log2(antibody firing / general firing)")
ax.set_ylabel("Density")
ax.set_title("Region detectors fire antibody-specifically", fontsize=7.5, fontweight="bold")
ax.legend(frameon=False, fontsize=6)
ax.grid(axis="y", ls="--", alpha=0.35)
fig.tight_layout()
save(fig, "fig3c_region_firing")
print("wrote fig2b,2c,2d,3a,3b,3c panels to", OUT)


# ---------- Fig 3d/e/f/g (run separately; require region_toplat_profile_rebuild95.json + umap_mixed_rebuild95.npz) ----------
def fig3defg():
    import json

    # 3d example trace
    d = json.load(open(f"{RB}/region_toplat_profile_rebuild95.json"))
    ex = d["example"]
    regions = ex["regions"]
    prof = ex["profiles"]
    L = ex["len"]
    pos = np.arange(L)
    detcol = {"CDR1": tab[1], "CDR2": tab[3], "CDR3": tab[2], "FR4": tab[0]}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    i = 0
    while i < L:
        j = i
        while j < L and regions[j] == regions[i]:
            j += 1
        if regions[i].startswith("CDR"):
            ax.axvspan(i - 0.5, j - 0.5, color="#ffcdd2", alpha=0.7, zorder=0)
        i = j
    for r in ["CDR1", "CDR2", "CDR3", "FR4"]:
        ax.plot(pos, prof[r], lw=1.3, color=detcol[r], label=f"{r} detector")
    ax.set_xlabel("residue position (heavy chain)")
    ax.set_ylabel("latent activation")
    ax.set_title(
        f"Region detectors localize to their regions (seq {ex['seq_id']})",
        fontsize=6.8,
        fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.set_xlim(-0.5, L - 0.5)
    ax.grid(axis="y", ls="--", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig3d_example_trace")
    # 3e/3f/3g from UMAP npz
    z = np.load(f"{RB}/umap_mixed_rebuild95.npz", allow_pickle=True)
    x, y = z["x"], z["y"]
    cat = z["cat"]
    ab = z["ab_freq"]
    gen = z["gen_freq"]
    catcol = {0: "0.75", 1: tab[0], 2: tab[3], 3: tab[2], 4: tab[4]}
    catname = {
        0: "shared-all",
        1: "base-only",
        2: "gained-at-unpaired",
        3: "gained-at-paired",
        4: "transient",
    }
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for cc in [5, 0, 4, 1, 2, 3]:
        m = cat == cc
        if m.sum():
            ax.scatter(
                x[m],
                y[m],
                s=2,
                color=catcol.get(cc, "0.9"),
                alpha=0.5,
                edgecolors="none",
                label=f"{catname[cc]} ({int(m.sum())})" if cc in catname else None,
                rasterized=True,
            )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Decoder UMAP by category (mixed)", fontsize=7.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=5.4, markerscale=3)
    fig.tight_layout()
    save(fig, "fig3e_umap_category")
    ratio = np.clip(np.log2((ab + eps) / (gen + eps)), -12, 12)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    sc = ax.scatter(
        x,
        y,
        s=2,
        c=ratio,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="none",
        vmin=-12,
        vmax=12,
        rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log2(ab / general firing)", fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Decoder UMAP by firing ratio (mixed)", fontsize=7.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3f_umap_firing")
    concepts = [str(c) for c in z["concepts"]]
    F = z["f1max_con_active"]
    flc = z["floor_concept"]
    det = [(i, c) for i, c in enumerate(concepts) if (F[:, i] > flc[i]).sum() > 0]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.scatter(x, y, s=2, color="0.85", alpha=0.45, edgecolors="none", rasterized=True, zorder=0)
    for k, (i, cn) in enumerate(det):
        m = F[:, i] > flc[i]
        ax.scatter(
            x[m],
            y[m],
            s=22,
            color=cmap(k % 10),
            edgecolors="k",
            lw=0.3,
            alpha=0.85,
            zorder=5,
            label=f"{cn} (n={int(m.sum())})",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Swiss-Prot concept detectors on decoder UMAP", fontsize=7, fontweight="bold")
    ax.legend(frameon=False, fontsize=5.6, markerscale=1.3)
    fig.tight_layout()
    save(fig, "fig3g_umap_concepts")
    print("wrote fig3d,3e,3f,3g")


if __name__ == "__main__":
    try:
        fig3defg()
    except Exception as e:
        print("fig3defg skipped:", e)
