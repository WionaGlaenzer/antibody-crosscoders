#!/usr/bin/env python3
"""Geometric separation of antibody-specific vs general-protein latent decoder directions
in the mixed crosscoder. Uses ProtBert readout (idx0, matches umap_mixed). Computes, for
category groups (gained-at-unpaired=antibody, base-only=general-only, shared-all) and a
firing-based split (log2 ab/gen): centroid angle, within-group coherence, effective rank
(participation ratio), and principal angles between top-PC subspaces. Writes decoder_geometry.json.
"""

import json, numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

RES = f"{BASE}/results"
W = np.load(f"{RES}/decoder_mixed.npz")["W"]  # (3, 32768, 1024)
D = W[0]  # ProtBert readout
lab = json.load(open(f"{RES}/geom_labels.json"))
import collections

cat = {r["latent_id"]: r["category"] for r in lab}
lg = {r["latent_id"]: r["log2_ab_over_gen"] for r in lab}


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def group(ids):
    X = unit(D[np.array(sorted(ids))])
    return X


def eff_rank(X):
    # participation ratio of singular values of the (centered) matrix
    s = np.linalg.svd(X - X.mean(0, keepdims=True), full_matrices=False, compute_uv=False)
    l = s**2
    return float((l.sum() ** 2) / (l**2).sum())


def within_coh(X):
    # mean pairwise cosine (subsample if large)
    if len(X) > 2000:
        idx = np.random.RandomState(0).choice(len(X), 2000, replace=False)
        X = X[idx]
    G = X @ X.T
    n = len(X)
    return float((G.sum() - np.trace(G)) / (n * (n - 1)))


def centroid(X):
    c = X.mean(0)
    return c / (np.linalg.norm(c) + 1e-9)


def ang(u, v):
    import numpy as np

    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def principal_angles(A, B, k=10):
    # top-k PC subspaces of each group, principal angles between them
    def pcs(X, k):
        Xc = X - X.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Vt[:k]

    PA = pcs(A, k)
    PB = pcs(B, k)
    s = np.linalg.svd(PA @ PB.T, compute_uv=False)
    return [float(np.degrees(np.arccos(np.clip(x, -1, 1)))) for x in s]


out = {}
# ---- category groups ----
ab = [i for i, c in cat.items() if c == "gained-at-unpaired"]
gen = [i for i, c in cat.items() if c == "base-only"]
sh = [i for i, c in cat.items() if c == "shared-all"]
Xab, Xgen, Xsh = group(ab), group(gen), group(sh)
cab, cgen, csh = centroid(Xab), centroid(Xgen), centroid(Xsh)
out["category"] = {
    "n": {"antibody(gained-unp)": len(ab), "general(base-only)": len(gen), "shared": len(sh)},
    "centroid_angle_deg": {
        "antibody_vs_general": ang(cab, cgen),
        "antibody_vs_shared": ang(cab, csh),
        "general_vs_shared": ang(cgen, csh),
    },
    "within_coherence_meancos": {
        "antibody": within_coh(Xab),
        "general": within_coh(Xgen),
        "shared": within_coh(Xsh),
    },
    "effective_rank": {
        "antibody": eff_rank(Xab),
        "general": eff_rank(Xgen),
        "shared": eff_rank(Xsh),
        "ambient": 1024,
    },
    "principal_angles_ab_vs_gen_top10": principal_angles(Xab, Xgen, 10),
}
# ---- firing-based groups ----
abf = [i for i, v in lg.items() if v > 2]
genf = [i for i, v in lg.items() if v < -2]
Xabf, Xgenf = group(abf), group(genf)
out["firing"] = {
    "n": {"antibody(log2>2)": len(abf), "general(log2<-2)": len(genf)},
    "centroid_angle_deg": ang(centroid(Xabf), centroid(Xgenf)),
    "within_coherence_meancos": {"antibody": within_coh(Xabf), "general": within_coh(Xgenf)},
    "effective_rank": {"antibody": eff_rank(Xabf), "general": eff_rank(Xgenf)},
    "principal_angles_top10": principal_angles(Xabf, Xgenf, 10),
}
json.dump(out, open(f"{RES}/decoder_geometry.json", "w"))
print(json.dumps(out, indent=1))
