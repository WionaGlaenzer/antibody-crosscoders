#!/usr/bin/env python3
"""Decoder geometry on the REBUILT MIXED crosscoder (cc_mixed_small_s0_L20).
Separation of antibody-specific (gained-at-unpaired) latents vs the rest in the mixed
decoder space (ProtBert readout, idx0). Self-contained: loads the seed-0 checkpoint decoder
directly and category labels from classification_mixed_small_s0_L20.csv.
Reports: silhouette score (cosine) for gained-vs-rest, centroid angle, within-group coherence,
effective rank (participation ratio), and principal angles between top-PC subspaces.
Writes geometry_rebuild95.json to D/results/."""

import json, numpy as np, torch as th, csv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(
    f"{CKPT}/cc_mixed_small_s0_L20.pt", device="cpu", dtype=th.float32
)
ae.eval()
W = ae.decoder.weight.detach().cpu().numpy()  # (3, DICT, 1024)
D = W[0]  # ProtBert readout
cat = {}
for row in csv.DictReader(open(f"{RES}/classification_mixed_small_s0_L20.csv")):
    cat[int(row["latent_id"])] = row["category"]


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def group(ids):
    return unit(D[np.array(sorted(ids))])


def eff_rank(X):
    s = np.linalg.svd(X - X.mean(0, keepdims=True), full_matrices=False, compute_uv=False)
    l = s**2
    return float((l.sum() ** 2) / (l**2).sum())


def within_coh(X):
    if len(X) > 2000:
        idx = np.random.RandomState(0).choice(len(X), 2000, replace=False)
        X = X[idx]
    G = X @ X.T
    nn = len(X)
    return float((G.sum() - np.trace(G)) / (nn * (nn - 1)))


def centroid(X):
    cc = X.mean(0)
    return cc / (np.linalg.norm(cc) + 1e-9)


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def principal_angles(A, B, k=10):
    def pcs(X, k):
        Xc = X - X.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Vt[:k]

    PA = pcs(A, k)
    PB = pcs(B, k)
    s = np.linalg.svd(PA @ PB.T, compute_uv=False)
    return [float(np.degrees(np.arccos(np.clip(x, -1, 1)))) for x in s]


ab = [i for i, cc in cat.items() if cc == "gained-at-unpaired"]  # antibody-specific
rest = [
    i for i, cc in cat.items() if cc not in ("gained-at-unpaired",)
]  # the rest (all other active)
base = [i for i, cc in cat.items() if cc == "base-only"]
sh = [i for i, cc in cat.items() if cc == "shared-all"]
Xab = group(ab)
Xrest = group(rest)
Xbase = group(base) if base else None
Xsh = group(sh)
# ---- silhouette (cosine): gained-at-unpaired vs rest; subsample rest for O(n^2) tractability ----
from sklearn.metrics import silhouette_score

rng = np.random.RandomState(0)
rest_ids = np.array(sorted(rest))
rest_samp = (
    rest_ids if len(rest_ids) <= 3000 else rest_ids[rng.choice(len(rest_ids), 3000, replace=False)]
)
Xrest_s = group(rest_samp.tolist())
Xall = np.vstack([Xab, Xrest_s])
ylab = np.array([1] * len(Xab) + [0] * len(Xrest_s))
sil = float(silhouette_score(Xall, ylab, metric="cosine"))
cab, crest, csh = centroid(Xab), centroid(Xrest), centroid(Xsh)
out = {
    "checkpoint": "cc_mixed_small_s0_L20.pt",
    "readout": "prot_bert(idx0)",
    "n": {
        "gained_at_unpaired": len(ab),
        "rest": len(rest),
        "base_only": len(base),
        "shared_all": len(sh),
    },
    "silhouette_cosine_gained_vs_rest": sil,
    "silhouette_n_rest_sampled": int(len(Xrest_s)),
    "centroid_angle_deg": {
        "gained_vs_rest": ang(cab, crest),
        "gained_vs_shared": ang(cab, csh),
        "gained_vs_base": (ang(cab, centroid(Xbase)) if Xbase is not None else None),
    },
    "within_coherence_meancos": {
        "gained_at_unpaired": within_coh(Xab),
        "rest": within_coh(Xrest),
        "shared": within_coh(Xsh),
        "base_only": (within_coh(Xbase) if Xbase is not None else None),
    },
    "effective_rank": {
        "gained_at_unpaired": eff_rank(Xab),
        "rest": eff_rank(Xrest),
        "shared": eff_rank(Xsh),
        "ambient": 1024,
    },
    "principal_angles_gained_vs_rest_top10": principal_angles(Xab, Xrest_s, 10),
}
json.dump(out, open(f"{RES}/geometry_rebuild95.json", "w"), indent=1)
print(json.dumps(out, indent=1))
