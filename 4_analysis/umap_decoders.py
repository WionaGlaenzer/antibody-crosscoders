#!/usr/bin/env python3
"""InterPLM-style UMAP of crosscoder decoder vectors. For each model figure, embed the
per-latent decoder vector (normalized) with UMAP + HDBSCAN cluster, restricted to ACTIVE latents.
paired -> IgBert-paired decoder (model idx 2); mixed -> ProtBert decoder (idx 0).
Returns only the 2D embedding + cluster id per latent (small). Writes umap_<tag>.json."""

import numpy as np, json, sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

RES = f"{BASE}/results"
import umap
from sklearn.cluster import HDBSCAN


def run(tag, model_idx, active_ids):
    W = np.load(f"{RES}/decoder_{tag}.npz")["W"]  # (3, 32768, 1024)
    D = W[model_idx]  # (32768, 1024)
    ids = np.array(sorted(active_ids))
    X = D[ids]
    # normalize each decoder vector to unit norm (InterPLM normalizes dictionary values)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0)
    emb = reducer.fit_transform(Xn)
    clus = HDBSCAN(min_cluster_size=15, min_samples=5).fit_predict(Xn)
    out = {
        "latent_id": ids.tolist(),
        "x": emb[:, 0].tolist(),
        "y": emb[:, 1].tolist(),
        "cluster": clus.tolist(),
        "model_idx": model_idx,
        "n": len(ids),
    }
    json.dump(out, open(f"{RES}/umap_{tag}.json", "w"))
    print(
        tag, "embedded", len(ids), "latents, n_clusters", len(set(clus)) - (1 if -1 in clus else 0)
    )
    return out


# active latents from the label tables pushed alongside
ab = json.load(open(f"{RES}/ab_active.json"))  # list of active latent ids (paired)
mx = json.load(open(f"{RES}/mixed_active.json"))  # list of active latent ids (mixed)
run("paired", 2, ab)
run("mixed", 0, mx)
