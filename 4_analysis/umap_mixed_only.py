#!/usr/bin/env python3
"""Mixed-model decoder UMAP only (paired already done). Multithreaded (no random_state) for speed."""

import numpy as np, json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

RES = f"{BASE}/results"
import umap
from sklearn.cluster import HDBSCAN

W = np.load(f"{RES}/decoder_mixed.npz")["W"]  # (3,32768,1024)
D = W[0]  # ProtBert decoder (general-protein readout)
ids = np.array(sorted(json.load(open(f"{RES}/mixed_active.json"))))
X = D[ids]
Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean").fit_transform(
    Xn
)  # no seed -> parallel
clus = HDBSCAN(min_cluster_size=15, min_samples=5).fit_predict(Xn)
json.dump(
    {
        "latent_id": ids.tolist(),
        "x": emb[:, 0].tolist(),
        "y": emb[:, 1].tolist(),
        "cluster": clus.tolist(),
        "model_idx": 0,
        "n": len(ids),
    },
    open(f"{RES}/umap_mixed.json", "w"),
)
print("mixed embedded", len(ids), "n_clusters", len(set(clus)) - (1 if -1 in clus else 0))
