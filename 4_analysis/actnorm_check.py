#!/usr/bin/env python3
"""Check per-model raw activation norms at layer 20 (ProtBert vs IgBert-unpaired vs IgBert-paired)
and confirm the crosscoder normalization. Minder et al.: larger activation norms in the fine-tuned
model shift the decoder-norm (Delta_norm) distribution rightward -> must normalize per model.
Reports: (a) per-model raw L2 norm distribution of layer-20 residue activations,
(b) the stored per-model train_mean/train_std used for normalization,
(c) per-model L2 norm AFTER normalization (should be comparable across models)."""

import json, time, numpy as np, h5py, torch as th

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


import sys

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_mixed_L20.pt", device=dev, dtype=th.float32)
ae.eval()
MODELS = ["ProtBert", "IgBert-unpaired", "IgBert-paired"]
h5 = f"{ACTS}/mixed_single_L20.h5"
with h5py.File(h5, "r") as h:
    print("keys:", list(h.keys()))
    acts = h["acts"]
    N = acts.shape[0]
    shp = acts.shape
    log(f"acts shape {shp}")
    tm = h["train_mean"][:] if "train_mean" in h else None
    ts = h["train_std"][:] if "train_std" in h else None
    # sample 200k residues spread across the file
    step = max(1, N // 200000)
    sel = np.arange(0, N, step)[:200000]
    # sequential blocks to avoid slow fancy index
    raw_norm = [[] for _ in range(3)]
    norm_norm = [[] for _ in range(3)]
    BLK = 100000
    for s in range(0, N, BLK):
        e = min(s + BLK, N)
        blk_sel = sel[(sel >= s) & (sel < e)] - s
        if len(blk_sel) == 0:
            continue
        x = acts[s:e][blk_sel]  # (b,3,1024) or (b,1024)?
        xb = th.tensor(x.astype(np.float32), device=dev)
        if xb.ndim == 3:
            for mi in range(3):
                raw_norm[mi].append(xb[:, mi, :].norm(dim=-1).cpu().numpy())
            xn = ae.normalize_activations(xb, inplace=False)  # per-model normalize
            for mi in range(3):
                norm_norm[mi].append(xn[:, mi, :].norm(dim=-1).cpu().numpy())
        if len(raw_norm[0]) >= 40:
            break
out = {"acts_shape": list(shp), "models": MODELS}
if tm is not None:
    tm = np.asarray(tm)
    ts = np.asarray(ts)
    out["train_mean_shape"] = list(tm.shape)
    out["train_std_shape"] = list(ts.shape)
    # per-model mean-abs and typical std magnitude
    if tm.ndim == 2 and tm.shape[0] == 3:
        out["train_mean_L2_per_model"] = [float(np.linalg.norm(tm[i])) for i in range(3)]
        out["train_std_mean_per_model"] = [float(ts[i].mean()) for i in range(3)]
for mi in range(3):
    if raw_norm[mi]:
        r = np.concatenate(raw_norm[mi])
        nn = np.concatenate(norm_norm[mi])
        out.setdefault("raw_norm_median", {})[MODELS[mi]] = float(np.median(r))
        out.setdefault("raw_norm_mean", {})[MODELS[mi]] = float(r.mean())
        out.setdefault("normalized_norm_median", {})[MODELS[mi]] = float(np.median(nn))
        out.setdefault("normalized_norm_mean", {})[MODELS[mi]] = float(nn.mean())
json.dump(out, open(f"{RES}/actnorm_check.json", "w"), indent=1)
log("DONE")
print(json.dumps(out, indent=1))
