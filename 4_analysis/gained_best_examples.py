#!/usr/bin/env python3
"""For each top-10 gained latent, find the heavy chain where it fires strongest,
and return that chain's per-residue trace + region labels."""

import json, time, numpy as np, h5py, torch as th

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
dev = "cuda" if th.cuda.is_available() else "cpu"
top = json.load(open(f"{RES}/top10_gained.json"))
lats = [t["latent_id"] for t in top]
col_t = th.tensor(lats, device=dev)
F = len(lats)
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_paired_L20.pt", device=dev, dtype=th.float32)
ae.eval()
h5 = f"{ACTS}/full_paired_L20.h5"
with h5py.File(h5, "r") as h:
    split = h["split"][:]
    chain = h["chain"][:]
    region = h["region_label"][:]
    sid = h["seq_id"][:].astype(np.int64)
    N = split.shape[0]
keep = (split == 2) & (chain == b"H")
idx = np.nonzero(keep)[0]
sid_h = sid[idx]
reg_h = region[idx].astype(int)
uniq, first, cnt = np.unique(sid_h, return_index=True, return_counts=True)
# only consider chains of reasonable length with all regions
chains = [
    (int(u), int(f0), int(c))
    for u, f0, c in zip(uniq, first, cnt)
    if 110 <= c <= 135 and len(set(reg_h[f0 : f0 + c].tolist())) == 7
]
# track per-latent best (max activation, seq_id, trace, region)
best_val = np.full(F, -1.0)
best = [None] * F
t0 = time.time()
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    for k, (u, f0, c) in enumerate(chains):
        gpos = idx[f0 : f0 + c]
        sub = acts[gpos.min() : gpos.max() + 1]
        x = th.tensor(sub[gpos - gpos.min()].astype(np.float32), device=dev)
        with th.no_grad():
            fc = ae.encode(x, use_threshold=True, normalize_activations=True)
        cc = fc.index_select(1, col_t).cpu().numpy()  # (c,F)
        mx = cc.max(0)
        for i in range(F):
            if mx[i] > best_val[i]:
                best_val[i] = mx[i]
                best[i] = dict(
                    seq_id=u, length=c, region=reg_h[f0 : f0 + c].tolist(), trace=cc[:, i].tolist()
                )
        if k % 400 == 0:
            log_t = time.time() - t0
            print(f"  {k}/{len(chains)} {log_t:.0f}s", flush=True)
out = {
    "latents": lats,
    "gains": [t["gain"] for t in top],
    "best": {str(lats[i]): best[i] for i in range(F)},
    "best_val": {str(lats[i]): float(best_val[i]) for i in range(F)},
}
json.dump(out, open(f"{RES}/gained_best_examples.json", "w"))
print("done; per-latent best max:", {lats[i]: round(float(best_val[i]), 3) for i in range(F)})
