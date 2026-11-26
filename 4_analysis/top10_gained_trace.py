#!/usr/bin/env python3
"""Per-residue traces of the top-10 most-gained latents along example heavy chain seq 167."""

import json, numpy as np, h5py, torch as th

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
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_paired_L20.pt", device=dev, dtype=th.float32)
ae.eval()
h5 = f"{ACTS}/full_paired_L20.h5"
TARGET = 167
with h5py.File(h5, "r") as h:
    split = h["split"][:]
    chain = h["chain"][:]
    region = h["region_label"][:]
    sid = h["seq_id"][:].astype(np.int64)
keep = (split == 2) & (chain == b"H") & (sid == TARGET)
gpos = np.nonzero(keep)[0]
reg = region[gpos].astype(int).tolist()
with h5py.File(h5, "r") as h:
    sub = h["acts"][gpos.min() : gpos.max() + 1]
x = th.tensor(sub[gpos - gpos.min()].astype(np.float32), device=dev)
with th.no_grad():
    fcode = ae.encode(x, use_threshold=True, normalize_activations=True)
cc = fcode.index_select(1, col_t).cpu().numpy()  # (L,10)
out = dict(
    seq_id=TARGET,
    length=len(reg),
    region=reg,
    latents=lats,
    gains=[t["gain"] for t in top],
    traces={str(lats[i]): cc[:, i].tolist() for i in range(len(lats))},
)
json.dump(out, open(f"{RES}/top10_gained_trace.json", "w"))
print("done; per-latent max:", {lats[i]: round(float(cc[:, i].max()), 3) for i in range(len(lats))})
