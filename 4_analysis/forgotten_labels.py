#!/usr/bin/env python3
"""Label the top 'forgotten' (base-only) latents: for each, find its top-activating
general-protein residues in the general_single_L20 store, and record the +-6 AA sequence
context window (one-letter) so we can read what motif each detects. Writes forgotten_labels.json."""

import json, time, numpy as np, h5py, torch as th

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
import sys

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
import os

os.environ["HF_HOME"] = f"{BASE}/caches/hf"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_mixed_L20.pt", device=dev, dtype=th.float32)
ae.eval()
FEATS = [5784, 18456, 25396, 12518, 11982, 25170, 24093, 29067, 15834, 8100, 6367, 7911]
FT = th.tensor(FEATS, device=dev)
# general protein sequences: general parquet gives AA; store gives seq_id + position order
import pandas as pd

gp = (
    pd.read_parquet(f"{BASE}/data/general_protein_96k.parquet")
    if os.path.exists(f"{BASE}/data/general_protein_96k.parquet")
    else None
)
h5 = f"{ACTS}/general_single_L20.h5"
with h5py.File(h5, "r") as h:
    sid = h["seq_id"][:].astype(np.int64)
    N = sid.shape[0]
u, first, cnt = np.unique(sid, return_index=True, return_counts=True)
within = np.empty(N, dtype=np.int64)
for f0, c in zip(first, cnt):
    within[f0 : f0 + c] = np.arange(c)
# top-K residues per feature via fused topk over encode blocks
F = len(FEATS)
K = 40
best_val = np.full((F, K), -1e9)
best_gr = np.full((F, K), -1, dtype=np.int64)  # global residue idx
BLK = 200000
ENC = 16384
t0 = time.time()
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    for s in range(0, N, BLK):
        e = min(s + BLK, N)
        x = acts[s:e]
        vals = np.full((F, e - s), -1e9)
        for b0 in range(0, e - s, ENC):
            b1 = min(b0 + ENC, e - s)
            xb = th.tensor(x[b0:b1].astype(np.float32), device=dev)
            with th.no_grad():
                f = ae.encode(xb, use_threshold=True, normalize_activations=True)
            sub = f[:, FT].cpu().numpy().T  # (F, b)
            vals[:, b0:b1] = sub
        # merge block into running top-K
        for fi in range(F):
            allv = np.concatenate([best_val[fi], vals[fi]])
            allg = np.concatenate([best_gr[fi], np.arange(s, e)])
            o = np.argsort(-allv)[:K]
            best_val[fi] = allv[o]
            best_gr[fi] = allg[o]
        if (s // BLK) % 10 == 0:
            log(f"  {e}/{N} {time.time()-t0:.0f}s")
# map top residues -> seq_id + within-position -> AA context
seq_of = sid
out = {}
for fi, feat in enumerate(FEATS):
    recs = []
    for gr, val in zip(best_gr[fi], best_val[fi]):
        if gr < 0 or val <= 0:
            continue
        s_id = int(seq_of[gr])
        pos = int(within[gr])
        recs.append({"seq_id": s_id, "pos": pos, "act": float(val)})
    out[str(feat)] = {"top": recs}
json.dump(out, open(f"{RES}/forgotten_labels.json", "w"))
log("DONE")
print(
    "features:",
    FEATS,
    "| example top counts:",
    {k: len(v["top"]) for k, v in list(out.items())[:3]},
)
