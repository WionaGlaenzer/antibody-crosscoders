#!/usr/bin/env python3
"""Per-residue traces of latents 1913 (V) and 6261 (J) along 10 example heavy chains."""

import json, numpy as np, h5py, torch as th

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
dev = "cuda" if th.cuda.is_available() else "cpu"
LAT_V = 1913
LAT_J = 6261
col_t = th.tensor([LAT_J, LAT_V], device=dev)  # order J,V
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
uniq, first, counts = np.unique(sid_h, return_index=True, return_counts=True)
# pick 10 example chains length 115-130 with all 7 regions
cand = []
for u, f0, c in zip(uniq, first, counts):
    if 115 <= c <= 130 and len(set(reg_h[f0 : f0 + c].tolist())) == 7:
        cand.append((int(u), int(f0), int(c)))
    if len(cand) >= 200:
        break
# encode candidates, keep those where BOTH latents fire, take first 10
out = []
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    for u, f0, c in cand:
        gpos = idx[f0 : f0 + c]
        sub = acts[gpos.min() : gpos.max() + 1]
        take = gpos - gpos.min()
        x = th.tensor(sub[take].astype(np.float32), device=dev)
        with th.no_grad():
            fcode = ae.encode(x, use_threshold=True, normalize_activations=True)
        cc = fcode.index_select(1, col_t).cpu().numpy()  # (L,2): [:,0]=J,[:,1]=V
        Jtr = cc[:, 0]
        Vtr = cc[:, 1]
        if Jtr.max() > 0 and Vtr.max() > 0:
            out.append(
                dict(
                    seq_id=u,
                    length=c,
                    region=reg_h[f0 : f0 + c].tolist(),
                    V_trace=Vtr.tolist(),
                    J_trace=Jtr.tolist(),
                )
            )
        if len(out) >= 10:
            break
json.dump(dict(lat_V=LAT_V, lat_J=LAT_J, seqs=out), open(f"{RES}/ten_chain_traces.json", "w"))
print("collected", len(out), "chains; lengths", [o["length"] for o in out])
