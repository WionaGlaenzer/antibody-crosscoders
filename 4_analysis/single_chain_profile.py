#!/usr/bin/env python3
"""Per-residue V/J-latent activation along ONE example heavy chain (actual positions)."""

import sys, time, json, numpy as np, h5py, torch as th, pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
REGION_NAMES = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
dev = "cuda" if th.cuda.is_available() else "cpu"
pv = pd.read_csv(f"{RES}/concept_j_positional_validation.csv")
p = pv[pv.config == "paired_L20"]
J = p[p.kind == "J"].latent_id.astype(int).tolist()
V = p[p.kind == "V"].latent_id.astype(int).tolist()
cols = J + V
grp = np.array(["J"] * len(J) + ["V"] * len(V))
col_t = th.tensor(cols, device=dev)
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
reg_h = region[idx].astype(np.int64)
# group heavy residues by seq_id; need contiguous per-seq blocks (they are, store grouped)
uniq, first, counts = np.unique(sid_h, return_index=True, return_counts=True)
# candidate example seqs: length 110-135, contains all 7 regions, decent length
cand = []
for u, f0, c in zip(uniq, first, counts):
    if 110 <= c <= 140:
        rr = reg_h[f0 : f0 + c]
        if len(set(rr.tolist())) == 7:  # all regions present
            cand.append((int(u), int(f0), int(c)))
    if len(cand) >= 40:
        break


# encode candidates' residues, score V and J mean activation, pick one where both fire
def encode_block(gidx):
    x = th.tensor(np.stack([np.zeros(1)]), device=dev)  # placeholder


best = None
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    for u, f0, c in cand:
        gpos = idx[f0 : f0 + c]  # global row indices (contiguous? not necessarily) -> gather
        sub = acts[gpos.min() : gpos.max() + 1]
        take = gpos - gpos.min()
        x = th.tensor(sub[take].astype(np.float32), device=dev)
        with th.no_grad():
            fcode = ae.encode(x, use_threshold=True, normalize_activations=True)
        cc = fcode.index_select(1, col_t).cpu().numpy()  # (L, ncol)
        vmean = cc[:, grp == "V"].max(1).mean()
        jmax = cc[:, grp == "J"].max(1).max()
        score = vmean + jmax
        rr = reg_h[f0 : f0 + c]
        rec = dict(
            seq_id=u,
            length=c,
            V=cc[:, grp == "V"].mean(1).tolist(),
            J=cc[:, grp == "J"].mean(1).tolist(),
            region=rr.tolist(),
            vmean=float(vmean),
            jmax=float(jmax),
        )
        if best is None or score > best["score"]:
            best = {"score": float(score), **rec}
json.dump(best, open(f"{RES}/single_chain_profile.json", "w"))
print(
    "picked seq_id",
    best["seq_id"],
    "len",
    best["length"],
    "vmean",
    round(best["vmean"], 4),
    "jmax",
    round(best["jmax"], 4),
)
