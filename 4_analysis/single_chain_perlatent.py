#!/usr/bin/env python3
"""Per-LATENT V/J activation traces along example heavy chain seq 167."""

import sys, json, numpy as np, h5py, torch as th, pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
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
TARGET = 167
with h5py.File(h5, "r") as h:
    split = h["split"][:]
    chain = h["chain"][:]
    region = h["region_label"][:]
    sid = h["seq_id"][:].astype(np.int64)
    N = split.shape[0]
keep = (split == 2) & (chain == b"H") & (sid == TARGET)
gpos = np.nonzero(keep)[0]
reg = region[gpos].astype(int).tolist()
with h5py.File(h5, "r") as h:
    sub = h["acts"][gpos.min() : gpos.max() + 1]
take = gpos - gpos.min()
x = th.tensor(sub[take].astype(np.float32), device=dev)
with th.no_grad():
    fcode = ae.encode(x, use_threshold=True, normalize_activations=True)
cc = fcode.index_select(1, col_t).cpu().numpy()  # (L, ncol)
# pick best single V and best single J latent by max activation along this chain
Vcols = [i for i, g in enumerate(grp) if g == "V"]
Jcols = [i for i, g in enumerate(grp) if g == "J"]
bestV = Vcols[int(cc[:, Vcols].max(0).argmax())]
bestJ = Jcols[int(cc[:, Jcols].max(0).argmax())]
out = dict(
    seq_id=TARGET,
    length=len(reg),
    region=reg,
    bestV_latent=int(cols[bestV]),
    bestJ_latent=int(cols[bestJ]),
    V_trace=cc[:, bestV].tolist(),
    J_trace=cc[:, bestJ].tolist(),
)
json.dump(out, open(f"{RES}/single_chain_perlatent.json", "w"))
print(
    "bestV latent",
    cols[bestV],
    "max",
    round(float(cc[:, bestV].max()), 4),
    "| bestJ latent",
    cols[bestJ],
    "max",
    round(float(cc[:, bestJ].max()), 4),
)
