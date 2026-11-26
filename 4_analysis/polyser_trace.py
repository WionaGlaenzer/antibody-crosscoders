#!/usr/bin/env python3
"""L916 (poly-Ser) per-residue activation along its top example proteins (general store)."""

import json, numpy as np, h5py, torch as th, pyarrow.parquet as pq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
dev = "cuda" if th.cuda.is_available() else "cpu"
LAT = 916
# example proteins (general parquet rows) + accessions
EX = {30965: "Q75JC9", 30967: "Q75JN3", 11033: "P04929"}
col_t = th.tensor([LAT], device=dev)
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_mixed_L20.pt", device=dev, dtype=th.float32)
ae.eval()
h5 = f"{ACTS}/general_single_L20.h5"
with h5py.File(h5, "r") as h:
    sid = h["seq_id"][:].astype(np.int64)
    N = sid.shape[0]
u, first, cnt = np.unique(sid, return_index=True, return_counts=True)
bounds = {int(uu): (int(f0), int(f0 + c)) for uu, f0, c in zip(u, first, cnt)}
t = pq.read_table(f"{BASE}/data/general_protein_sample.parquet", columns=["id", "seq"]).to_pydict()
ids = t["id"]
seqs = t["seq"]
out = {"latent": LAT, "proteins": {}}
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    for row, acc in EX.items():
        a, b = bounds[row]
        x = th.tensor(acts[a:b].astype(np.float32), device=dev)
        with th.no_grad():
            f = ae.encode(x, use_threshold=True, normalize_activations=True)
        tr = f.index_select(1, col_t)[:, 0].cpu().numpy().tolist()
        out["proteins"][acc] = {"row": row, "seq": seqs[row], "n_res": b - a, "act": tr}
json.dump(out, open(f"{RES}/polyser_trace.json", "w"))
print(
    "done:",
    {
        acc: (len(out["proteins"][acc]["act"]), round(max(out["proteins"][acc]["act"]), 3))
        for acc in out["proteins"]
    },
)
