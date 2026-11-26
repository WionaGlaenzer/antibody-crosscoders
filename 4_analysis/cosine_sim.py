#!/usr/bin/env python
"""(C) Raw-activation similarity, no crosscoder needed.
For each layer L in {6,12,20,28}:
  - within the PAIRED store, mean cosine(igbert_unpaired ch1, igbert_paired ch2) per residue (val split)
  - within the SINGLE store, same
  - cross-store: for residues matched by (seq_id,chain,order), cosine between the igbert_paired
    channel in the single store vs the paired store  -> "how much does pairing change IgBert-paired acts"
Writes cosine_sim.json
"""

import json, time, numpy as np, h5py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

ACT = f"{BASE}/activations"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def cos_rows(A, B):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-8
    return num / den


def sample_val(h5, n=200000, seed=0):
    f = h5py.File(h5, "r")
    sp = f["split"][:]
    idx = np.nonzero(sp == 1)[0]
    if idx.size > n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
    acts = f["acts"]
    a = np.asarray(acts[list(idx)], dtype=np.float32)  # (n,3,1024)
    seqid = f["seq_id"][:][idx]
    chain = f["chain"][:][idx]
    region = f["region_label"][:][idx]
    f.close()
    return a, seqid, chain, region, idx


out = {}
for L in (6, 12, 20, 28):
    rec = {}
    # paired store: ch1 vs ch2 (unpaired vs paired IgBert)
    ap, sp_p, ch_p, rg_p, _ = sample_val(f"{ACT}/full_paired_L{L}.h5")
    c_pair = cos_rows(ap[:, 1, :], ap[:, 2, :])
    rec["cos_unp_vs_pair__paired_store"] = dict(
        mean=float(c_pair.mean()),
        std=float(c_pair.std()),
        p05=float(np.percentile(c_pair, 5)),
        p50=float(np.percentile(c_pair, 50)),
        p95=float(np.percentile(c_pair, 95)),
    )
    # single store: ch1 vs ch2
    a_s, sp_s, ch_s, rg_s, _ = sample_val(f"{ACT}/full_single_L{L}.h5")
    c_sing = cos_rows(a_s[:, 1, :], a_s[:, 2, :])
    rec["cos_unp_vs_pair__single_store"] = dict(mean=float(c_sing.mean()), std=float(c_sing.std()))
    # also prot_bert vs igbert_paired (base vs specialized)
    c_bp = cos_rows(ap[:, 0, :], ap[:, 2, :])
    rec["cos_prot_vs_pair__paired_store"] = dict(mean=float(c_bp.mean()), std=float(c_bp.std()))
    c_bu = cos_rows(ap[:, 0, :], ap[:, 1, :])
    rec["cos_prot_vs_unp__paired_store"] = dict(mean=float(c_bu.mean()), std=float(c_bu.std()))
    out[f"L{L}"] = rec
    log(
        f"L{L}: unp-vs-pair(paired store) cos={c_pair.mean():.4f}  (single store) {c_sing.mean():.4f}  "
        f"prot-vs-pair {c_bp.mean():.4f} prot-vs-unp {c_bu.mean():.4f}"
    )

json.dump(out, open("cosine_sim.json", "w"), indent=2)
log("wrote cosine_sim.json")
