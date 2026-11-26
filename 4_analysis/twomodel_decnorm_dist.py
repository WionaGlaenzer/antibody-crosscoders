#!/usr/bin/env python
"""Compute the 2-model (unpaired vs paired) relative paired-decoder-norm
distribution over ACTIVE latents, per layer (L6/L12/L20/L28). Threshold-free view:
shows the r_pair distribution shifts slightly with depth but never crosses the
paired-specific classification threshold (HI=0.5 on the 2-way sum-normalized norm).
r_pair = ||d_pair|| / (||d_unp|| + ||d_pair||),  raw decoder norm (matches classify).
Active = val firing frequency > 1e-4 (same criterion as the 2-model training)."""

import os, sys, json, time
import numpy as np, h5py, torch as th

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
from dictionary_learning.dictionary import BatchTopKCrossCoder

CK = f"{BASE}/checkpoints"
ACT = f"{BASE}/activations"
RES = f"{BASE}/results"
COLS = [1, 2]  # unpaired, paired columns in the paired store
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


out = {}
for L in [6, 12, 20, 28]:
    ck = f"{CK}/cc2_unpVpair_L{L}.pt"
    if not os.path.exists(ck):
        log("missing", ck)
        continue
    sd = th.load(ck, map_location="cpu")
    Wdec = sd["decoder.weight"]  # (2, dict, 1024)
    n = th.linalg.norm(Wdec, dim=2).transpose(0, 1)  # (dict,2)  [unp, pair]
    s = n.sum(1, keepdim=True)
    rel = n / (s + 1e-12)
    r_pair_all = rel[:, 1].numpy()
    # active mask via val firing frequency on the paired store
    h5 = f"{ACT}/full_paired_L{L}.h5"
    with h5py.File(h5, "r") as f:
        sp = f["split"][:]
        vidx = np.nonzero(sp == 1)[0]
        if len(vidx) > 300000:
            vidx = vidx[:300000]
        tmean = th.tensor(f["train_mean"][:], dtype=th.float32)[COLS]
        tstd = th.tensor(f["train_std"][:], dtype=th.float32)[COLS]
        acts = f["acts"]
        dict_size = Wdec.shape[1]
        ae = BatchTopKCrossCoder(
            activation_dim=1024,
            dict_size=dict_size,
            num_layers=2,
            k=32,
            activation_mean=tmean,
            activation_std=tstd,
            target_rms=1.0,
            code_normalization="CROSSCODER",
        )
        ae.load_state_dict(sd)
        ae = ae.to(dev).eval()
        cnt = np.zeros(dict_size)
        nb = 0
        BLK = 8192
        for st in range(0, len(vidx), BLK):
            ii = vidx[st : st + BLK]
            xb = th.tensor(acts[ii.tolist()][:, COLS, :], dtype=th.float32, device=dev)
            xn = ae.normalize_activations(xb, inplace=False)
            fcode = ae.encode(xn, normalize_activations=False, use_threshold=False)
            cnt += (fcode > 0).float().sum(0).cpu().numpy()
            nb += len(ii)
    freq = cnt / max(nb, 1)
    active = freq > 1e-4
    rp = r_pair_all[active]
    out[f"L{L}"] = {
        "n_active": int(active.sum()),
        "r_pair": rp.tolist(),
        "median": float(np.median(rp)),
        "p90": float(np.percentile(rp, 90)),
        "p99": float(np.percentile(rp, 99)),
        "max": float(rp.max()),
    }
    log(
        f"L{L}: n_active {int(active.sum())} median {np.median(rp):.4f} p99 {np.percentile(rp,99):.4f} max {rp.max():.4f}"
    )

json.dump(out, open(f"{RES}/twomodel_decnorm_dist.json", "w"))
log("DONE", f"{RES}/twomodel_decnorm_dist.json")
