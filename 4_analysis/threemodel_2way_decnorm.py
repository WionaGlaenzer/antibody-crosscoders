#!/usr/bin/env python
"""3-model crosscoder: the unpaired-vs-paired 2-WAY decoder-norm share, computed
by DROPPING the ProtBERT decoder and renormalizing over the two IgBERT decoders:
    r_pair_2way = ||d_pair|| / (||d_unp|| + ||d_pair||)   (RAW norms, cols 1,2 of 3)
so it is directly comparable to the dedicated 2-model crosscoder's r_pair.
Active = val firing frequency > 1e-4 (paired-regime store). Per layer L6/12/20/28,
3-model 'paired' config checkpoints cc_paired_L{L}.pt (decoder (3,dict,1024))."""

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
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


out = {}
for L in [6, 12, 20, 28]:
    ck = f"{CK}/cc_paired_L{L}.pt"
    if not os.path.exists(ck):
        log("missing", ck)
        continue
    sd = th.load(ck, map_location="cpu")
    Wdec = sd["decoder.weight"]  # (3, dict, 1024): 0=prot,1=unp,2=pair
    n = th.linalg.norm(Wdec, dim=2)  # (3, dict)
    n_unp = n[1]
    n_pair = n[2]
    r_pair_2way_all = (n_pair / (n_unp + n_pair + 1e-12)).numpy()
    # active mask from paired-regime val firing
    h5 = f"{ACT}/full_paired_L{L}.h5"
    with h5py.File(h5, "r") as f:
        sp = f["split"][:]
        vidx = np.nonzero(sp == 1)[0]
        if len(vidx) > 300000:
            vidx = vidx[:300000]
        tmean = th.tensor(f["train_mean"][:], dtype=th.float32)  # (3,1024)
        tstd = th.tensor(f["train_std"][:], dtype=th.float32)
        acts = f["acts"]
        dict_size = Wdec.shape[1]
        ae = BatchTopKCrossCoder(
            activation_dim=1024,
            dict_size=dict_size,
            num_layers=3,
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
            xb = th.tensor(acts[ii.tolist()], dtype=th.float32, device=dev)  # (b,3,1024)
            xn = ae.normalize_activations(xb, inplace=False)
            fcode = ae.encode(xn, normalize_activations=False, use_threshold=False)
            cnt += (fcode > 0).float().sum(0).cpu().numpy()
            nb += len(ii)
    active = (cnt / max(nb, 1)) > 1e-4
    rp = r_pair_2way_all[active]
    out[f"L{L}"] = {
        "n_active": int(active.sum()),
        "r_pair_2way": rp.tolist(),
        "median": float(np.median(rp)),
        "p90": float(np.percentile(rp, 90)),
        "p99": float(np.percentile(rp, 99)),
        "max": float(rp.max()),
    }
    log(
        f"L{L}: n_active {int(active.sum())} median {np.median(rp):.4f} p99 {np.percentile(rp,99):.4f} max {rp.max():.4f}"
    )
json.dump(out, open(f"{RES}/threemodel_2way_decnorm.json", "w"))
log("DONE")
