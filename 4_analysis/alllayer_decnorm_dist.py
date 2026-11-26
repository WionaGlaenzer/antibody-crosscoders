#!/usr/bin/env python
"""All-layer 2-model relative paired-decoder-norm distribution from saved checkpoints.
Replicates orig twomodel_dist.py per layer: r_pair = ||d_pair|| / (||d_unp|| + ||d_pair||)
(raw decoder norm, matches classify_decoder_norms), restricted to ACTIVE latents
(val firing freq > 1e-4). Uses cc2_paired_alllayer_L{N}_last.pt + exp_paired_small_L{N}.h5.
Dumps per-layer r_pair arrays to alllayer_decnorm_dist.npz."""

import os, sys, json, time, glob
import numpy as np, h5py, torch as th

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
from dictionary_learning.dictionary import BatchTopKCrossCoder

CK = f"{BASE}/checkpoints"
ACT = f"{BASE}/activations"
COLS = [1, 2]  # unpaired, paired columns
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


layers = []
for L in range(1, 31):
    ck = f"{CK}/cc2_paired_alllayer_L{L}_last.pt"
    if not os.path.exists(ck):
        ck = f"{CK}/cc2_paired_alllayer_L{L}_best.pt"
    h5 = f"{ACT}/exp_paired_small_L{L}.h5"
    if os.path.exists(ck) and os.path.exists(h5):
        layers.append((L, ck, h5))
log("layers with ckpt+store:", [L for L, _, _ in layers])

out = {}
for L, ck, h5 in layers:
    sd = th.load(ck, map_location="cpu")
    Wdec = sd["decoder.weight"]  # (2, dict, 1024) [unp, pair]
    n = th.linalg.norm(Wdec, dim=2).transpose(0, 1)  # (dict,2)
    s = n.sum(1, keepdim=True)
    rel = n / (s + 1e-12)
    r_pair_all = rel[:, 1].numpy()  # 2-way ratio, centered 0.5
    dict_size = Wdec.shape[1]
    with h5py.File(h5, "r") as f:
        sp = f["split"][:]
        vidx = np.nonzero(sp == 1)[0]
        if len(vidx) > 60000:
            vidx = vidx[:60000]  # CPU-friendly active-mask subset
        tmean = th.tensor(f["train_mean"][:], dtype=th.float32)[COLS]
        tstd = th.tensor(f["train_std"][:], dtype=th.float32)[COLS]
        acts = f["acts"]
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
        with th.no_grad():
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
    out[f"L{L}"] = rp.astype(np.float32)
    log(
        f"L{L}: n_active {int(active.sum())} median {np.median(rp):.4f} p99 {np.percentile(rp,99):.4f} max {rp.max():.4f}"
    )

np.savez("alllayer_decnorm_dist.npz", **out)
log("DONE ->", "alllayer_decnorm_dist.npz", "layers:", sorted(int(k[1:]) for k in out))
