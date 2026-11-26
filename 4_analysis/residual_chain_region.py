#!/usr/bin/env python
"""Chain x region breakdown of the excess paired residual.
Same as residual_structure but accumulates residual norm by (chain H/L) x region,
to answer: does the FR1/FR4 excess live in the HEAVY or the LIGHT chain?"""

import json, time, numpy as np, h5py, torch as th, sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


from dictionary_learning.dictionary import BatchTopKCrossCoder

dev = "cuda" if th.cuda.is_available() else "cpu"
ae = BatchTopKCrossCoder.from_pretrained(
    f"{BASE}/checkpoints/cc2_unpVpair_L20.pt", device=dev, dtype=th.float32
).eval()
H5 = f"{BASE}/activations/full_paired_L20.h5"
COLS = [1, 2]
reg_names = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
with h5py.File(H5, "r") as f:
    print("keys:", list(f.keys()), flush=True)
    tmean = th.tensor(f["train_mean"][:], dtype=th.float32)[COLS].to(dev)
    tstd = th.tensor(f["train_std"][:], dtype=th.float32)[COLS].to(dev)
    sp = f["split"][:]
    region = f["region_label"][:]
    chain = f["chain"][:]
    acts = f["acts"]
    idx = np.nonzero(sp == 2)[0]
    N = len(idx)
    log("test residues", N)
    reg_test = region[idx]
    chain_test = chain[idx]
    # chain field is bytes b'H'/b'L'
    is_heavy = np.array([c == b"H" or c == "H" for c in chain_test])
    log("heavy residues", int(is_heavy.sum()), "light", int((~is_heavy).sum()))
    # accumulators: [chain(0=H,1=L)][region] -> summed residual norm^2, count, per model
    acc_unp = np.zeros((2, 7))
    acc_pair = np.zeros((2, 7))
    cnt = np.zeros((2, 7))
    BLK = 16384
    for b0 in range(0, N, BLK):
        bidx = idx[b0 : b0 + BLK]
        br = reg_test[b0 : b0 + BLK]
        bh = is_heavy[b0 : b0 + BLK]
        x = th.tensor(acts[bidx][:, COLS, :].astype(np.float32), device=dev)
        xn = (x - tmean) / tstd
        with th.no_grad():
            z = ae.encode(xn, use_threshold=True)
            xh = ae.decode(z)
            e = xn - xh
        rn_u = e[:, 0, :].pow(2).sum(-1).cpu().numpy()
        rn_p = e[:, 1, :].pow(2).sum(-1).cpu().numpy()
        for ci, cmask in enumerate([bh, ~bh]):
            for r in range(7):
                m = cmask & (br == r)
                if m.any():
                    acc_unp[ci, r] += rn_u[m].sum()
                    acc_pair[ci, r] += rn_p[m].sum()
                    cnt[ci, r] += m.sum()
        if (b0 // BLK) % 20 == 0:
            log(f"  {b0+len(bidx)}/{N}")
    mean_u = acc_unp / np.maximum(cnt, 1)
    mean_p = acc_pair / np.maximum(cnt, 1)
    out = {}
    for ci, cn in enumerate(["heavy", "light"]):
        out[cn] = dict(
            region=reg_names,
            unp=mean_u[ci].tolist(),
            pair=mean_p[ci].tolist(),
            excess=(mean_p[ci] - mean_u[ci]).tolist(),
            count=cnt[ci].tolist(),
        )
    # totals per chain
    out["chain_totals"] = dict(
        heavy_excess_total=float((acc_pair[0] - acc_unp[0]).sum()),
        light_excess_total=float((acc_pair[1] - acc_unp[1]).sum()),
        heavy_n=int(cnt[0].sum()),
        light_n=int(cnt[1].sum()),
        heavy_mean_excess=float((acc_pair[0].sum() - acc_unp[0].sum()) / max(cnt[0].sum(), 1)),
        light_mean_excess=float((acc_pair[1].sum() - acc_unp[1].sum()) / max(cnt[1].sum(), 1)),
    )
    json.dump(out, open(f"{BASE}/results/residual_chain_region.json", "w"), indent=2, default=float)
    log("SUMMARY", json.dumps(out, default=float)[:1500])
    log("ALL DONE")
