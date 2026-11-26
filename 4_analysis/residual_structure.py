#!/usr/bin/env python
"""Residual-structure test: does the paired-model reconstruction residual contain
low-rank, localized structure (=> paired-specific features the shared crosscoder missed),
or is it diffuse/isotropic (=> paired just marginally harder to fit)?

For the standard 2-model crosscoder cc2_unpVpair_L20, on test residues:
  x = (unp, pair) normalized; z = encode(x); x_hat = decode(z); e = x - x_hat.
Analyze per-model residual covariance C_m = E[e_m e_m^T] (1024x1024):
  - MSE, total residual variance per model
  - eigenspectrum; participation ratio PR = (sum lam)^2/sum lam^2  (effective rank)
  - fraction of residual variance in top-k PCs
  - EXCESS covariance C_pair - C_unp: its eigenspectrum tells whether the *extra*
    paired residual is low-rank (few large +eigenvalues) or diffuse.
  - localization: mean residual norm by region (paired vs unpaired) - is excess in CDR3?
"""

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
with h5py.File(H5, "r") as f:
    tmean = th.tensor(f["train_mean"][:], dtype=th.float32)[COLS].to(dev)
    tstd = th.tensor(f["train_std"][:], dtype=th.float32)[COLS].to(dev)
    sp = f["split"][:]
    region = f["region_label"][:]
    acts = f["acts"]
    idx = np.nonzero(sp == 2)[0]
    N = len(idx)
    log("test residues", N)
    reg_test = region[idx]
    Cunp = th.zeros(1024, 1024, device=dev)
    Cpair = th.zeros(1024, 1024, device=dev)
    sse = th.zeros(2, device=dev)
    svar = th.zeros(2, device=dev)
    ntot = 0
    # residual norm by region: region_map FR1..FR4 = 0..6
    reg_rn_unp = np.zeros(7)
    reg_rn_pair = np.zeros(7)
    reg_cnt = np.zeros(7)
    BLK = 16384
    for b0 in range(0, N, BLK):
        bidx = idx[b0 : b0 + BLK]
        br = reg_test[b0 : b0 + BLK]
        x = th.tensor(acts[bidx][:, COLS, :].astype(np.float32), device=dev)
        xn = (x - tmean) / tstd
        with th.no_grad():
            z = ae.encode(xn, use_threshold=True)
            xh = ae.decode(z)
            e = xn - xh  # (B,2,1024)
        sse += e.pow(2).sum(-1).sum(0)
        svar += xn.pow(2).sum(-1).sum(0)  # xn already ~zero-mean/unit
        eu = e[:, 0, :]
        ep = e[:, 1, :]
        Cunp += eu.T @ eu
        Cpair += ep.T @ ep
        ntot += e.shape[0]
        rn_u = eu.pow(2).sum(-1).cpu().numpy()
        rn_p = ep.pow(2).sum(-1).cpu().numpy()
        for r in range(7):
            m = br == r
            if m.any():
                reg_rn_unp[r] += rn_u[m].sum()
                reg_rn_pair[r] += rn_p[m].sum()
                reg_cnt[r] += m.sum()
        if (b0 // BLK) % 20 == 0:
            log(f"  {b0+len(bidx)}/{N}")
    Cunp /= ntot
    Cpair /= ntot
    mse = (sse / ntot).cpu().numpy()
    var = (svar / ntot).cpu().numpy()
    ve = 1 - mse / var
    # eigenspectra
    lam_u = th.linalg.eigvalsh(Cunp).flip(0).cpu().numpy().clip(min=0)
    lam_p = th.linalg.eigvalsh(Cpair).flip(0).cpu().numpy().clip(min=0)
    lam_ex = th.linalg.eigvalsh(Cpair - Cunp).flip(0).cpu().numpy()  # excess (can be +/-)

    def PR(l):
        l = l[l > 0]
        return float((l.sum() ** 2) / (np.square(l).sum()))

    def topfrac(l, k):
        return float(l[:k].sum() / l.sum())

    reg_names = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
    reg_mean_unp = (reg_rn_unp / np.maximum(reg_cnt, 1)).tolist()
    reg_mean_pair = (reg_rn_pair / np.maximum(reg_cnt, 1)).tolist()
    out = dict(
        n_test=int(ntot),
        mse=dict(unp=float(mse[0]), pair=float(mse[1])),
        total_resid_var=dict(unp=float(lam_u.sum()), pair=float(lam_p.sum())),
        ve=dict(unp=float(ve[0]), pair=float(ve[1])),
        participation_ratio=dict(unp=PR(lam_u), pair=PR(lam_p)),  # effective rank of residual
        resid_var_in_top=dict(
            unp_top10=topfrac(lam_u, 10),
            pair_top10=topfrac(lam_p, 10),
            unp_top50=topfrac(lam_u, 50),
            pair_top50=topfrac(lam_p, 50),
        ),
        excess_covariance=dict(
            total_excess=float((lam_p.sum() - lam_u.sum())),
            top10_positive_eigs=lam_ex[:10].tolist(),
            excess_var_in_top10=(
                float(lam_ex[lam_ex > 0][:10].sum() / lam_ex[lam_ex > 0].sum())
                if (lam_ex > 0).any()
                else 0.0
            ),
            n_positive_excess_eigs=int((lam_ex > 0).sum()),
            PR_excess_positive=PR(lam_ex[lam_ex > 0]),
        ),
        residual_norm_by_region=dict(
            region=reg_names,
            unp=reg_mean_unp,
            pair=reg_mean_pair,
            excess=[p - u for p, u in zip(reg_mean_pair, reg_mean_unp)],
        ),
    )
    np.savez(f"{BASE}/results/residual_structure_L20.npz", lam_u=lam_u, lam_p=lam_p, lam_ex=lam_ex)
    json.dump(
        out, open(f"{BASE}/results/residual_structure_summary.json", "w"), indent=2, default=float
    )
    log("SUMMARY", json.dumps(out, default=float)[:1200])
    log("ALL DONE")
