#!/usr/bin/env python
"""CORRECTED analysis:
A. Continuous decoder-norm r_pair distribution over ACTIVE latents; counts at
   thresholds 0.6/0.7/0.8; latent-scaling artifact test on each pool.
B. Residual structure with CORRECT normalization (matches training eval:
   xn=ae.normalize_activations(x); encode(normalize_activations=False);
   decode(denormalize_activations=False); e=xn-xhat), plus chain x region excess.
Fixes the double-normalization bug (previous VE was -49, impossible).
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
reg_names = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]

# ---- A. decoder norms (weights only) ----
Wdec = ae.decoder.weight.detach()  # (2, dict, act)
n = th.linalg.norm(Wdec, dim=2).transpose(0, 1)  # (dict,2) [unp,pair]
s = n.sum(1, keepdim=True)
rel = n / (s + 1e-12)
rpr = rel[:, 1].cpu().numpy()  # r_pair

with h5py.File(H5, "r") as f:
    tmean = f["train_mean"][:]
    tstd = f["train_std"][:]
    sp = f["split"][:]
    region = f["region_label"][:]
    chain = f["chain"][:]
    acts = f["acts"]
    idx = np.nonzero(sp == 2)[0]
    N = len(idx)
    log("test residues", N)
    reg_test = region[idx]
    chain_test = chain[idx]
    is_heavy = np.array([c == b"H" or c == "H" for c in chain_test])
    # activation freq + residual, in ONE pass, CORRECT normalization
    act_cnt = th.zeros(ae.dict_size, device=dev)
    Cunp = th.zeros(1024, 1024, device=dev)
    Cpair = th.zeros(1024, 1024, device=dev)
    tot_e = th.zeros(2, device=dev)
    tot_var = th.zeros(2, device=dev)
    acc_unp = np.zeros((2, 7))
    acc_pair = np.zeros((2, 7))
    cnt = np.zeros((2, 7))
    # need global mean for tot_var (use running; approximate with per-block mean-subtract like training)
    BLK = 16384
    ntot = 0
    for b0 in range(0, N, BLK):
        bidx = idx[b0 : b0 + BLK]
        br = reg_test[b0 : b0 + BLK]
        bh = is_heavy[b0 : b0 + BLK]
        xb = th.tensor(acts[bidx][:, COLS, :].astype(np.float32), device=dev)
        with th.no_grad():
            xn = ae.normalize_activations(xb, inplace=False)  # normalize ONCE, via buffers
            fcode = ae.encode(xn, normalize_activations=False, use_threshold=True)
            xhat = ae.decode(fcode, denormalize_activations=False)  # stay normalized
            e = xn - xhat
        act_cnt += (fcode > 0).float().sum(0)
        tot_e += e.pow(2).sum(-1).sum(0)
        tot_var += (xn - xn.mean(0, keepdim=True)).pow(2).sum(-1).sum(0)
        eu = e[:, 0, :]
        ep = e[:, 1, :]
        Cunp += eu.T @ eu
        Cpair += ep.T @ ep
        ntot += e.shape[0]
        rn_u = eu.pow(2).sum(-1).cpu().numpy()
        rn_p = ep.pow(2).sum(-1).cpu().numpy()
        for ci, cm in enumerate([bh, ~bh]):
            for r in range(7):
                m = cm & (br == r)
                if m.any():
                    acc_unp[ci, r] += rn_u[m].sum()
                    acc_pair[ci, r] += rn_p[m].sum()
                    cnt[ci, r] += m.sum()
        if (b0 // BLK) % 20 == 0:
            log(f"  {b0+len(bidx)}/{N}")
    act_freq = (act_cnt / ntot).cpu().numpy()
    active = act_freq > 1e-4
    ve = (1.0 - tot_e / (tot_var + 1e-12)).cpu().numpy()
    log(f"CORRECTED VE unp {ve[0]:.4f} pair {ve[1]:.4f}  (should be ~0.98)")
    Cunp /= ntot
    Cpair /= ntot
    lam_ex = th.linalg.eigvalsh(Cpair - Cunp).flip(0).cpu().numpy()
    pos = lam_ex[lam_ex > 0]

    # continuous distribution + threshold counts (active only)
    rpr_active = rpr[active]
    hist, edges = np.histogram(rpr_active, bins=50, range=(0, 1))
    thresholds = {}
    for thr in [0.6, 0.7, 0.8]:
        pool = np.nonzero(active & (rpr > thr))[0]
        thresholds[str(thr)] = dict(
            n_paired=int(pool.size), n_unpaired=int((active & (rpr < (1 - thr))).sum())
        )

    # latent scaling on r>0.6 pool (largest) using CORRECT norm
    def latent_scale(pool):
        if pool.size == 0:
            return dict(n=0)
        gidx = th.tensor(pool, device=dev, dtype=th.long)
        Dg = Wdec[:, gidx, :]
        Dn = th.linalg.norm(Dg, dim=2)
        Dh = Dg / (Dn.unsqueeze(2) + 1e-12)
        G = pool.size
        num_r = th.zeros(2, G, device=dev)
        num_e = th.zeros(2, G, device=dev)
        den = th.zeros(G, device=dev)
        with th.no_grad():
            for b0 in range(0, N, BLK):
                bidx = idx[b0 : b0 + BLK]
                xb = th.tensor(acts[bidx][:, COLS, :].astype(np.float32), device=dev)
                xn = ae.normalize_activations(xb, inplace=False)
                fc = ae.encode(xn, normalize_activations=False, use_threshold=False)
                xhat = ae.decode(fc, denormalize_activations=False)
                fg = fc[:, gidx]
                den += (fg * fg).sum(0)
                err = xn - xhat
                for m in range(2):
                    num_r[m] += ((xhat[:, m, :] @ Dh[m].t()) * fg).sum(0)
                    num_e[m] += ((err[:, m, :] @ Dh[m].t() + fg * Dn[m].unsqueeze(0)) * fg).sum(0)
        br_ = (num_r / (den.unsqueeze(0) + 1e-12)).cpu().numpy()
        be = (num_e / (den.unsqueeze(0) + 1e-12)).cpu().numpy()
        nu_r = np.clip(br_[0] / (np.abs(br_[1]) + 1e-9), -1, 5)
        nu_e = np.clip(be[0] / (np.abs(be[1]) + 1e-9), -1, 5)
        art = (nu_r >= 0.5) | (nu_e >= 0.5)
        return dict(n=int(G), flagged_artifact=int(art.sum()), survive=int((~art).sum()))

    ls06 = latent_scale(np.nonzero(active & (rpr > 0.6))[0])
    ls07 = latent_scale(np.nonzero(active & (rpr > 0.7))[0])
    log(f"latent-scale r>0.6: {ls06} | r>0.7: {ls07}")

    out = dict(
        corrected_ve=dict(unp=float(ve[0]), pair=float(ve[1])),
        n_active=int(active.sum()),
        rpr_hist=dict(counts=hist.tolist(), edges=edges.tolist()),
        rpr_active_median=float(np.median(rpr_active)),
        rpr_active_p90=float(np.percentile(rpr_active, 90)),
        rpr_active_p99=float(np.percentile(rpr_active, 99)),
        rpr_active_max=float(rpr_active.max()),
        thresholds=thresholds,
        latent_scaling=dict(r_gt_0p6=ls06, r_gt_0p7=ls07),
        excess_low_rank=dict(
            top_frac=float(pos[0] / pos.sum()),
            ratio_1_2=float(pos[0] / pos[1]),
            PR=float((pos.sum() ** 2) / np.square(pos).sum()),
            top10=pos[:10].tolist(),
        ),
        chain_region=dict(
            heavy=dict(
                region=reg_names,
                excess=(
                    acc_pair[0] / np.maximum(cnt[0], 1) - acc_unp[0] / np.maximum(cnt[0], 1)
                ).tolist(),
            ),
            light=dict(
                region=reg_names,
                excess=(
                    acc_pair[1] / np.maximum(cnt[1], 1) - acc_unp[1] / np.maximum(cnt[1], 1)
                ).tolist(),
            ),
            heavy_mean_excess=float((acc_pair[0].sum() - acc_unp[0].sum()) / max(cnt[0].sum(), 1)),
            light_mean_excess=float((acc_pair[1].sum() - acc_unp[1].sum()) / max(cnt[1].sum(), 1)),
            heavy_n=int(cnt[0].sum()),
            light_n=int(cnt[1].sum()),
        ),
    )
    np.savez(f"{BASE}/results/decnorm_corrected_L20.npz", rpr=rpr, active=active, lam_ex=lam_ex)
    json.dump(
        out, open(f"{BASE}/results/decnorm_residual_corrected.json", "w"), indent=2, default=float
    )
    log("SUMMARY", json.dumps(out, default=float)[:1400])
    log("ALL DONE")
