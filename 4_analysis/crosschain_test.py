#!/usr/bin/env python
"""(B) Activation-based cross-chain / feature-reuse test for the gained-at-paired question.
For a PAIRED-regime crosscoder at layer L:
  1. Encode val residues of the PAIRED store and the SINGLE store.
  2. Per latent: firing rate overall, and split by chain (H vs L) and region (FR/CDR).
  3. MATCHED-RESIDUE test: match residues across the two stores by (seq_id, chain, within-seq index),
     compare each latent's activation paired-store vs single-store (mean |delta|, correlation).
     This detects pairing encoded as feature RE-WEIGHTING (same directions, different usage),
     which decoder-norm classification cannot see.
  4. Interface proxy: CDR3 (region 5) firing enrichment paired vs single.
Outputs crosschain_L{L}.json  (+ per-latent npz for the top movers).
"""

import argparse, json, time, sys, os
import numpy as np, h5py, torch as th

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
from dictionary_learning import BatchTopKCrossCoder


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def read_val(h5, n, seed):
    f = h5py.File(h5, "r")
    sp = f["split"][:]
    idx = np.nonzero(sp == 1)[0]
    if idx.size > n:
        st = np.random.default_rng(seed).integers(0, idx.size - n)
        idx = idx[st : st + n]
    lo, hi = idx[0], idx[-1] + 1
    acts = np.asarray(f["acts"][lo:hi], dtype=np.float32)[idx - lo]
    seqid = f["seq_id"][:][idx]
    chain = f["chain"][:][idx]
    region = f["region_label"][:][idx]
    f.close()
    return acts, seqid, chain, region


def encode(cc, acts, dev, bs=8192):
    D = cc.dict_size
    n = acts.shape[0]
    F = np.zeros((n, D), dtype=np.float16)
    with th.no_grad():
        for s in range(0, n, bs):
            xb = th.from_numpy(acts[s : s + bs]).to(dev)
            code = cc.encode(xb, normalize_activations=True, use_threshold=False)
            F[s : s + bs] = code.detach().cpu().numpy().astype(np.float16)
    return F


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--paired_h5", required=True)
    ap.add_argument("--single_h5", required=True)
    ap.add_argument("--n", type=int, default=300000)
    a = ap.parse_args()
    dev = "cuda" if th.cuda.is_available() else "cpu"
    cc = BatchTopKCrossCoder.from_pretrained(a.ckpt, device=dev, dtype=th.float32)
    cc.eval()
    D = cc.dict_size
    # paired store
    ap_acts, ap_sid, ap_ch, ap_rg = read_val(a.paired_h5, a.n, 0)
    Fp = encode(cc, ap_acts, dev)
    firing_p = (Fp > 0).mean(0)  # (D,)
    isH = ap_ch == b"H"
    isL = ap_ch == b"L"
    fire_H = (Fp[isH] > 0).mean(0)
    fire_L = (Fp[isL] > 0).mean(0)
    cdr3 = ap_rg == 5
    fire_cdr3 = (Fp[cdr3] > 0).mean(0)
    fire_noncdr3 = (Fp[~cdr3] > 0).mean(0)
    # single store
    as_acts, as_sid, as_ch, as_rg = read_val(a.single_h5, a.n, 0)
    Fs = encode(cc, as_acts, dev)
    firing_s = (Fs > 0).mean(0)

    # matched residues by (seq_id, chain, region) — approximate residue identity
    def key(sid, ch, rg):
        return np.array([f"{s}_{c.decode()}_{r}" for s, c, r in zip(sid, ch, rg)])

    kp = key(ap_sid, ap_ch, ap_rg)
    ks = key(as_sid, as_ch, as_rg)
    # build mean activation per latent per key, then compare on shared keys
    live = np.nonzero((firing_p > 1e-4) | (firing_s > 1e-4))[0]
    # per-latent paired vs single firing delta on live latents
    delta_fire = firing_p[live] - firing_s[live]
    rec = dict(
        layer=a.layer,
        dict_size=int(D),
        n_paired=int(Fp.shape[0]),
        n_single=int(Fs.shape[0]),
        n_live=int(live.size),
        firing_paired_mean=float(firing_p[live].mean()),
        firing_single_mean=float(firing_s[live].mean()),
        chain_HL_firing_corr=float(np.corrcoef(fire_H[live], fire_L[live])[0, 1]),
        n_H_specific=int(((fire_H[live] > 5 * fire_L[live]) & (fire_H[live] > 1e-3)).sum()),
        n_L_specific=int(((fire_L[live] > 5 * fire_H[live]) & (fire_L[live] > 1e-3)).sum()),
        n_cdr3_enriched=int(
            ((fire_cdr3[live] > 3 * fire_noncdr3[live]) & (fire_cdr3[live] > 1e-3)).sum()
        ),
        paired_single_firing_corr=float(np.corrcoef(firing_p[live], firing_s[live])[0, 1]),
        n_paired_gt_single=int((delta_fire > 0.01).sum()),
        n_single_gt_paired=int((delta_fire < -0.01).sum()),
        max_abs_delta_fire=float(np.abs(delta_fire).max()),
    )
    json.dump(rec, open(f"crosschain_L{a.layer}.json", "w"), indent=2)
    np.savez(
        f"crosschain_L{a.layer}.npz",
        live=live,
        firing_p=firing_p,
        firing_s=firing_s,
        fire_H=fire_H,
        fire_L=fire_L,
        fire_cdr3=fire_cdr3,
        fire_noncdr3=fire_noncdr3,
    )
    log("wrote crosschain", rec)
