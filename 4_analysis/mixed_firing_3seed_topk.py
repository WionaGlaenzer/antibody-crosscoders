#!/usr/bin/env python3
"""Fig2b/2c domain-specificity firing on the rebuild95 MIXED crosscoder, 3 seeds.
ENCODE PATH IDENTICAL TO ORIGINAL mixed_label.py: manual (x-train_mean)/train_std
normalization then ae.encode(xn, use_threshold=False) (true BatchTopK, NOT the JumpReLU
per-latent threshold). 400k-residue cap on the test split (kind-balanced), matching original.
Category from canonical decoder-norm rule. Writes mixed_firing_3seed_topk.npz."""

import os, sys, time, numpy as np, h5py, torch as th

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
os.environ["HF_HOME"] = f"{BASE}/caches/hf"
ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


DICT = 32768
CAP = 400000
from dictionary_learning.dictionary import BatchTopKCrossCoder

H5 = f"{ACTS}/exp_mixed_small_L20.h5"
with h5py.File(H5, "r") as f:
    tmean = th.tensor(f["train_mean"][:], device=dev, dtype=th.float32)
    tstd = th.tensor(f["train_std"][:], device=dev, dtype=th.float32)
    sp = f["split"][:]
    kd = f["kind"][:]
idx = np.nonzero(sp == 2)[0]
rng = np.random.default_rng(0)
if len(idx) > CAP:
    idx = np.sort(rng.choice(idx, CAP, replace=False))
kind_sel = kd[idx]
ngen = int((kind_sel == 1).sum())
nab = int((kind_sel == 0).sum())
log(f"test residues {len(idx)} | general {ngen} ab {nab}")
AB = np.zeros((3, DICT))
GEN = np.zeros((3, DICT))
ANY = np.zeros((3, DICT))
CAT = np.zeros((3, DICT), dtype=np.int64)
REL = np.zeros((3, DICT, 3))
for seed in [0, 1, 2]:
    ae = BatchTopKCrossCoder.from_pretrained(
        f"{CKPT}/cc_mixed_small_s{seed}_L20.pt", device=dev, dtype=th.float32
    )
    ae.eval()
    # canonical category from decoder norms
    W = ae.decoder.weight.detach()
    n = th.linalg.norm(W, dim=2).transpose(0, 1)
    rel = n / (n.sum(1, keepdim=True) + 1e-12)
    rp, ru, rpr = rel[:, 0], rel[:, 1], rel[:, 2]
    HI, LO = 0.5, 0.2
    cat = th.full((DICT,), 5, dtype=th.int64)
    cat[(rp > LO) & (ru > LO) & (rpr > LO)] = 0
    cat[(rp > HI) & (ru < LO) & (rpr < LO)] = 1
    cat[(rp < LO) & (ru > LO) & (rpr > LO)] = 2
    cat[(ru > HI) & (rpr < LO)] = 4
    cat[(rp < LO) & (ru < LO) & (rpr > HI)] = 3
    CAT[seed] = cat.cpu().numpy()
    REL[seed] = rel.cpu().numpy()
    fire_ab = np.zeros(DICT, np.int64)
    fire_gen = np.zeros(DICT, np.int64)
    fire_any = np.zeros(DICT, np.int64)
    BLK = 8192
    t0 = time.time()
    with h5py.File(H5, "r") as f:
        acts = f["acts"]
        for b0 in range(0, len(idx), BLK):
            bidx = idx[b0 : b0 + BLK]
            bkind = kd[bidx]
            x = th.tensor(acts[bidx].astype(np.float32), device=dev)  # (b,3,1024)
            xn = (x - tmean) / tstd
            fc = ae.encode(xn, use_threshold=False)  # true top-k
            fa = (fc > 0).cpu().numpy()  # (b,DICT)
            gm = bkind == 1
            am = bkind == 0
            fire_gen += fa[gm].sum(0)
            fire_ab += fa[am].sum(0)
            fire_any += fa.sum(0)
            if (b0 // BLK) % 25 == 0:
                log(f"seed{seed} {b0+len(bidx)}/{len(idx)} {time.time()-t0:.0f}s")
    AB[seed] = fire_ab / max(nab, 1)
    GEN[seed] = fire_gen / max(ngen, 1)
    ANY[seed] = fire_any / max(len(idx), 1)
    log(f"seed{seed} done")
np.savez(
    "mixed_firing_3seed_topk.npz",
    ab_freq=AB,
    gen_freq=GEN,
    any_freq=ANY,
    cat=CAT,
    rel=REL,
    nab=nab,
    ngen=ngen,
    cap=CAP,
    encode="use_threshold=False manual-norm",
)
log("ALL DONE")
