#!/usr/bin/env python3
"""Train-VE vs Val-VE at a fixed checkpoint (no retraining). Same VE formula as the
trainer's eval: VE = 1 - sum(e^2)/sum((x-xbar)^2) on normalized activations, per model.
Evaluates split==0 (train) and split==1 (val), each capped, one forward pass. Writes
trainval_ve_{tag}.json."""

import json, time, argparse, numpy as np, h5py, torch as th

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE as DEFAULT_BASE

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--h5", required=True)
ap.add_argument("--base", default=DEFAULT_BASE)
ap.add_argument("--cap", type=int, default=500000)
args = ap.parse_args()
BASE = args.base
RES = f"{BASE}/results"
import sys

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
import os

os.environ["HF_HOME"] = f"{BASE}/caches/hf"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(args.ckpt, device=dev, dtype=th.float32)
ae.eval()
log("loaded", args.ckpt)


def ve_for_split(h5, split_val, cap):
    # SEQUENTIAL block reads (fancy-indexing scattered rows is too slow), but sample
    # SPREAD ACROSS THE WHOLE FILE (not first-N) so the sample is representative even when
    # the store is block-ordered by kind (mixed: antibody-block first) or when two stores
    # share a sorted prefix. Within each contiguous block keep split rows, then subsample
    # each block at rate = cap/total_split so coverage spans the entire file.
    with h5py.File(h5, "r") as h:
        sp = h["split"][:]
        acts = h["acts"]
        N = acts.shape[0]
        n_split = int((sp == split_val).sum())
        BLK = 262144
        ABS = 8192
        # block-level stride: read evenly-spaced blocks across the whole file (few reads,
        # full-distribution coverage). Target ~cap split rows given the split's density.
        density = n_split / N
        blocks = list(range(0, N, BLK))
        nblk = len(blocks)
        want_blocks = max(1, int(np.ceil(cap / max(density * BLK, 1))))
        stride = max(1, nblk // max(want_blocks, 1))
        sel_blocks = blocks[::stride]
        fv_e = th.zeros(3, device=dev)
        fv_var = th.zeros(3, device=dev)
        nres = 0
        with th.no_grad():
            for start in sel_blocks:
                if nres >= cap:
                    break
                stop = min(start + BLK, N)
                bm = sp[start:stop] == split_val
                if not bm.any():
                    continue
                chunk = acts[start:stop][bm]  # sequential read of a spread block
                if len(chunk) == 0:
                    continue
                for s in range(0, len(chunk), ABS):
                    xb = th.tensor(chunk[s : s + ABS].astype(np.float32), device=dev)
                    xn = ae.normalize_activations(xb, inplace=False)
                    f = ae.encode(xn, normalize_activations=False, use_threshold=False)
                    xhat = ae.decode(f, denormalize_activations=False)
                    e = xn - xhat
                    fv_e += e.pow(2).sum(-1).sum(0)
                    fv_var += (xn - xn.mean(0, keepdim=True)).pow(2).sum(-1).sum(0)
                    nres += len(xb)
    ve = (1.0 - fv_e / (fv_var + 1e-12)).cpu().numpy()
    return ve, nres


out = {"tag": args.tag, "models": ["prot_bert", "igbert_unpaired", "igbert_paired"]}
for name, sv in [("train", 0), ("val", 1), ("test", 2)]:
    ve, n = ve_for_split(args.h5, sv, args.cap)
    out[name] = {"ve": [float(x) for x in ve], "n_residues": int(n), "ve_mean": float(ve.mean())}
    log(f"{name}: n={n} VE[p/u/pr]={ve[0]:.4f}/{ve[1]:.4f}/{ve[2]:.4f} mean={ve.mean():.4f}")
out["gap_train_minus_val"] = [out["train"]["ve"][i] - out["val"]["ve"][i] for i in range(3)]
json.dump(out, open(f"{RES}/trainval_ve_{args.tag}.json", "w"), indent=1)
log("DONE")
print(json.dumps(out, indent=1))
