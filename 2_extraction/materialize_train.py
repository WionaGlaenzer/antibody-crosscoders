#!/usr/bin/env python
"""One-time rewrite of the paired-L20 scale-up train split (split==0) into a
CONTIGUOUS raw fp16 file on scratch, so a true one-epoch training pass can read
it as fast sequential slices (no HDF5 chunk overhead, no per-block boolean mask).
Also writes a small meta json (n_train, shape) alongside. Norm stats stay in the
source h5 and are read by the trainer."""

import os, sys, json, time
import numpy as np, h5py

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE, SCRATCH

H5 = f"{BASE}/activations/scaleup_paired_L20.h5"
OUT = f"{SCRATCH}/scaleup_train_contig.f16"
META = f"{SCRATCH}/scaleup_train_contig.meta.json"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


f = h5py.File(H5, "r")
split = f["split"][:]
acts = f["acts"]
N = acts.shape[0]
n_train = int((split == 0).sum())
log("N", N, "n_train", n_train, "-> file GB", round(n_train * 3 * 1024 * 2 / 1e9, 1))
mm = np.memmap(OUT, dtype=np.float16, mode="w+", shape=(n_train, 3, 1024))
BLK = 524288
wp = 0
t0 = time.time()
for start in range(0, N, BLK):
    stop = min(start + BLK, N)
    m = split[start:stop] == 0
    k = int(m.sum())
    if k == 0:
        continue
    mm[wp : wp + k] = acts[start:stop][m]
    wp += k
    if (start // BLK) % 20 == 0:
        el = time.time() - t0
        log(f"  {wp}/{n_train} ({100*wp/n_train:.1f}%) {el/60:.1f}min elapsed")
assert wp == n_train, (wp, n_train)
mm.flush()
del mm
f.close()
json.dump(
    {"path": OUT, "n_train": n_train, "shape": [n_train, 3, 1024], "dtype": "float16"},
    open(META, "w"),
)
log("DONE", OUT, "n_train", n_train)
