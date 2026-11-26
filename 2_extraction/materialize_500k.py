import os, json, time, numpy as np, h5py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE, SCRATCH

H5 = (
    f"{BASE}/activations/scaleup500k/scaleup500k_paired_L20.h5"
)
OUT = f"{SCRATCH}/scale500k_train_contig.f16"
META = f"{SCRATCH}/scale500k_train_contig.meta.json"


def log(*a):
    print("[" + time.strftime("%H:%M:%S") + "]", *a, flush=True)


f = h5py.File(H5, "r")
split = f["split"][:]
acts = f["acts"]
N = acts.shape[0]
n_train = int((split == 0).sum())
log("N", N, "n_train", n_train, "GB", round(n_train * 3 * 1024 * 2 / 1e9, 1))
mm = np.memmap(OUT, dtype=np.float16, mode="w+", shape=(n_train, 3, 1024))
BLK = 524288
wp = 0
t0 = time.time()
for s in range(0, N, BLK):
    e = min(s + BLK, N)
    m = split[s:e] == 0
    k = int(m.sum())
    if k == 0:
        continue
    mm[wp : wp + k] = acts[s:e][m]
    wp += k
    if (s // BLK) % 20 == 0:
        log(
            str(wp)
            + "/"
            + str(n_train)
            + " "
            + str(round(100 * wp / n_train, 1))
            + "% "
            + str(round((time.time() - t0) / 60, 1))
            + "min"
        )
assert wp == n_train, (wp, n_train)
mm.flush()
del mm
f.close()
json.dump({"path": OUT, "n_train": n_train, "shape": [n_train, 3, 1024]}, open(META, "w"))
log("DONE", OUT, n_train)
