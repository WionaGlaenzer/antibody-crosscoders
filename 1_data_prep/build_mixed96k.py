"""Build balanced mixed antibody+general activation store (single regime, L20).
FAST version: sequential block-streaming with a precomputed selection mask (no
fancy-indexing of scattered rows). ~3.3M antibody + ~3.3M general train residues
(50/50), plus proportional val/test. Recomputes mean/std over mixed train.
kind label 0=antibody 1=general; seq_id namespaced (+100000 for general)."""

import time, json, numpy as np, h5py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

ACTS = f"{BASE}/activations"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


AB = f"{ACTS}/ab96k/ab96k_single_L20.h5"
GN = f"{ACTS}/general_96k_single_L20.h5"
OUT = f"{ACTS}/mixed96k_single_L20.h5"
HID = 1024
CAPS = {0: 15_600_000, 1: 3_300_000, 2: 3_300_000}
rng = np.random.default_rng(0)


def build_selection(h5):
    """Return a per-row int8 array: -1 skip, else split value (0/1/2), capped per split."""
    with h5py.File(h5, "r") as f:
        sp = f["split"][:]
    sel = np.full(sp.shape[0], -1, dtype=np.int8)
    for sv in (0, 1, 2):
        idx = np.nonzero(sp == sv)[0]
        if len(idx) > CAPS[sv]:
            idx = rng.choice(idx, CAPS[sv], replace=False)
        sel[idx] = sv
    return sel


selAB = build_selection(AB)
selGN = build_selection(GN)
totAB = int((selAB >= 0).sum())
totGN = int((selGN >= 0).sum())
total = totAB + totGN
log(f"selected antibody={totAB} general={totGN} total={total}")
hf = h5py.File(OUT, "w")
acts_ds = hf.create_dataset("acts", shape=(total, 3, HID), dtype=np.float16, chunks=(8192, 3, HID))
split_ds = hf.create_dataset("split", shape=(total,), dtype=np.int8)
kind_ds = hf.create_dataset("kind", shape=(total,), dtype=np.int8)
sid_ds = hf.create_dataset("seq_id", shape=(total,), dtype=np.int32)
acc_sum = np.zeros((3, HID), np.float64)
acc_sq = np.zeros((3, HID), np.float64)
acc_cnt = 0
wp = 0
BLK = 262144
for h5, dom, sel, off in [(AB, 0, selAB, 0), (GN, 1, selGN, 100000)]:
    t0 = time.time()
    with h5py.File(h5, "r") as f:
        acts = f["acts"]
        sid = f["seq_id"][:]
        N = acts.shape[0]
        for start in range(0, N, BLK):
            stop = min(start + BLK, N)
            bsel = sel[start:stop]
            m = bsel >= 0
            k = int(m.sum())
            if k == 0:
                continue
            chunk = acts[start:stop][m]  # sequential read then in-memory mask
            acts_ds[wp : wp + k] = chunk
            split_ds[wp : wp + k] = bsel[m]
            kind_ds[wp : wp + k] = dom
            sid_ds[wp : wp + k] = sid[start:stop][m].astype(np.int32) + off
            tr = bsel[m] == 0
            if tr.any():
                xf = chunk[tr].astype(np.float64)
                acc_sum += xf.sum(0)
                acc_sq += (xf**2).sum(0)
                acc_cnt += int(tr.sum())
            wp += k
            if (start // BLK) % 20 == 0:
                log(f"  dom{dom} {wp}/{total} ({time.time()-t0:.0f}s)")
    log(f"finished dom{dom}, wp={wp}")
assert wp == total, (wp, total)
tmean = (acc_sum / acc_cnt).astype(np.float32)
tstd = np.sqrt(np.clip(acc_sq / acc_cnt - (acc_sum / acc_cnt) ** 2, 1e-8, None)).astype(np.float32)
hf.create_dataset("train_mean", data=tmean)
hf.create_dataset("train_std", data=tstd)
hf.attrs["model_order"] = json.dumps(["prot_bert", "igbert_unpaired", "igbert_paired"])
hf.attrs["n_residues"] = total
hf.attrs["train_residues"] = acc_cnt
hf.attrs["balance"] = "50/50"
hf.close()
log(f"DONE {OUT} total={total} train={acc_cnt} mean_norm={np.linalg.norm(tmean):.2f}")
