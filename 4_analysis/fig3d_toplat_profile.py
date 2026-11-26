#!/usr/bin/env python3
"""Fig3d rebuild95: per-residue activation of each detected region's TOP detector latent
along one example heavy chain. Adapts orig region_toplat_profile.py to rebuild95:
 - store exp_paired_small_L20.h5 (has region_label + chain), checkpoint cc_paired_small_s0_L20.pt
 - encode use_threshold=True normalize_activations=True (matches original region methodology)
 - TOPLAT from verified fig3reuse_s0 (top-F1 detector per detected region)
Writes region_toplat_profile_rebuild95.json."""

import os, sys, json, time, numpy as np, h5py, torch as th

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


from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(
    f"{CKPT}/cc_paired_small_s0_L20.pt", device=dev, dtype=th.float32
)
ae.eval()
TOPLAT = {"CDR1": 7245, "CDR2": 22804, "CDR3": 28462, "FR4": 15752}
REGION_MAP = {"FR1": 0, "CDR1": 1, "FR2": 2, "CDR2": 3, "FR3": 4, "CDR3": 5, "FR4": 6}
INV = {v: k for k, v in REGION_MAP.items()}
h5 = f"{ACTS}/exp_paired_small_L20.h5"
with h5py.File(h5, "r") as h:
    split = h["split"][:]
    seqid = h["seq_id"][:]
    region = h["region_label"][:]
    chain = h["chain"][:]
    tmean = th.tensor(h["train_mean"][:], device=dev, dtype=th.float32)
    tstd = th.tensor(h["train_std"][:], device=dev, dtype=th.float32)
# chain stored as byte strings b'H'/b'L'
is_h = (
    (chain == b"H")
    if chain.dtype.kind == "S"
    else (chain == ord("H")) if chain.dtype.kind in "iu" else (chain == b"H")
)
log("is_h sum", int(is_h.sum()), "chain dtype", str(chain.dtype))
mask = (split == 2) & is_h
sids = np.unique(seqid[mask])
best = None
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    for sid in sids[:4000]:
        gi = np.nonzero((seqid == sid) & is_h & (split == 2))[0]
        if len(gi) < 110 or len(gi) > 140:
            continue
        rl = region[gi]
        if len(set(rl.tolist())) < 7:
            continue
        x = acts[gi.min() : gi.max() + 1]
        take = gi - gi.min()
        xb = th.tensor(x[take].astype(np.float32), device=dev)
        xn = (xb - tmean) / tstd
        with th.no_grad():
            f = ae.encode(xn, use_threshold=True, normalize_activations=False)
        prof = {r: f[:, l].cpu().numpy().tolist() for r, l in TOPLAT.items()}
        sc = sum(max(prof[r]) for r in TOPLAT)
        if best is None or sc > best["score"]:
            best = {
                "seq_id": int(sid),
                "len": int(len(gi)),
                "regions": [INV[int(v)] for v in rl],
                "profiles": prof,
                "score": float(sc),
            }
log(f"picked seq {best['seq_id']} len {best['len']} score {best['score']:.2f}")
json.dump({"toplat": TOPLAT, "example": best}, open("region_toplat_profile_rebuild95.json", "w"))
log("DONE")
print(
    json.dumps(
        {
            "seq_id": best["seq_id"],
            "len": best["len"],
            "peak": {r: round(max(best["profiles"][r]), 3) for r in TOPLAT},
        },
        indent=1,
    )
)
