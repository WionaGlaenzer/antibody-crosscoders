#!/usr/bin/env python3
"""Per-residue activation of each region's TOP detector latent along one example heavy chain.
Picks a heavy chain (all 7 regions present, len 110-140) where the region detectors fire, returns
per-residue traces + region labels. Writes region_toplat_profile.json."""

import json, time, numpy as np, h5py, torch as th

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_paired_L20.pt", device=dev, dtype=th.float32)
ae.eval()
# region -> top latent per region (all 7; FR1/FR2/FR3 are sub-threshold, included for curiosity)
TOPLAT = {
    "FR1": 4548,
    "CDR1": 7245,
    "FR2": 28844,
    "CDR2": 11364,
    "FR3": 4548,
    "CDR3": 28462,
    "FR4": 2440,
}
REGION_MAP = {"FR1": 0, "CDR1": 1, "FR2": 2, "CDR2": 3, "FR3": 4, "CDR3": 5, "FR4": 6}
INV = {v: k for k, v in REGION_MAP.items()}
lats = sorted(set(TOPLAT.values()))
h5 = f"{ACTS}/full_paired_L20.h5"
with h5py.File(h5, "r") as h:
    split = h["split"][:]
    seqid = h["seq_id"][:]
    region = h["region_label"][:]
    chain = h["chain"][:]
# chain stored as byte strings b'H'/b'L'
is_h = chain == b"H"
mask = (split == 2) & is_h
sids = np.unique(seqid[mask])
# find a seq with all 7 heavy regions present and length 110-140
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
        # encode this chain
        x = acts[gi.min() : gi.max() + 1]
        take = gi - gi.min()
        xb = th.tensor(x[take].astype(np.float32), device=dev)
        with th.no_grad():
            f = ae.encode(xb, use_threshold=True, normalize_activations=True)
        prof = {r: f[:, l].cpu().numpy().tolist() for r, l in TOPLAT.items()}
        # score: sum of max activations across the 4 latents (want a chain where they all fire)
        sc = sum(max(prof[r]) for r in TOPLAT)
        if best is None or sc > best["score"]:
            best = {
                "seq_id": int(sid),
                "len": int(len(gi)),
                "regions": [INV[int(v)] for v in rl],
                "profiles": prof,
                "score": float(sc),
            }
        if best and best["score"] > 2.0:
            pass
log(f"picked seq {best['seq_id']} len {best['len']} score {best['score']:.2f}")
json.dump({"toplat": TOPLAT, "example": best}, open(f"{RES}/region_toplat_profile.json", "w"))
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
