#!/usr/bin/env python3
"""Permutation null for region detectors: is the # of region-detector latents more than
chance under label shuffling? Encode antibody test residues through cc_paired_L20, collect
sparse (latent, activation, region) triples, compute OBSERVED detector counts (exact sweep,
Haque floor), then P permutations of the region labels (sizes preserved -> floors unchanged).
Writes region_permutation_null.json."""

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


REGS = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
NR = 7
NPERM = 50
MARGIN = 0.2
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_paired_L20.pt", device=dev, dtype=th.float32)
ae.eval()
DICT = 32768
h5 = f"{ACTS}/full_paired_L20.h5"
with h5py.File(h5, "r") as h:
    split = h["split"][:]
    region = h["region_label"][:]
    N = split.shape[0]
keep = (split == 2) & (region >= 0) & (region < 7)
idx = np.nonzero(keep)[0]
reg = region[idx].astype(np.int8)
Nt = len(idx)
log(f"test residues with region label: {Nt}")
Pr = np.array([(reg == r).sum() for r in range(NR)], dtype=np.int64)
prev = Pr / Nt
floor = 2 * prev / (1 + prev) + MARGIN
log(f"prevalence {dict(zip(REGS,np.round(prev,4)))}")
log(f"floor {dict(zip(REGS,np.round(floor,3)))}")

# --- encode: collect nonzero (latent, activation) with residue index ---
lat_b = []
act_b = []
res_b = []
BLK = 200000
ENC = 16384
t0 = time.time()
tot = 0
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    for s in range(0, Nt, BLK):
        e = min(s + BLK, Nt)
        gi = idx[s:e]
        x = acts[gi.min() : gi.max() + 1]
        take = gi - gi.min()
        xb_all = x[take]
        for b0 in range(0, e - s, ENC):
            b1 = min(b0 + ENC, e - s)
            xb = th.tensor(xb_all[b0:b1].astype(np.float32), device=dev)
            with th.no_grad():
                f = ae.encode(xb, use_threshold=True, normalize_activations=True)
            nz = f.nonzero(as_tuple=True)
            ri, li = nz
            lat_b.append(li.to(th.int32).cpu().numpy())
            act_b.append(f[ri, li].cpu().numpy())
            res_b.append((s + b0 + ri).cpu().numpy().astype(np.int64))
            tot += len(ri)
        if (s // BLK) % 10 == 0:
            log(f"  encode {e}/{Nt} nz={tot} {time.time()-t0:.0f}s")
lat = np.concatenate(lat_b)
act = np.concatenate(act_b)
res = np.concatenate(res_b)
del lat_b, act_b, res_b
log(f"nonzeros {len(lat)} ({len(lat)/Nt:.1f}/res)")

# group by latent, sort within latent by activation desc; store residue index order
order = np.lexsort((-act, lat))  # sort by latent, then act desc within latent
lat_s = lat[order]
res_s = res[order]
la_u, la_first, la_cnt = np.unique(lat_s, return_index=True, return_counts=True)


def count_detectors(region_vec):
    """max-F1 per (latent,region) via exact sweep on precomputed per-latent residue order; return per-region detector count and best-F1."""
    ndet = np.zeros(NR, dtype=np.int64)
    bestf1 = np.zeros(NR)
    for L, f0, c in zip(la_u, la_first, la_cnt):
        r_res = res_s[f0 : f0 + c]  # residues of latent L, already sorted by act desc
        rlab = region_vec[r_res]  # their region labels
        # membership matrix (c,NR)
        Ymem = rlab[:, None] == np.arange(NR)[None, :]
        tp = np.cumsum(Ymem, axis=0).astype(np.float64)  # (c,NR)
        pp = np.arange(1, c + 1, dtype=np.float64)[:, None]
        prec = tp / pp
        rec = tp / np.maximum(Pr[None, :], 1)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
        mf = f1.max(0)  # (NR,)
        for r in range(NR):
            if mf[r] > bestf1[r]:
                bestf1[r] = mf[r]
            if mf[r] > floor[r]:
                ndet[r] += 1
    return ndet, bestf1


t0 = time.time()
obs_ndet, obs_best = count_detectors(reg.astype(np.int64))
log(f"OBSERVED detectors {dict(zip(REGS,obs_ndet.tolist()))} in {time.time()-t0:.0f}s")

# --- permutation null ---
rng = np.random.default_rng(0)
null_counts = np.zeros((NPERM, NR), dtype=np.int64)
for p in range(NPERM):
    perm = reg.copy()
    rng.shuffle(perm)
    nd, _ = count_detectors(perm.astype(np.int64))
    null_counts[p] = nd
    if p % 10 == 0:
        log(f"  perm {p}/{NPERM} {time.time()-t0:.0f}s")
out = {
    "regions": REGS,
    "prevalence": prev.tolist(),
    "floor": floor.tolist(),
    "observed_detectors": obs_ndet.tolist(),
    "observed_best_f1": obs_best.tolist(),
    "null_mean": null_counts.mean(0).tolist(),
    "null_max": null_counts.max(0).tolist(),
    "null_p95": np.percentile(null_counts, 95, axis=0).tolist(),
    "null_counts": null_counts.tolist(),
    "n_perm": NPERM,
    "p_value": [
        float(((null_counts[:, r] >= obs_ndet[r]).sum() + 1) / (NPERM + 1)) for r in range(NR)
    ],
}
json.dump(out, open(f"{RES}/region_permutation_null.json", "w"))
log("DONE")
print(
    json.dumps(
        {k: out[k] for k in ["observed_detectors", "null_mean", "null_max", "p_value"]}, indent=1
    )
)
