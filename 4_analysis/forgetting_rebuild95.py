#!/usr/bin/env python3
"""Forgetting analysis on the REBUILT MIXED crosscoder (cc_mixed_small_s0_L20).
(A) Per-latent decoder-norm forgetting: identify base-only latents (ProtBert-retained,
    IgBert-dropped: rel_prot>0.5 & rel_unp<0.2 & rel_pair<0.2) from the mixed decoder norms,
    report count + relative-norm stats.
(B) Concept-level reconstruction retained-vs-forgotten: pass general proteins through the
    mixed crosscoder, reconstruct each model's activations, per-Swiss-Prot-concept VE for
    ProtBert (base) vs IgBert readouts. A concept is 'forgotten' if VE_ProtBert >> VE_IgBert.
Adapted from forgetting_recon.py, repointed at exp_mixed_small_L20.h5 general rows (kind==1,
seq_id-100000 = concept-pool row). Writes forgetting_rebuild95.json to D/results/."""

import json, time, numpy as np, h5py, torch as th

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
import os

os.environ["HF_HOME"] = f"{BASE}/caches/hf"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(
    f"{CKPT}/cc_mixed_small_s0_L20.pt", device=dev, dtype=th.float32
)
ae.eval()
DICT = ae.dict_size
# ---- (A) per-latent decoder-norm forgetting ----
Wdec = ae.decoder.weight.detach()  # (3, DICT, 1024): [prot, unp, pair]
n = th.linalg.norm(Wdec, dim=2).transpose(0, 1)  # (DICT,3)
s = n.sum(dim=1, keepdim=True)
rel = (n / (s + 1e-12)).cpu().numpy()
rp, ru, rpr = rel[:, 0], rel[:, 1], rel[:, 2]
HI, LO = 0.5, 0.2
base_only = (rp > HI) & (ru < LO) & (rpr < LO)
gain_unp = (rp < LO) & (ru > LO) & (rpr > LO)
shared = (rp > LO) & (ru > LO) & (rpr > LO)
partA = {
    "n_base_only": int(base_only.sum()),
    "n_gained_at_unpaired": int(gain_unp.sum()),
    "n_shared_all": int(shared.sum()),
    "base_only_rel_prot_mean": float(rp[base_only].mean()) if base_only.any() else None,
    "base_only_rel_igunp_mean": float(ru[base_only].mean()) if base_only.any() else None,
    "base_only_rel_igpair_mean": float(rpr[base_only].mean()) if base_only.any() else None,
}
log(f"(A) base-only={partA['n_base_only']} gained-unp={partA['n_gained_at_unpaired']}")
# ---- (B) concept-level reconstruction VE ----
CL = json.load(open(f"{RES}/concept_labels.json"))
CONCEPTS = CL["concepts"]
NC = len(CONCEPTS)
acc_to_row = CL["acc_to_row"]
labels = CL["labels"]
row_labels = {}
for acc, d in labels.items():
    r = acc_to_row.get(acc)
    if r is None:
        continue
    row_labels[r] = {int(ci): set(p - 1 for p in poss) for ci, poss in d.items()}
h5 = f"{ACTS}/exp_mixed_small_L20.h5"
with h5py.File(h5, "r") as h:
    sid = h["seq_id"][:].astype(np.int64)
    kind = h["kind"][:]
    N = sid.shape[0]
u, first, cnt = np.unique(sid, return_index=True, return_counts=True)
cmask = np.zeros(N, dtype=np.int32)
bits = np.array([1 << i for i in range(NC)], dtype=np.int32)
for f0, c, sq in zip(first, cnt, u):
    rl = row_labels.get(int(sq) - 100000)
    if not rl:
        continue
    for ci, poss in rl.items():
        bit = 1 << ci
        for p in poss:
            if 0 <= p < c:
                cmask[f0 + p] |= bit
gen = kind == 1
log(f"(B) general residues={int(gen.sum())} labeled proteins present")
NCp = NC + 1
sse = np.zeros((NCp, 3))
ssv = np.zeros((NCp, 3))
cnts = np.zeros(NCp)
ENC = 16384
t0 = time.time()
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    for s0 in range(0, N, ENC):
        e = min(s0 + ENC, N)
        gmask_blk = gen[s0:e]
        if not gmask_blk.any():
            continue
        xb = th.tensor(acts[s0:e][gmask_blk].astype(np.float32), device=dev)
        with th.no_grad():
            xn = ae.normalize_activations(xb, inplace=False)
            f = ae.encode(xn, normalize_activations=False, use_threshold=True)
            xhat = ae.decode(f, denormalize_activations=False)
        err = ((xn - xhat) ** 2).sum(-1).cpu().numpy()
        var = ((xn - xn.mean(0, keepdim=True)) ** 2).sum(-1).cpu().numpy()
        cmb = cmask[s0:e][gmask_blk]
        sse[NC] += err.sum(0)
        ssv[NC] += var.sum(0)
        cnts[NC] += err.shape[0]
        for ci in range(NC):
            mm = (cmb & bits[ci]) > 0
            if mm.any():
                sse[ci] += err[mm].sum(0)
                ssv[ci] += var[mm].sum(0)
                cnts[ci] += int(mm.sum())
        if (s0 // ENC) % 40 == 0:
            log(f"  {e}/{N} {time.time()-t0:.0f}s")
ve = 1.0 - sse / np.maximum(ssv, 1e-9)
MODELS = ["prot_bert", "igbert_unpaired", "igbert_paired"]
partB = {"concepts": CONCEPTS + ["OVERALL"], "models": MODELS, "n": cnts.tolist(), "ve": {}}
for i, cn in enumerate(CONCEPTS + ["OVERALL"]):
    partB["ve"][cn] = {MODELS[m]: float(ve[i, m]) for m in range(3)}
    partB["ve"][cn]["forgetting_prot_minus_igunp"] = float(ve[i, 0] - ve[i, 1])
# overall forgetting summary
overall = partB["ve"]["OVERALL"]
out = {
    "checkpoint": "cc_mixed_small_s0_L20.pt",
    "store": "exp_mixed_small_L20.h5",
    "decoder_norm_forgetting": partA,
    "reconstruction_ve": partB,
    "summary": {
        "overall_ve_prot_bert": overall["prot_bert"],
        "overall_ve_igbert_unpaired": overall["igbert_unpaired"],
        "overall_ve_igbert_paired": overall["igbert_paired"],
        "overall_forgetting_prot_minus_igunp": overall["forgetting_prot_minus_igunp"],
        "n_base_only_latents": partA["n_base_only"],
    },
}
json.dump(out, open(f"{RES}/forgetting_rebuild95.json", "w"), indent=1)
log("DONE")
print(json.dumps(out["summary"], indent=1))
print(json.dumps(partB["ve"], indent=1))
