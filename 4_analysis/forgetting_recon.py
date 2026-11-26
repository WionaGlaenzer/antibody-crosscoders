#!/usr/bin/env python3
"""Per-concept reconstruction 'forgetting': pass general proteins through the mixed
crosscoder (shared code), reconstruct each model's activations, and compute per-Swiss-Prot-
concept variance-explained for the ProtBert (base) readout vs the IgBert (fine-tuned) readouts.
A concept is 'forgotten' if VE_ProtBert(concept) >> VE_IgBert(concept). Writes forgetting_recon.json.
"""

import json, time, numpy as np, h5py, torch as th

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
import sys

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
import os

os.environ["HF_HOME"] = f"{BASE}/caches/hf"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_mixed_L20.pt", device=dev, dtype=th.float32)
ae.eval()
# concept labels (per-residue masks for general proteins) - same as concept_f1_folds_rebuild95v2
CL = json.load(open(f"{RES}/concept_labels.json"))
CONCEPTS = CL["concepts"]
NC = len(CONCEPTS)
acc_to_row = CL["acc_to_row"]
labels = CL["labels"]
row_labels = {}
for acc, d in labels.items():
    row = acc_to_row.get(acc)
    if row is None:
        continue
    row_labels[row] = {int(ci): set(p - 1 for p in poss) for ci, poss in d.items()}
h5 = f"{ACTS}/general_single_L20.h5"
with h5py.File(h5, "r") as h:
    sid = h["seq_id"][:].astype(np.int64)
    N = sid.shape[0]
u, first, cnt = np.unique(sid, return_index=True, return_counts=True)
# per-residue concept bitmask
cmask = np.zeros(N, dtype=np.int32)
for f0, c, s in zip(first, cnt, u):
    rl = row_labels.get(int(s))
    if not rl:
        continue
    for ci, poss in rl.items():
        bit = 1 << ci
        for p in poss:
            if 0 <= p < c:
                cmask[f0 + p] |= bit
log(f"N={N} residues, labeled proteins={len(row_labels)}")
# accumulators: per concept (+overall), per model: sum sq err, sum var
NCp = NC + 1  # last = overall (all residues)
sse = np.zeros((NCp, 3))
ssv = np.zeros((NCp, 3))
cnts = np.zeros(NCp)
bits = np.array([1 << i for i in range(NC)], dtype=np.int32)
# global per-model mean for variance (normalized space -> mean ~0, but compute properly)
ENC = 16384
t0 = time.time()
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    # first pass: per-model mean in normalized space (concept-agnostic, use overall)
    # normalized activations have ~0 mean by construction; variance vs per-batch mean is fine.
    for s in range(0, N, ENC):
        e = min(s + ENC, N)
        xb = th.tensor(acts[s:e].astype(np.float32), device=dev)
        with th.no_grad():
            xn = ae.normalize_activations(xb, inplace=False)
            f = ae.encode(xn, normalize_activations=False, use_threshold=True)
            xhat = ae.decode(f, denormalize_activations=False)
        err = ((xn - xhat) ** 2).sum(-1).cpu().numpy()  # (b,3)
        var = ((xn - xn.mean(0, keepdim=True)) ** 2).sum(-1).cpu().numpy()  # (b,3)
        cmb = cmask[s:e]
        # overall
        sse[NC] += err.sum(0)
        ssv[NC] += var.sum(0)
        cnts[NC] += e - s
        for ci in range(NC):
            mm = (cmb & bits[ci]) > 0
            if mm.any():
                sse[ci] += err[mm].sum(0)
                ssv[ci] += var[mm].sum(0)
                cnts[ci] += int(mm.sum())
        if (s // ENC) % 40 == 0:
            log(f"  {e}/{N} {time.time()-t0:.0f}s")
ve = 1.0 - sse / np.maximum(ssv, 1e-9)  # (NCp,3)
MODELS = ["prot_bert", "igbert_unpaired", "igbert_paired"]
out = {"concepts": CONCEPTS + ["OVERALL"], "models": MODELS, "n": cnts.tolist(), "ve": {}}
for i, cn in enumerate(CONCEPTS + ["OVERALL"]):
    out["ve"][cn] = {MODELS[m]: float(ve[i, m]) for m in range(3)}
    out["ve"][cn]["forgetting_prot_minus_igunp"] = float(ve[i, 0] - ve[i, 1])
json.dump(out, open(f"{RES}/forgetting_recon.json", "w"), indent=1)
log("DONE")
print(json.dumps(out["ve"], indent=1))
