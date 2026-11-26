"""Isotype detector latents: does any crosscoder latent fire specifically on a heavy-chain
isotype (IgG/IgM/IgA/...)? Encodes the paired store, aggregates per-sequence latent scores,
and runs the one-vs-rest max-F1 / Haque-floor detection test per isotype. Writes isotype_perlatent.parquet.
"""

import os, sys, json, time
import numpy as np, h5py, torch as th, pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

RES = f"{BASE}/results"
ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
CODE = f"{BASE}/code/dictionary_learning_jkminder"
sys.path.insert(0, CODE)
LABELS = f"{BASE}/data/oas_full_annotated.parquet"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


dev = "cuda" if th.cuda.is_available() else "cpu"
log("device", dev)
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_paired_L20.pt", device=dev, dtype=th.float32)
ae.eval()
cls = pd.read_csv(f"{RES}/classification_paired_L20.csv")
active_ids = cls["latent_id"].to_numpy().astype(np.int64)
A = len(active_ids)
category = cls["category"].to_numpy()
active_t = th.tensor(active_ids, device=dev)
with h5py.File(f"{ACTS}/full_paired_L20.h5", "r") as h:
    split = h["split"][:]
    sid = h["seq_id"][:].astype(np.int64)
    N = split.shape[0]
test = split == 2
test_seq = np.unique(sid[test])
n_seq = len(test_seq)
seq2c = -np.ones(int(sid.max()) + 1, dtype=np.int64)
seq2c[test_seq] = np.arange(n_seq)
log(f"active={A} test_seqs={n_seq}")
# isotype labels
df = pd.read_parquet(LABELS, columns=["Isotype"])
iso = df["Isotype"].astype("object").to_numpy()[test_seq]
classes = ["IGHM", "IGHA", "IGHG", "IGHD"]
Y = {c: (iso == c) for c in classes}
# sequence-level mean-of-nonzero aggregation (paired encode), also track activation
S = th.zeros(n_seq, A, device=dev)
C = th.zeros(n_seq, A, device=dev)
Nt = 0
BLK = 262144
ENC = 8192
t0 = time.time()
with h5py.File(f"{ACTS}/full_paired_L20.h5", "r") as h:
    acts = h["acts"]
    for s in range(0, N, BLK):
        e = min(s + BLK, N)
        m = test[s:e]
        if not m.any():
            continue
        loc = np.nonzero(m)[0]
        blk = acts[s:e]
        sub = blk[loc]
        comp = seq2c[sid[s:e][loc]]
        comp_t = th.tensor(comp, device=dev)
        for b0 in range(0, sub.shape[0], ENC):
            b1 = min(b0 + ENC, sub.shape[0])
            x = th.tensor(sub[b0:b1].astype(np.float32), device=dev)
            with th.no_grad():
                f = ae.encode(x, use_threshold=True, normalize_activations=True)
            cd = f.index_select(1, active_t)
            cf = cd.float()
            nz = (cd > 0).float()
            cc = comp_t[b0:b1]
            S.index_add_(0, cc, cf)
            C.index_add_(0, cc, nz)
            Nt += b1 - b0
score = S / th.clamp(C, min=1.0)  # (n_seq, A) seq-level mean activation when firing
log(f"aggregated {Nt} residues {time.time()-t0:.0f}s")
# per-latent F1 for each isotype (threshold sweep over seq scores)
sc = score
order = th.argsort(sc, 0, descending=True)
ns = order.shape[0]
cumP = th.arange(1, ns + 1, device=dev, dtype=th.float32).unsqueeze(1)
perlat = {"latent_id": active_ids, "category": category}
mean_act_pos = {}
mean_act_neg = {}
for c in classes:
    y = Y[c]
    P = int(y.sum())
    pi = P / ns
    yt = th.tensor(y.astype(np.float32), device=dev)
    ys = yt[order]
    cum = th.cumsum(ys, 0)
    pr = cum / cumP
    rc = cum / P
    f1 = 2 * pr * rc / th.clamp(pr + rc, min=1e-9)
    fmax = f1.max(0).values.cpu().numpy()
    perlat[f"F1_{c}"] = fmax
    # mean seq-activation of each latent on positive vs negative seqs
    pos = sc[th.tensor(y, device=dev)].mean(0).cpu().numpy()
    neg = sc[th.tensor(~y, device=dev)].mean(0).cpu().numpy()
    perlat[f"meanact_pos_{c}"] = pos
    perlat[f"meanact_neg_{c}"] = neg
    log(f"{c} pi={pi:.3f} bestF1={fmax.max():.3f}")
pdf = pd.DataFrame(perlat)
pdf.to_parquet(f"{RES}/isotype_perlatent.parquet")
# gained/shared numerical breakdown
is_shared = np.array([str(x) == "shared-all" for x in category])
is_gained = np.array([str(x).startswith("gained") for x in category])
out = {}
for c in classes:
    f1 = pdf[f"F1_{c}"].to_numpy()
    for thr in [0.3, 0.5]:
        det = f1 > thr
        out.setdefault(c, {})[f"n_detect_gt{thr}"] = int(det.sum())
        out[c][f"n_gained_gt{thr}"] = int((det & is_gained).sum())
        out[c][f"n_shared_gt{thr}"] = int((det & is_shared).sum())
    out[c]["best_F1"] = float(f1.max())
    out[c]["best_is_gained"] = bool(is_gained[f1.argmax()])
    out[c]["mean_F1_gained"] = float(f1[is_gained].mean())
    out[c]["mean_F1_shared"] = float(f1[is_shared].mean())
    # mean activation on positive seqs, gained vs shared latents (top detectors)
    pos = pdf[f"meanact_pos_{c}"].to_numpy()
    top = f1 > 0.3
    out[c]["meanact_pos_gained_detectors"] = (
        float(pos[top & is_gained].mean()) if (top & is_gained).any() else 0.0
    )
    out[c]["meanact_pos_shared_detectors"] = (
        float(pos[top & is_shared].mean()) if (top & is_shared).any() else 0.0
    )
out["_meta"] = {"n_active": A, "n_gained": int(is_gained.sum()), "n_shared": int(is_shared.sum())}
json.dump(out, open(f"{RES}/isotype_gained_shared.json", "w"), indent=2)
log("WROTE isotype_perlatent.parquet + isotype_gained_shared.json")
log("SUMMARY " + json.dumps(out))
