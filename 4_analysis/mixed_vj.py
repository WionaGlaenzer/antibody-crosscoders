#!/usr/bin/env python
"""Does mixing general proteins into training change the % of V/J-gene detector latents?
Compute germline V/J detection for cc_single_L20 (antibody-only) vs cc_mixed_L20 (general+antibody),
BOTH evaluated on the SAME antibody TEST heavy residues (full_single_L20.h5) so only the dictionary differs.
Detection = per-latent, per-concept max-F1 over threshold sweep > 2*pi/(1+pi)+0.2 (the pipeline criterion).
Report % of active latents detecting any V / any J, overall and by decoder-norm category.
"""

import h5py, numpy as np, pandas as pd, torch as th, json, sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

DEV = "cuda" if th.cuda.is_available() else "cpu"
sys.path.insert(0, BASE + "/code/dictionary_learning_jkminder")
from dictionary_learning.dictionary import BatchTopKCrossCoder


def log(*a):
    print(*a, flush=True)


def crit(pi):
    return 2 * pi / (1 + pi) + 0.2


H5 = BASE + "/activations/full_single_L20.h5"
# --- metadata (shared across both models) ---
with h5py.File(H5, "r") as h:
    split = h["split"][:]
    chain = h["chain"][:]
    seqid = h["seq_id"][:].astype(np.int64)
    N = split.shape[0]
test = split == 2
test_seq = np.unique(seqid[test])
n_seq = len(test_seq)
seq2c = -np.ones(int(seqid.max()) + 1, dtype=np.int64)
seq2c[test_seq] = np.arange(n_seq)
log(f"test residues {int(test.sum())}  test seqs {n_seq}")

# --- V/J labels per test sequence (same columns as pipeline: v_family_heavy, j_call_heavy) ---
df = pd.read_parquet(BASE + "/data/oas_full_annotated.parquet")
vf = df["v_family_heavy"].to_numpy()
jf = df["j_call_heavy"].to_numpy()
Vfams = [f"IGHV{i}" for i in range(1, 8)]
Jfams = [f"IGHJ{i}" for i in range(1, 7)]
memb = {}
for f in Vfams:
    memb[("V", f)] = vf[test_seq] == f
for f in Jfams:
    memb[("J", f)] = jf[test_seq] == f


def max_f1(score, y, min_pos=15):
    P = int(y.sum())
    pi = P / len(y)
    if P < min_pos:
        return None, pi
    sc = th.tensor(score, device=DEV)
    order = th.argsort(sc, dim=0, descending=True)
    del sc
    yt = th.tensor(y.astype(np.float32), device=DEV)[order]
    cumTP = th.cumsum(yt, 0)
    cumP = th.arange(1, len(y) + 1, device=DEV, dtype=th.float32).unsqueeze(1)
    prec = cumTP / cumP
    rec = cumTP / P
    f1 = (2 * prec * rec / th.clamp(prec + rec, min=1e-9)).max(0).values.cpu().numpy()
    del order, yt, cumTP, prec, rec
    th.cuda.empty_cache()
    return f1, pi


def run(ckpt, cls_csv, tag):
    log("==== ", tag, " ====")
    ae = BatchTopKCrossCoder.from_pretrained(ckpt, device=DEV)
    ae.eval()
    cls = pd.read_csv(cls_csv)
    active = cls["latent_id"].to_numpy().astype(np.int64)
    A = len(active)
    cat = cls["category"].to_numpy()
    active_t = th.tensor(active, device=DEV)
    # per-seq mean-of-nonzero on HEAVY test residues
    sum_H = th.zeros(n_seq, A, device=DEV, dtype=th.float32)
    cnt_H = th.zeros(n_seq, A, device=DEV, dtype=th.float32)
    BLK = 200000
    ENC = 8192
    with h5py.File(H5, "r") as h:
        acts = h["acts"]
        for s in range(0, N, BLK):
            e = min(s + BLK, N)
            tm = test[s:e] & (chain[s:e] == b"H")
            loc = np.nonzero(tm)[0]
            if len(loc) == 0:
                continue
            blk = acts[s:e]
            sub = blk[loc]
            comp = th.tensor(seq2c[seqid[s:e][loc]], device=DEV)
            for b0 in range(0, len(loc), ENC):
                b1 = min(b0 + ENC, len(loc))
                x = th.tensor(sub[b0:b1].astype(np.float32), device=DEV)
                with th.no_grad():
                    f = ae.encode(x, use_threshold=True, normalize_activations=True).index_select(
                        1, active_t
                    )
                cp = comp[b0:b1]
                nz = (f > 0).float()
                sum_H.index_add_(0, cp, f.float() * nz)
                cnt_H.index_add_(0, cp, nz)
            if s % (BLK * 10) == 0:
                log(f"  {tag} {s}/{N}")
    scoreH = (sum_H / th.clamp(cnt_H, min=1.0)).cpu().numpy()
    del sum_H, cnt_H
    th.cuda.empty_cache()
    # detection
    detV = np.zeros(A, bool)
    detJ = np.zeros(A, bool)
    for (kind, f), y in memb.items():
        f1, pi = max_f1(scoreH, y)
        if f1 is None:
            continue
        d = f1 > crit(pi)
        if kind == "V":
            detV |= d
        else:
            detJ |= d
    del ae, scoreH
    th.cuda.empty_cache()

    # summarize overall + by category
    def rate(mask, sub=None):
        m = mask if sub is None else (mask & sub)
        base = A if sub is None else int(sub.sum())
        return 100 * m.sum() / max(base, 1)

    res = {"tag": tag, "n_active": int(A), "pctV_all": rate(detV), "pctJ_all": rate(detJ)}
    for c in ["shared-all", "gained-at-unpaired", "base-only"]:
        sub = cat == c
        if sub.sum() == 0:
            continue
        res[f"pctV_{c}"] = rate(detV, sub)
        res[f"pctJ_{c}"] = rate(detJ, sub)
        res[f"n_{c}"] = int(sub.sum())
    return res


out = {}
out["single"] = run(
    BASE + "/checkpoints/cc_single_L20.pt",
    BASE + "/results/classification_single_L20.csv",
    "single_L20",
)
out["mixed"] = run(
    BASE + "/checkpoints/cc_mixed_L20.pt",
    BASE + "/results/classification_mixed_L20.csv",
    "mixed_L20",
)
json.dump(out, open(BASE + "/results/mixed_vj_compare.json", "w"), indent=2)
log("DONE")
log(json.dumps(out, indent=2))
