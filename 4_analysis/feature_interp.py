"""Find top-activating residues (and their source proteins) for selected
domain-specific latents of cc_mixed_L20. Encodes the mixed store (kind 0=antibody
1=general, seq_id namespaced +100000 for general). Writes feature_interp.json."""

import h5py, numpy as np, torch as th, json, time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

D = BASE
DEV = "cuda" if th.cuda.is_available() else "cpu"
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(
    f"{D}/checkpoints/cc_mixed_L20.pt", device=DEV, dtype=th.float32
)
ae.eval()

AB = [9315, 28057, 22964, 16581, 3748]  # antibody-specific
GEN = [916, 16162, 6862, 7647, 22332]  # general-specific
FEATS = AB + GEN
TOPK = 40
feat_t = th.tensor(FEATS, device=DEV)

h = h5py.File(f"{D}/activations/mixed_single_L20.h5", "r")
acts = h["acts"]
kind = h["kind"][:]
sid = h["seq_id"][:].astype(np.int64)
N = acts.shape[0]
# per-sequence residue index (position within the seq's residue block)
# store is grouped by seq_id contiguously; compute position as running count per seq
pos = np.zeros(N, dtype=np.int32)
_, first = np.unique(sid, return_index=True)
# vectorized position within group: idx - first_index_of_that_seq
order = np.argsort(sid, kind="stable")  # already grouped but be safe
# since grouped contiguously, positions = arange - group start
starts = {}
# compute group start via searchsorted on sorted-unique-first approach
uniq = np.unique(sid)
startidx = np.searchsorted(sid, uniq)  # works because contiguous & increasing
start_of = np.zeros(sid.max() + 1, dtype=np.int64)
start_of[uniq] = startidx
pos = (np.arange(N) - start_of[sid]).astype(np.int32)

# running top-K per feature: keep (val, global_idx)
best_val = th.full((len(FEATS), TOPK), -1e9, device=DEV)
best_idx = th.full((len(FEATS), TOPK), -1, dtype=th.long, device=DEV)
# also per-feature domain firing tally to confirm
fire_ab = th.zeros(len(FEATS), device=DEV)
fire_gen = th.zeros(len(FEATS), device=DEV)
n_ab = int((kind == 0).sum())
n_gen = int((kind == 1).sum())

F = len(FEATS)
BLK = 262144
ENC = 8192
t0 = time.time()
for s in range(0, N, BLK):
    e = min(s + BLK, N)
    blk = acts[s:e]
    kb = th.tensor(kind[s:e], device=DEV)
    # collect this block's codes for the F features, then do ONE fused topk update
    code_parts = []
    for b0 in range(0, e - s, ENC):
        b1 = min(b0 + ENC, e - s)
        x = th.tensor(blk[b0:b1].astype(np.float32), device=DEV)
        with th.no_grad():
            f = ae.encode(x, use_threshold=True, normalize_activations=True)
        code_parts.append(f.index_select(1, feat_t))  # (bb,F)
    code = th.cat(code_parts, 0)  # (blk,F)
    fire = code > 0
    fire_ab += (fire & (kb == 0).unsqueeze(1)).sum(0)
    fire_gen += (fire & (kb == 1).unsqueeze(1)).sum(0)
    gi = th.arange(s, e, device=DEV)
    # fused per-feature topk: transpose to (F,blk), cat with running best, topk along dim1
    cat_v = th.cat([best_val, code.T], dim=1)  # (F, TOPK+blk)
    cat_i = th.cat([best_idx, gi.unsqueeze(0).expand(F, -1)], dim=1)  # (F, TOPK+blk)
    tv, ti = th.topk(cat_v, TOPK, dim=1)
    best_val = tv
    best_idx = th.gather(cat_i, 1, ti)
    if s % (BLK * 20) == 0:
        print(f"[{time.time()-t0:.0f}s] {e}/{N}", flush=True)

out = {
    "feats": FEATS,
    "antibody_feats": AB,
    "general_feats": GEN,
    "topk": TOPK,
    "n_ab": n_ab,
    "n_gen": n_gen,
    "records": {},
}
bv = best_val.cpu().numpy()
bi = best_idx.cpu().numpy()
fa = fire_ab.cpu().numpy()
fg = fire_gen.cpu().numpy()
for fi, feat in enumerate(FEATS):
    recs = []
    for j in range(TOPK):
        gi = int(bi[fi, j])
        if gi < 0:
            continue
        recs.append(
            {
                "val": float(bv[fi, j]),
                "global_idx": gi,
                "seq_id": int(sid[gi]),
                "kind": int(kind[gi]),
                "pos": int(pos[gi]),
            }
        )
    out["records"][str(feat)] = {
        "top": recs,
        "freq_ab": float(fa[fi] / n_ab),
        "freq_gen": float(fg[fi] / n_gen),
    }
json.dump(out, open(f"{D}/results/feature_interp.json", "w"))
print(
    "DONE. wrote feature_interp.json; example freqs:",
    {FEATS[i]: (round(float(fa[i] / n_ab), 4), round(float(fg[i] / n_gen), 4)) for i in range(3)},
)
