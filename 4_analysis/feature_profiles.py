"""Per-residue activation profile of selected latents along example sequences.
For each (feat, seq_id) pair, encode that sequence's full residue block through
cc_mixed_L20 and record the latent's activation at every residue. Writes
feature_profiles.json."""

import h5py, numpy as np, torch as th, json

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

# (feat, seq_id) pairs to profile
PAIRS = {
    "9315": [399, 22833, 22911],
    "22964": [5206, 29014, 2365],
    "16581": [9560, 9656, 26418],
    "916": [130965, 139813, 130967],
    "6862": [111033, 113401, 112739],
    "22332": [136938, 131045, 110530],
}
feats = [int(f) for f in PAIRS]
feat_t = th.tensor(feats, device=DEV)
feat_idx = {int(f): i for i, f in enumerate(feats)}

h = h5py.File(f"{D}/activations/mixed_single_L20.h5", "r")
acts = h["acts"]
kind = h["kind"][:]
sid = h["seq_id"][:].astype(np.int64)
N = acts.shape[0]
# group boundaries (store grouped contiguously by seq_id)
uniq = np.unique(sid)
startidx = np.searchsorted(sid, uniq)
start_of = dict(zip(uniq.tolist(), startidx.tolist()))
# end of each group
order = np.argsort(uniq)
ends = {}
for i, u in enumerate(uniq):
    s0 = startidx[i]
    s1 = startidx[i + 1] if i + 1 < len(uniq) else N
    ends[int(u)] = int(s1)

out = {"feats": feats, "profiles": {}}
for fstr, sids in PAIRS.items():
    f = int(fstr)
    fi = feat_idx[f]
    out["profiles"][fstr] = {}
    for s in sids:
        if s not in start_of:
            out["profiles"][fstr][str(s)] = {"error": "seq_id not in store"}
            continue
        a = start_of[s]
        b = ends[s]
        blk = acts[a:b]  # (L,3,1024)
        kk = int(kind[a])  # domain of this seq
        x = th.tensor(blk.astype(np.float32), device=DEV)
        with th.no_grad():
            code = ae.encode(x, use_threshold=True, normalize_activations=True)  # (L,dict)
        prof = code.index_select(1, feat_t)[:, fi].cpu().numpy().tolist()  # (L,)
        out["profiles"][fstr][str(s)] = {"kind": kk, "n_res": int(b - a), "act": prof}
json.dump(out, open(f"{D}/results/feature_profiles.json", "w"))
print("DONE profiles for", list(PAIRS.keys()))
