"""Per-residue activation profiles from SOURCE stores (full sequences, in order).
Antibody: full_single_L20.h5 (has region_label, chain). General: general_single_L20.h5.
Encodes through cc_mixed_L20. Writes feature_profiles_src.json."""

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

# antibody features -> seq_ids (OAS rows); general features -> seq_ids (general parquet rows)
AB = {"9315": [399, 22833, 22911], "22964": [5206, 29014, 2365], "16581": [9560, 9656, 26418]}
GEN = {
    "916": [130965 - 100000, 139813 - 100000, 130967 - 100000],
    "6862": [111033 - 100000, 113401 - 100000, 112739 - 100000],
    "22332": [136938 - 100000, 131045 - 100000, 110530 - 100000],
}
ALLF = [int(f) for f in list(AB) + list(GEN)]
feat_t = th.tensor(ALLF, device=DEV)
fidx = {f: i for i, f in enumerate(ALLF)}


def group_bounds(sid_arr, N):
    uniq = np.unique(sid_arr)
    starts = np.searchsorted(sid_arr, uniq)
    d = {}
    for i, u in enumerate(uniq):
        d[int(u)] = (int(starts[i]), int(starts[i + 1]) if i + 1 < len(uniq) else N)
    return d


out = {"ab": {}, "gen": {}}
# --- antibody source store ---
h = h5py.File(f"{D}/activations/full_single_L20.h5", "r")
acts = h["acts"]
sid = h["seq_id"][:].astype(np.int64)
reg = h["region_label"][:]
ch = h["chain"][:]
N = acts.shape[0]
gb = group_bounds(sid, N)
for fstr, sids in AB.items():
    f = int(fstr)
    out["ab"][fstr] = {}
    for s in sids:
        if s not in gb:
            out["ab"][fstr][str(s)] = {"error": "absent"}
            continue
        a, b = gb[s]
        x = th.tensor(acts[a:b].astype(np.float32), device=DEV)
        with th.no_grad():
            code = ae.encode(x, use_threshold=True, normalize_activations=True)
        p = code.index_select(1, feat_t)[:, fidx[f]].cpu().numpy().tolist()
        out["ab"][fstr][str(s)] = {
            "act": p,
            "region": [int(r) for r in reg[a:b]],
            "chain": [
                bytes(c).decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in ch[a:b]
            ],
        }
h.close()
# --- general source store ---
h = h5py.File(f"{D}/activations/general_single_L20.h5", "r")
acts = h["acts"]
sid = h["seq_id"][:].astype(np.int64)
N = acts.shape[0]
gb = group_bounds(sid, N)
for fstr, sids in GEN.items():
    f = int(fstr)
    out["gen"][fstr] = {}
    for s in sids:
        if s not in gb:
            out["gen"][fstr][str(s)] = {"error": "absent"}
            continue
        a, b = gb[s]
        x = th.tensor(acts[a:b].astype(np.float32), device=DEV)
        with th.no_grad():
            code = ae.encode(x, use_threshold=True, normalize_activations=True)
        p = code.index_select(1, feat_t)[:, fidx[f]].cpu().numpy().tolist()
        out["gen"][fstr][str(s)] = {"act": p}
h.close()
json.dump(out, open(f"{D}/results/feature_profiles_src.json", "w"))
print("DONE src profiles")
