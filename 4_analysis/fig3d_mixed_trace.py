#!/usr/bin/env python3
"""Fig3d on MIXED crosscoder: positional firing of each region's top-F1 detector
along one example antibody's FULL sequence (heavy + light concatenated). Region codes
reconstructed from orig_id->oas_pool_95nr (heavy then light). TOPLAT from mixed seed0 F1.
Writes region_toplat_profile_mixed.json (includes heavy_len to mark the H/L boundary)."""

import os, sys, json, time, numpy as np, h5py, torch as th, pyarrow.parquet as pq

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
os.environ["HF_HOME"] = f"{BASE}/caches/hf"
ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
DATA = f"{BASE}/data"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(
    f"{CKPT}/cc_mixed_small_s0_L20.pt", device=dev, dtype=th.float32
)
ae.eval()
TOPLAT = {"CDR1": 23889, "CDR2": 16128, "CDR3": 27515, "FR4": 22477}
REGION_MAP = {"FR1": 0, "CDR1": 1, "FR2": 2, "CDR2": 3, "FR3": 4, "CDR3": 5, "FR4": 6}
INV = {v: k for k, v in REGION_MAP.items()}

mt = pq.read_table(
    f"{DATA}/exp_mixed_small.parquet", columns=["seq_id", "orig_id", "kind"]
).to_pydict()
op = pq.read_table(
    f"{DATA}/oas_pool_95nr.parquet", columns=["id", "region_labels_heavy", "region_labels_light"]
).to_pydict()
idmap = {op["id"][i]: i for i in range(len(op["id"]))}
seqreg = {}
seqhlen = {}
for row in range(len(mt["kind"])):
    if mt["kind"][row] != 0:
        continue
    sid = int(mt["seq_id"][row])
    oid = mt["orig_id"][row]
    j = idmap.get(oid)
    if j is None:
        continue
    rh = op["region_labels_heavy"][j]
    rl = op["region_labels_light"][j]
    hcodes = [REGION_MAP.get(x, -1) for x in (list(rh) if rh is not None else [])]
    lcodes = [REGION_MAP.get(x, -1) for x in (list(rl) if rl is not None else [])]
    seqreg[sid] = np.array(hcodes + lcodes, dtype=np.int8)
    seqhlen[sid] = len(hcodes)
log(f"region labels for {len(seqreg)} ab seqs")

with h5py.File(f"{ACTS}/exp_mixed_small_L20.h5", "r") as h:
    sp = h["split"][:]
    kd = h["kind"][:]
    sid = h["seq_id"][:]
    tmean = th.tensor(h["train_mean"][:], device=dev, dtype=th.float32)
    tstd = th.tensor(h["train_std"][:], device=dev, dtype=th.float32)
mask = (sp == 2) & (kd == 0)
ab = np.nonzero(mask)[0]
order = np.argsort(sid[ab], kind="stable")
ab_sorted = ab[order]
s_sorted = sid[ab_sorted]
uniq, starts = np.unique(s_sorted, return_index=True)
starts = list(starts) + [len(ab_sorted)]
best = None
with h5py.File(f"{ACTS}/exp_mixed_small_L20.h5", "r") as h:
    acts = h["acts"]
    cnt = 0
    for u_i, u in enumerate(uniq):
        rc = seqreg.get(int(u))
        hl = seqhlen.get(int(u))
        if rc is None or hl is None:
            continue
        rows = ab_sorted[starts[u_i] : starts[u_i + 1]]
        m = min(len(rows), len(rc))
        if m < 200 or m > 280:
            continue  # full H+L length window
        rows = rows[:m]
        codes = rc[:m]
        # require both chains well represented: all 7 regions present somewhere
        if len(set(codes.tolist())) < 7:
            continue
        x = acts[rows.min() : rows.max() + 1]
        take = rows - rows.min()
        xb = th.tensor(x[take].astype(np.float32), device=dev)
        xn = (xb - tmean) / tstd
        with th.no_grad():
            f = ae.encode(xn, use_threshold=True, normalize_activations=False)
        prof = {r: f[:, l].cpu().numpy().tolist() for r, l in TOPLAT.items()}
        sc = sum(max(prof[r]) for r in TOPLAT)
        if best is None or sc > best["score"]:
            best = {
                "seq_id": int(u),
                "len": int(m),
                "heavy_len": int(hl),
                "regions": [INV[int(v)] for v in codes],
                "profiles": prof,
                "score": float(sc),
            }
        cnt += 1
        if cnt >= 3000:
            break
log(
    f"picked seq {best['seq_id']} len {best['len']} heavy_len {best['heavy_len']} score {best['score']:.2f}"
)
json.dump({"toplat": TOPLAT, "example": best}, open("region_toplat_profile_mixed.json", "w"))
print(
    json.dumps(
        {
            "seq_id": best["seq_id"],
            "len": best["len"],
            "heavy_len": best["heavy_len"],
            "peak": {r: round(max(best["profiles"][r]), 3) for r in TOPLAT},
        },
        indent=1,
    )
)
log("DONE")
