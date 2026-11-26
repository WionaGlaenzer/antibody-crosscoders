"""Diagnostic sanity check on the mixed activation store: counts non-finite (NaN/Inf) entries in
the per-model train_mean/train_std, the minimum std per model column, and per-kind bad-row counts.
Used to catch normalization-stat corruption before training. Writes mixed_col_diag.json.
"""

import h5py, numpy as np, json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

f = h5py.File(f"{BASE}/activations/mixed_single_L20.h5", "r")
tm = f["train_mean"][:]
ts = f["train_std"][:]
out = {}
out["mean_nonfinite_per_col"] = [int((~np.isfinite(tm[c])).sum()) for c in range(3)]
out["std_nonfinite_per_col"] = [int((~np.isfinite(ts[c])).sum()) for c in range(3)]
out["std_min_per_col"] = [float(ts[c].min()) for c in range(3)]
acts = f["acts"]
kind = f["kind"][:]
N = acts.shape[0]
BLK = 262144
bad = {0: 0, 1: 0, 2: 0}
badgen = {0: 0, 1: 0, 2: 0}
absmax = {0: 0.0, 1: 0.0, 2: 0.0}
for s in range(0, N, BLK):
    e = min(s + BLK, N)
    blk = acts[s:e].astype(np.float32)
    k = kind[s:e]
    for col in range(3):
        col_data = blk[:, col, :]
        fin = np.isfinite(col_data)
        m = ~fin.all(axis=1)
        bad[col] += int(m.sum())
        badgen[col] += int((m & (k == 1)).sum())
        if fin.any():
            absmax[col] = max(absmax[col], float(np.abs(col_data[fin]).max()))
out["nonfinite_rows_per_col"] = bad
out["nonfinite_general_per_col"] = badgen
out["absmax_per_col"] = absmax
out["N"] = int(N)
f.close()
json.dump(
    out,
    open(f"{BASE}/results/mixed_col_diag.json", "w"),
    indent=2,
)
print(json.dumps(out, indent=2))
