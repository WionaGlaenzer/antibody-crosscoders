"""Per-model single-SAE activation-frequency stats: for each trained per-model BatchTopK SAE
(sae_{prot,unp,pair}_L20, plain and squared-loss variants), encode the full paired-L20 val
store and report firing frequency / active-latent counts. Baseline comparison for the crosscoder.
"""

import os, json, h5py, numpy as np, torch as th

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

D = BASE
H5 = f"{D}/activations/full_paired_L20.h5"
from dictionary_learning.dictionary import BatchTopKSAE

COL = {"prot": 0, "unp": 1, "pair": 2}
RUNS = [
    ("sae_prot_L20", "prot"),
    ("sae_prot_L20_sq", "prot"),
    ("sae_unp_L20", "unp"),
    ("sae_unp_L20_sq", "unp"),
    ("sae_pair_L20", "pair"),
    ("sae_pair_L20_sq", "pair"),
]
dev = "cuda" if th.cuda.is_available() else "cpu"
f = h5py.File(H5, "r")
sv = f["split"][:]
testmask = sv == 2
acts = f["acts"]
idx = np.where(testmask)[0]
np.random.seed(0)
idx = np.sort(np.random.choice(idx, size=min(300000, len(idx)), replace=False))
print("test residues sampled:", len(idx))
tmean = f["train_mean"][:]
tstd = f["train_std"][:]
out = {}
for name, mk in RUNS:
    col = COL[mk]
    sd = th.load(f"{D}/checkpoints/{name}.pt", map_location="cpu")
    thr = sd.get("threshold", None)
    ae = BatchTopKSAE(activation_dim=1024, dict_size=16384, k=32)
    missing, unexpected = ae.load_state_dict(sd, strict=False)
    ae = ae.to(dev).eval()
    mean = th.tensor(tmean[col], dtype=th.float32, device=dev)
    std = th.tensor(tstd[col], dtype=th.float32, device=dev)
    fire = np.zeros(16384, dtype=np.int64)
    actsum = np.zeros(16384, dtype=np.float64)
    ntok = 0
    k_per_tok = []
    B = 8192
    for s in range(0, len(idx), B):
        rows = idx[s : s + B].tolist()
        x = acts[rows][:, col, :].astype(np.float32)
        xt = th.from_numpy(x).to(dev)
        xn = (xt - mean) / std
        with th.no_grad():
            fcode = ae.encode(
                xn, use_threshold=False, normalize_activations=False
            )  # TRUE batch top-k
        fc = fcode.float()
        fm = fc > 0
        fire += fm.sum(0).cpu().numpy().astype(np.int64)
        actsum += fc.sum(0).cpu().numpy().astype(np.float64)
        ntok += fc.shape[0]
        k_per_tok.append(float(fm.sum(1).float().mean().cpu()))
    freq = fire / max(ntok, 1)
    mwa = np.where(fire > 0, actsum / np.maximum(fire, 1), 0.0)
    out[name] = {
        "col": col,
        "model": mk,
        "ntok": int(ntok),
        "freq": freq.tolist(),
        "mean_when_active": mwa.tolist(),
        "n_dead": int((fire == 0).sum()),
        "n_active": int((fire > 0).sum()),
        "mean_k_per_token": float(np.mean(k_per_tok)),
        "threshold_loaded": (thr is not None),
        "threshold_val": (
            float(thr.mean()) if hasattr(thr, "mean") else (float(thr) if thr is not None else None)
        ),
    }
    print(
        name,
        "n_active",
        int((fire > 0).sum()),
        "dead",
        int((fire == 0).sum()),
        "mean_k/tok",
        round(float(np.mean(k_per_tok)), 1),
        "med_freq_active",
        round(float(np.median(freq[freq > 0])), 5) if (freq > 0).any() else 0,
        "thr_loaded",
        thr is not None,
    )
json.dump(out, open(f"{D}/results/sae_feature_frequencies.json", "w"))
print("WROTE")
