#!/usr/bin/env python
"""Stage 2: compare crosscoder latent firing across SHM bins.
Encode the antibody single_L20 crosscoder over the SINGLE-store test residues, aggregate
per-latent firing frequency within each SHM bin (near-germline / low / mid / high), and
find SHM-selective latents. Cross with decoder-norm category (shared / gained-at-unpaired)
to test whether fine-tuning-gained latents are enriched on affinity-matured sequences.
seq_id in the store = 0-based row index into oas_full_annotated.parquet."""

import json, time, numpy as np, h5py, torch as th, pandas as pd, sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


dev = "cuda" if th.cuda.is_available() else "cpu"
log("device", dev)
from dictionary_learning.dictionary import BatchTopKCrossCoder

H5 = f"{BASE}/activations/full_single_L20.h5"
ae = BatchTopKCrossCoder.from_pretrained(
    f"{BASE}/checkpoints/cc_single_L20.pt", device=dev, dtype=th.float32
)
ae.eval()
cls = pd.read_csv(f"{BASE}/results/classification_single_L20.csv")
active = cls["latent_id"].to_numpy().astype(np.int64)
A = len(active)
cat = cls["category"].to_numpy()
active_t = th.tensor(active, device=dev)
log(f"{A} active latents")
# SHM per sequence (row index = seq_id)
shm = pd.read_parquet(f"{BASE}/results/shm_per_sequence.parquet").reset_index(drop=True)
shm_bin = shm["shm_bin"].to_numpy()  # aligned to row index
BINS = ["near-germline (>=98%)", "low-SHM (95-98%)", "mid-SHM (90-95%)", "high-SHM (<90%)"]
with h5py.File(H5, "r") as f:
    tmean = th.tensor(f["train_mean"][:], device=dev, dtype=th.float32)
    tstd = th.tensor(f["train_std"][:], device=dev, dtype=th.float32)
    sp = f["split"][:]
    seqid = f["seq_id"][:]
    acts = f["acts"]
    test_mask = sp == 2
    idx = np.nonzero(test_mask)[0]
    log(f"test residues {len(idx)}")
    # per-residue SHM bin via seq_id
    res_bin = shm_bin[seqid[idx]]
    fire = np.zeros((len(BINS), A), np.int64)
    nres = np.zeros(len(BINS), np.int64)
    bin2i = {b: i for i, b in enumerate(BINS)}
    BLK = 8192
    for b0 in range(0, len(idx), BLK):
        bidx = idx[b0 : b0 + BLK]
        bb = res_bin[b0 : b0 + BLK]
        x = th.tensor(acts[bidx].astype(np.float32), device=dev)
        xn = (x - tmean) / tstd
        fc = ae.encode(xn, use_threshold=False)
        fa = (fc[:, active_t] > 0).cpu().numpy()
        for bi in range(len(BINS)):
            m = bb == BINS[bi]
            if m.any():
                fire[bi] += fa[m].sum(0)
                nres[bi] += int(m.sum())
        if (b0 // BLK) % 40 == 0:
            log(f"  {b0+len(bidx)}/{len(idx)}")
freq = fire / np.maximum(nres[:, None], 1)  # (nbin, A)
df = pd.DataFrame({"latent_id": active, "category": cat})
for i, b in enumerate(BINS):
    df[f"freq_{['near','low','mid','high'][i]}"] = freq[i]
# SHM selectivity: log2(high / near)
df["log2_high_over_near"] = np.log2((df["freq_high"] + 1e-6) / (df["freq_near"] + 1e-6))
df.to_parquet(f"{BASE}/results/shm_firing_perlatent.parquet")
# summary by category
summ = {"n_residues_per_bin": {b: int(nres[i]) for i, b in enumerate(BINS)}}
for c in ["shared-all", "gained-at-unpaired"]:
    d = df[df.category == c]
    if len(d) == 0:
        continue
    summ[c] = dict(
        n=int(len(d)),
        median_log2_high_over_near=float(d["log2_high_over_near"].median()),
        frac_high_selective=float((d["log2_high_over_near"] > 1).mean()),
        frac_near_selective=float((d["log2_high_over_near"] < -1).mean()),
        mean_freq_near=float(d["freq_near"].mean()),
        mean_freq_high=float(d["freq_high"].mean()),
    )
# top SHM-selective latents overall
top_high = df.nlargest(20, "log2_high_over_near")[
    ["latent_id", "category", "freq_near", "freq_high", "log2_high_over_near"]
]
summ["top20_high_SHM_selective"] = top_high.to_dict("records")
json.dump(summ, open(f"{BASE}/results/shm_firing_summary.json", "w"), indent=2, default=float)
log("DONE")
print(json.dumps(summ, indent=2, default=float)[:1500])
