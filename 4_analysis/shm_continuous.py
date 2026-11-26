#!/usr/bin/env python
"""Stage 3: CONTINUOUS SHM analysis (no bins). Per test sequence, aggregate each
active latent's firing rate (fraction of the sequence's residues where it fires).
Then per latent, Spearman-correlate that per-sequence firing vector against the
sequence's EXACT SHM level (n_mismatch and 100 - v_identity). Report the
distribution of correlations and the most SHM-correlated latents."""

import json, time, numpy as np, h5py, torch as th, pandas as pd, sys
from scipy.stats import spearmanr

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


dev = "cuda" if th.cuda.is_available() else "cpu"
log("device", dev)
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(
    f"{BASE}/checkpoints/cc_single_L20.pt", device=dev, dtype=th.float32
)
ae.eval()
cls = pd.read_csv(f"{BASE}/results/classification_single_L20.csv")
active = cls["latent_id"].to_numpy().astype(np.int64)
A = len(active)
cat = cls["category"].to_numpy()
active_t = th.tensor(active, device=dev)
shm = pd.read_parquet(f"{BASE}/results/shm_per_sequence.parquet").reset_index(drop=True)
vid = shm["v_identity"].to_numpy()
nmis = shm["n_mismatch"].to_numpy().astype(float)
shm_pct = 100.0 - vid  # % mutated (continuous)
H5 = f"{BASE}/activations/full_single_L20.h5"
with h5py.File(H5, "r") as f:
    tmean = th.tensor(f["train_mean"][:], device=dev, dtype=th.float32)
    tstd = th.tensor(f["train_std"][:], device=dev, dtype=th.float32)
    sp = f["split"][:]
    seqid = f["seq_id"][:]
    acts = f["acts"]
    test_mask = sp == 2
    idx = np.nonzero(test_mask)[0]
    tsid = seqid[idx]
    uniq = np.unique(tsid)
    n_seq = len(uniq)
    sid2row = {int(s): i for i, s in enumerate(uniq)}
    log(f"test residues {len(idx)} | unique sequences {n_seq}")
    # accumulate per-sequence firing count + residue count
    fire_sum = np.zeros((n_seq, A), np.float32)
    res_count = np.zeros(n_seq, np.float32)
    BLK = 8192
    for b0 in range(0, len(idx), BLK):
        bidx = idx[b0 : b0 + BLK]
        bs = tsid[b0 : b0 + BLK]
        x = th.tensor(acts[bidx].astype(np.float32), device=dev)
        xn = (x - tmean) / tstd
        fa = (ae.encode(xn, use_threshold=False)[:, active_t] > 0).cpu().numpy().astype(np.float32)
        rows = np.array([sid2row[int(s)] for s in bs])
        np.add.at(fire_sum, rows, fa)
        np.add.at(res_count, rows, 1.0)
        if (b0 // BLK) % 40 == 0:
            log(f"  {b0+len(bidx)}/{len(idx)}")
firing_rate = fire_sum / np.maximum(res_count[:, None], 1)  # (n_seq, A) per-seq firing rate
# align SHM to these sequences
seq_shm_pct = shm_pct[uniq]
seq_nmis = nmis[uniq]
valid = np.isfinite(seq_shm_pct)
log(
    f"valid seqs {valid.sum()}/{n_seq} | SHM%% range {np.nanmin(seq_shm_pct):.1f}-{np.nanmax(seq_shm_pct):.1f}"
)
# per-latent Spearman correlation with continuous SHM%
rho = np.full(A, np.nan)
pval = np.full(A, np.nan)
fr_v = firing_rate[valid]
shm_v = seq_shm_pct[valid]
for j in range(A):
    col = fr_v[:, j]
    if col.std() > 0:
        r, p = spearmanr(col, shm_v)
        rho[j] = r
        pval[j] = p
df = pd.DataFrame(
    {
        "latent_id": active,
        "category": cat,
        "spearman_rho_shm": rho,
        "pval": pval,
        "mean_firing_rate": firing_rate.mean(0),
    }
)
df.to_parquet(f"{BASE}/results/shm_continuous_perlatent.parquet")
# also save per-sequence mean firing of top latents for scatter + overall
np.savez(
    f"{BASE}/results/shm_continuous_seq.npz",
    seq_shm_pct=seq_shm_pct,
    seq_nmis=seq_nmis,
    uniq=uniq,
    valid=valid,
)


# summary: correlation distribution overall and by category
def qstats(d):
    r = d["spearman_rho_shm"].dropna()
    return dict(
        n=int(len(r)),
        median_rho=float(r.median()),
        mean_rho=float(r.mean()),
        frac_pos_sig=float(((r > 0) & (d["pval"] < 0.01)).mean()),
        frac_rho_gt_0_2=float((r > 0.2).mean()),
        frac_rho_lt_neg0_2=float((r < -0.2).mean()),
    )


summ = dict(n_seq=int(valid.sum()), overall=qstats(df))
for c in ["shared-all", "gained-at-unpaired"]:
    summ[c] = qstats(df[df.category == c])
top = df.dropna(subset=["spearman_rho_shm"]).nlargest(20, "spearman_rho_shm")[
    ["latent_id", "category", "spearman_rho_shm", "mean_firing_rate"]
]
summ["top20_shm_correlated"] = top.to_dict("records")
json.dump(summ, open(f"{BASE}/results/shm_continuous_summary.json", "w"), indent=2, default=float)
log("DONE")
print(
    json.dumps(
        {k: summ[k] for k in ["n_seq", "overall", "shared-all", "gained-at-unpaired"]},
        indent=2,
        default=float,
    )
)
