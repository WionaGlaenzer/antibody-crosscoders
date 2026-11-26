"""Label the MIXED crosscoder latents by domain-specificity (antibody vs general
firing) and, for antibody-firing latents, germline V/J + region concept F1.
Focus: characterize the revived base-only latents (37->1045 vs antibody-only).
Encodes mixed_single_L20.h5 test split through cc_mixed_L20, splits by kind."""

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
os.environ["HF_HOME"] = f"{BASE}/caches/hf"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


dev = "cuda" if th.cuda.is_available() else "cpu"
log("device", dev)
from dictionary_learning.dictionary import BatchTopKCrossCoder

H5 = f"{ACTS}/mixed_single_L20.h5"
ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_mixed_L20.pt", device=dev, dtype=th.float32)
ae.eval()
cls = pd.read_csv(f"{RES}/classification_mixed_L20.csv")
active = cls["latent_id"].to_numpy().astype(np.int64)
A = len(active)
cat = cls["category"].to_numpy()
active_t = th.tensor(active, device=dev)
log(f"{A} active latents")
with h5py.File(H5, "r") as f:
    tmean = th.tensor(f["train_mean"][:], device=dev, dtype=th.float32)
    tstd = th.tensor(f["train_std"][:], device=dev, dtype=th.float32)
    sp = f["split"][:]
    kd = f["kind"][:]
    test_mask = sp == 2
    idx = np.nonzero(test_mask)[0]
    # cap for speed: 400k test residues, keep kind balance
    rng = np.random.default_rng(0)
    if len(idx) > 400000:
        idx = np.sort(rng.choice(idx, 400000, replace=False))
    kind_sel = kd[idx]
    log(
        f"test residues {len(idx)} | general {int((kind_sel==1).sum())} ab {int((kind_sel==0).sum())}"
    )
    acts = f["acts"]
    # accumulate per-latent firing counts on general vs antibody
    fire_gen = np.zeros(A, np.int64)
    fire_ab = np.zeros(A, np.int64)
    ngen = int((kind_sel == 1).sum())
    nab = int((kind_sel == 0).sum())
    BLK = 8192
    for b0 in range(0, len(idx), BLK):
        bidx = idx[b0 : b0 + BLK]
        bkind = kd[bidx]
        x = th.tensor(acts[bidx].astype(np.float32), device=dev)  # (b,3,1024)
        xn = (x - tmean) / tstd
        f_codes = ae.encode(xn, use_threshold=False)  # true top-k (b,dict)
        fa = f_codes[:, active_t] > 0  # (b,A) bool
        fa = fa.cpu().numpy()
        gm = bkind == 1
        am = bkind == 0
        fire_gen += fa[gm].sum(0)
        fire_ab += fa[am].sum(0)
        if (b0 // BLK) % 20 == 0:
            log(f"  {b0+len(bidx)}/{len(idx)}")
freq_gen = fire_gen / max(ngen, 1)
freq_ab = fire_ab / max(nab, 1)
log2ratio = np.log2((freq_ab + 1e-6) / (freq_gen + 1e-6))
df = pd.DataFrame(
    dict(
        latent_id=active,
        category=cat,
        freq_general=freq_gen,
        freq_antibody=freq_ab,
        log2_ab_over_gen=log2ratio,
    )
)
df.to_parquet(f"{RES}/mixed_domain_perlatent.parquet")
# summary by category
summ = {}
for c in sorted(set(cat.tolist())):
    d = df[df["category"] == c]
    summ[c] = dict(
        n=int(len(d)),
        median_freq_gen=float(d["freq_general"].median()),
        median_freq_ab=float(d["freq_antibody"].median()),
        median_log2ratio=float(d["log2_ab_over_gen"].median()),
        frac_general_active=float((d["freq_general"] > 0.0005).mean()),
        frac_antibody_specific=float(
            ((d["freq_antibody"] > 0.0005) & (d["freq_general"] < 0.0001)).mean()
        ),
        frac_general_specific=float(
            ((d["freq_general"] > 0.0005) & (d["freq_antibody"] < 0.0001)).mean()
        ),
    )
json.dump(summ, open(f"{RES}/mixed_domain_summary.json", "w"), indent=2)
log("DONE")
print(json.dumps(summ, indent=2))
