"""Label a MIXED crosscoder's latents by domain-specificity (antibody vs general firing).
Parametrized version: --tag --ckpt --h5 --cls. Encodes the store's test split through the
checkpoint, splits residues by kind (0=antibody,1=general), and computes per-latent firing
frequency on each domain + log2 ratio. Writes mixed_domain_perlatent_{tag}.parquet and
mixed_domain_summary_{tag}.json. Firing logic identical to the original mixed_label.py."""

import os, sys, json, time, argparse
import numpy as np, h5py, torch as th, pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE as DEFAULT_BASE

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--h5", required=True)
ap.add_argument("--cls", required=True, help="classification_{tag}.csv with latent_id,category")
ap.add_argument("--base", default=DEFAULT_BASE)
ap.add_argument("--cap", type=int, default=400000)
args = ap.parse_args()
BASE = args.base
RES = f"{BASE}/results"
CODE = f"{BASE}/code/dictionary_learning_jkminder"
sys.path.insert(0, CODE)
os.environ["HF_HOME"] = f"{BASE}/caches/hf"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


dev = "cuda" if th.cuda.is_available() else "cpu"
log("device", dev)
from dictionary_learning.dictionary import BatchTopKCrossCoder

H5 = args.h5
ae = BatchTopKCrossCoder.from_pretrained(args.ckpt, device=dev, dtype=th.float32)
ae.eval()
cls = pd.read_csv(args.cls)
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
    rng = np.random.default_rng(0)
    if len(idx) > args.cap:
        idx = np.sort(rng.choice(idx, args.cap, replace=False))
    kind_sel = kd[idx]
    log(
        f"test residues {len(idx)} | general {int((kind_sel==1).sum())} ab {int((kind_sel==0).sum())}"
    )
    acts = f["acts"]
    fire_gen = np.zeros(A, np.int64)
    fire_ab = np.zeros(A, np.int64)
    ngen = int((kind_sel == 1).sum())
    nab = int((kind_sel == 0).sum())
    BLK = 8192
    for b0 in range(0, len(idx), BLK):
        bidx = idx[b0 : b0 + BLK]
        bkind = kd[bidx]
        x = th.tensor(acts[bidx].astype(np.float32), device=dev)
        xn = (x - tmean) / tstd
        f_codes = ae.encode(xn, use_threshold=False)
        fa = (f_codes[:, active_t] > 0).cpu().numpy()
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
df.to_parquet(f"{RES}/mixed_domain_perlatent_{args.tag}.parquet")
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
json.dump(summ, open(f"{RES}/mixed_domain_summary_{args.tag}.json", "w"), indent=2)
log("DONE", args.tag)
print(json.dumps(summ, indent=2))
