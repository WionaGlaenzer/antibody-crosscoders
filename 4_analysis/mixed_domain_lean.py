#!/usr/bin/env python3
"""Lean domain-specificity for a MIXED crosscoder: category straight from decoder norms
(no val-memmap / VE recompute), plus ONE capped firing pass over the store's test split
split by kind (0=antibody,1=general). Writes mixed_domain_perlatent_{tag}.parquet +
mixed_domain_summary_{tag}.json. Category logic identical to classify_only.classify_decoder_norms;
firing logic identical to mixed_label.py. active = fires >=1x in the sample."""

import os, sys, json, time, argparse
import numpy as np, h5py, torch as th, pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE as DEFAULT_BASE

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--h5", required=True)
ap.add_argument("--base", default=DEFAULT_BASE)
ap.add_argument("--cap", type=int, default=400000)
args = ap.parse_args()
BASE = args.base
RES = f"{BASE}/results"
sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
os.environ["HF_HOME"] = f"{BASE}/caches/hf"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


dev = "cuda" if th.cuda.is_available() else "cpu"
log("device", dev)
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(args.ckpt, device=dev, dtype=th.float32)
ae.eval()
log("loaded", args.ckpt, "dict", ae.dict_size)

# ---- category from decoder norms (identical to classify_only.classify_decoder_norms) ----
Wdec = ae.decoder.weight.detach()
n = th.linalg.norm(Wdec, dim=2).transpose(0, 1)  # (dict,3) [prot,unp,pair]
s = n.sum(dim=1, keepdim=True)
rel = n / (s + 1e-12)
rp, ru, rpr = rel[:, 0], rel[:, 1], rel[:, 2]
HI, LO = 0.5, 0.2
cat = th.full((rel.shape[0],), 5, dtype=th.int64)
cat[(rp > LO) & (ru > LO) & (rpr > LO)] = 0  # shared-all
cat[(rp > HI) & (ru < LO) & (rpr < LO)] = 1  # base-only
cat[(rp < LO) & (ru > LO) & (rpr > LO)] = 2  # gained-at-unpaired
cat[(ru > HI) & (rpr < LO)] = 4  # transient
cat[(rp < LO) & (ru < LO) & (rpr > HI)] = 3  # gained-at-paired
cat_np = cat.cpu().numpy()
CAT_NAMES = {
    0: "shared-all",
    1: "base-only",
    2: "gained-at-unpaired",
    3: "gained-at-paired",
    4: "transient",
    5: "other",
    -1: "dead",
}

# ---- single capped firing pass over test split, split by kind ----
with h5py.File(args.h5, "r") as f:
    tmean = th.tensor(f["train_mean"][:], device=dev, dtype=th.float32)
    tstd = th.tensor(f["train_std"][:], device=dev, dtype=th.float32)
    sp = f["split"][:]
    kd = f["kind"][:]
    idx = np.nonzero(sp == 2)[0]
    rng = np.random.default_rng(0)
    if len(idx) > args.cap:
        idx = np.sort(rng.choice(idx, args.cap, replace=False))
    kind_sel = kd[idx]
    ngen = int((kind_sel == 1).sum())
    nab = int((kind_sel == 0).sum())
    log(f"test residues {len(idx)} | general {ngen} ab {nab}")
    acts = f["acts"]
    DICT = ae.dict_size
    fire_gen = np.zeros(DICT, np.int64)
    fire_ab = np.zeros(DICT, np.int64)
    fire_tot = np.zeros(DICT, np.int64)
    BLK = 8192
    for b0 in range(0, len(idx), BLK):
        bidx = idx[b0 : b0 + BLK]
        bkind = kd[bidx]
        x = th.tensor(acts[bidx].astype(np.float32), device=dev)
        xn = (x - tmean) / tstd
        fc = ae.encode(xn, use_threshold=False)
        fa = (fc > 0).cpu().numpy()  # (b,DICT) bool
        gm = bkind == 1
        am = bkind == 0
        fire_gen += fa[gm].sum(0)
        fire_ab += fa[am].sum(0)
        fire_tot += fa.sum(0)
        if (b0 // BLK) % 20 == 0:
            log(f"  {b0+len(bidx)}/{len(idx)}")
active_mask = fire_tot > 0
freq_gen = fire_gen / max(ngen, 1)
freq_ab = fire_ab / max(nab, 1)
log2ratio = np.log2((freq_ab + 1e-6) / (freq_gen + 1e-6))
act = np.nonzero(active_mask)[0]
df = pd.DataFrame(
    dict(
        latent_id=act,
        category=[CAT_NAMES[int(cat_np[j])] for j in act],
        freq_general=freq_gen[act],
        freq_antibody=freq_ab[act],
        log2_ab_over_gen=log2ratio[act],
    )
)
df.to_parquet(f"{RES}/mixed_domain_perlatent_{args.tag}.parquet")
log(f"active {len(act)} | cats {df.category.value_counts().to_dict()}")
summ = {}
for c in sorted(set(df.category)):
    d = df[df.category == c]
    summ[c] = dict(
        n=int(len(d)),
        median_freq_gen=float(d.freq_general.median()),
        median_freq_ab=float(d.freq_antibody.median()),
        median_log2ratio=float(d.log2_ab_over_gen.median()),
        frac_general_active=float((d.freq_general > 0.0005).mean()),
        frac_antibody_specific=float(
            ((d.freq_antibody > 0.0005) & (d.freq_general < 0.0001)).mean()
        ),
        frac_general_specific=float(
            ((d.freq_general > 0.0005) & (d.freq_antibody < 0.0001)).mean()
        ),
    )
json.dump(summ, open(f"{RES}/mixed_domain_summary_{args.tag}.json", "w"), indent=2)
log("DONE", args.tag)
print(json.dumps(summ, indent=2))
