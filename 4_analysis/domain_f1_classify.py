#!/usr/bin/env python3
"""Antibody-vs-general domain classification by per-latent max-F1 (Haque criterion),
as an alternative to the log-ratio. Encodes the mixed store test split through the
mixed crosscoder; label = kind (0=antibody, 1=general) per residue. Two one-vs-rest
concepts: antibody-detector (positive=antibody residues) and general-detector
(positive=general residues). Exact sparse sorted-threshold max-F1 sweep exploiting
BatchTopK sparsity. Detector iff max-F1 > 2*pi/(1+pi)+0.2. Latent category from decoder
norms (classify_decoder_norms logic). Writes domain_f1_{tag}.npz + domain_f1_{tag}.json."""

import json, time, argparse, numpy as np, h5py, torch as th

import sys
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
import sys

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
import os

os.environ["HF_HOME"] = f"{BASE}/caches/hf"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(args.ckpt, device=dev, dtype=th.float32)
ae.eval()
DICT = ae.dict_size
log("loaded", args.ckpt, "dict", DICT)

# ---- category from decoder norms (identical to classify_decoder_norms) ----
Wdec = ae.decoder.weight.detach()
nn = th.linalg.norm(Wdec, dim=2).transpose(0, 1)
rel = nn / (nn.sum(1, keepdim=True) + 1e-12)
rp, ru, rpr = rel[:, 0], rel[:, 1], rel[:, 2]
HI, LO = 0.5, 0.2
cat = th.full((DICT,), 5, dtype=th.int64)
cat[(rp > LO) & (ru > LO) & (rpr > LO)] = 0
cat[(rp > HI) & (ru < LO) & (rpr < LO)] = 1
cat[(rp < LO) & (ru > LO) & (rpr > LO)] = 2
cat[(ru > HI) & (rpr < LO)] = 4
cat[(rp < LO) & (ru < LO) & (rpr > HI)] = 3
cat_np = cat.cpu().numpy()
CAT = {
    0: "shared-all",
    1: "base-only",
    2: "gained-at-unpaired",
    3: "gained-at-paired",
    4: "transient",
    5: "other",
}

# ---- test residues, balanced kind, capped ----
with h5py.File(args.h5, "r") as h:
    sp = h["split"][:]
    kd = h["kind"][:]
    tmean = th.tensor(h["train_mean"][:], device=dev, dtype=th.float32)
    tstd = th.tensor(h["train_std"][:], device=dev, dtype=th.float32)
    idx = np.nonzero(sp == 2)[0]
    rng = np.random.default_rng(0)
    if len(idx) > args.cap:
        idx = np.sort(rng.choice(idx, args.cap, replace=False))
    kind_sel = kd[idx]  # 0=ab,1=gen
    N = len(idx)
    nab = int((kind_sel == 0).sum())
    ngen = int((kind_sel == 1).sum())
    log(f"test residues {N} | ab {nab} gen {ngen}")
    # two labels: concept0 = antibody(positive if kind==0), concept1 = general(positive if kind==1)
    y_ab = (kind_sel == 0).astype(np.int8)
    y_gen = (kind_sel == 1).astype(np.int8)
    Pab = int(y_ab.sum())
    Pgen = int(y_gen.sum())
    # --- encode pass: collect nonzero (latent, activation, is_ab) ---
    acts = h["acts"]
    ENC = 16384
    lat_buf = []
    act_buf = []
    ab_buf = []
    tot = 0
    t0 = time.time()
    for b0 in range(0, N, ENC):
        b1 = min(b0 + ENC, N)
        bidx = idx[b0:b1]
        xb = th.tensor(acts[bidx].astype(np.float32), device=dev)
        with th.no_grad():
            f = ae.encode(xb, use_threshold=True, normalize_activations=True)
        r_i, l_i = f.nonzero(as_tuple=True)
        a_v = f[r_i, l_i]
        isab = th.tensor(y_ab[b0:b1], device=dev)[r_i]
        lat_buf.append(l_i.to(th.int32).cpu().numpy())
        act_buf.append(a_v.cpu().numpy())
        ab_buf.append(isab.cpu().numpy())
        tot += len(r_i)
        if b0 == 0:
            per_res = len(r_i) / (b1 - b0)
            assert (
                per_res < 200
            ), f"encode not sparse: {per_res:.0f} nz/res (expected ~k=32); wrong normalize convention"
            log(f"  sparsity ok: {per_res:.1f} nz/res")
        if (b0 // ENC) % 40 == 0:
            log(f"  encode {b1}/{N} nz={tot} {time.time()-t0:.0f}s")
lat = np.concatenate(lat_buf)
act = np.concatenate(act_buf)
isab = np.concatenate(ab_buf).astype(np.int8)
del lat_buf, act_buf, ab_buf
log(f"nonzeros {len(lat)} ({len(lat)/N:.1f}/res)")
active = np.unique(lat)
# ---- exact per-latent max-F1 for antibody(+) and general(+) ----
order = np.argsort(lat, kind="stable")
lat_s = lat[order]
act_s = act[order]
ab_s = isab[order]
la_u, la_first, la_cnt = np.unique(lat_s, return_index=True, return_counts=True)
maxf1 = np.zeros((DICT, 2), dtype=np.float32)  # col0=antibody detector, col1=general detector
t0 = time.time()
for k, (L, f0, c) in enumerate(zip(la_u, la_first, la_cnt)):
    a = act_s[f0 : f0 + c]
    ab = ab_s[f0 : f0 + c]
    o = np.argsort(-a, kind="stable")
    abo = ab[o].astype(np.float64)
    pp = np.arange(1, c + 1, dtype=np.float64)
    # antibody positive
    tp = np.cumsum(abo)
    prec = tp / pp
    rec = tp / max(Pab, 1)
    f1a = np.where((prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    # general positive (1-ab)
    tpg = np.cumsum(1.0 - abo)
    precg = tpg / pp
    recg = tpg / max(Pgen, 1)
    f1g = np.where((precg + recg) > 0, 2 * precg * recg / np.maximum(precg + recg, 1e-12), 0.0)
    maxf1[L, 0] = f1a.max()
    maxf1[L, 1] = f1g.max()
    if k % 3000 == 0:
        log(f"  sweep {k}/{len(la_u)} {time.time()-t0:.0f}s")
pi_ab = Pab / N
pi_gen = Pgen / N
floor_ab = 2 * pi_ab / (1 + pi_ab) + 0.2
floor_gen = 2 * pi_gen / (1 + pi_gen) + 0.2
det_ab = maxf1[:, 0] > floor_ab
det_gen = maxf1[:, 1] > floor_gen
np.savez(
    f"{RES}/domain_f1_{args.tag}.npz",
    maxf1=maxf1,
    active=active,
    cat=cat_np,
    pi_ab=pi_ab,
    pi_gen=pi_gen,
    floor_ab=floor_ab,
    floor_gen=floor_gen,
    det_ab=det_ab,
    det_gen=det_gen,
)
# summary by category (active only)
act_mask = np.zeros(DICT, bool)
act_mask[active] = True
out = {
    "tag": args.tag,
    "n_active": int(len(active)),
    "pi_ab": pi_ab,
    "pi_gen": pi_gen,
    "floor_ab": float(floor_ab),
    "floor_gen": float(floor_gen),
    "by_category": {},
}
for cc in [1, 0, 2]:  # base-only, shared, gained-unp
    m = (cat_np == cc) & act_mask
    n = int(m.sum())
    out["by_category"][CAT[cc]] = {
        "n_active": n,
        "antibody_detectors": int((det_ab & m).sum()),
        "general_detectors": int((det_gen & m).sum()),
        "pct_antibody": round(100 * float((det_ab & m).sum()) / max(n, 1), 1),
        "pct_general": round(100 * float((det_gen & m).sum()) / max(n, 1), 1),
    }
out["total_antibody_detectors"] = int((det_ab & act_mask).sum())
out["total_general_detectors"] = int((det_gen & act_mask).sum())
json.dump(out, open(f"{RES}/domain_f1_{args.tag}.json", "w"), indent=1)
log("DONE")
print(json.dumps(out, indent=1))
