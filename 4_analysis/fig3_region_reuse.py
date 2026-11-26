#!/usr/bin/env python3
"""Fig3 a-c per-latent data, 3 seeds, REUSING the verified ab_f1_folds_rebuild95 F1 code.
Per seed dumps fig3_perlatent_paired_s{seed}.npz with:
  f1max_wholetest (DICT,7)  - per-latent whole-test max-F1 per region (same math as folds, folds merged)
  region_floor_wholetest(7) - Haque floor on whole test
  cat (DICT,)               - canonical decoder-norm category
  ab_freq (DICT,)           - antibody firing frequency over all test residues
  region_names, region_best_fold(5,7), region_floor_fold(5,7), region_ndet_fold(5,7)  - fold summaries (verify vs JSON)
"""

import json, time, numpy as np, h5py, torch as th, pyarrow.parquet as pq, os, sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
os.environ["HF_HOME"] = f"{BASE}/caches/hf"
ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


NFOLD = 5
MARGIN = 0.2
MIN_PREV = 0.005
MIN_POS = 15
DICT = 32768
REGION_NAMES = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
NBIN = 40
from dictionary_learning.dictionary import BatchTopKCrossCoder

h5 = f"{ACTS}/exp_paired_small_L20.h5"
with h5py.File(h5, "r") as h:
    split = h["split"][:]
    seqid = h["seq_id"][:]
    region_all = h["region_label"][:]
    N = split.shape[0]
test = split == 2
idx = np.nonzero(test)[0]
seqs_test = np.unique(seqid[test])
rng = np.random.default_rng(0)
perm = rng.permutation(len(seqs_test))
seqfold = (perm % NFOLD).astype(np.int8)
res_fold = np.full(N, -1, dtype=np.int8)
pos = np.searchsorted(seqs_test, seqid)
pos = np.clip(pos, 0, len(seqs_test) - 1)
is_t = test & (seqs_test[pos] == seqid)
res_fold[is_t] = seqfold[pos[is_t]]
sid2row = {int(s): i for i, s in enumerate(seqs_test)}
nS = len(seqs_test)
EDGES = np.concatenate([[0], np.logspace(-3, 1.0, NBIN - 1)]).astype(np.float32)
edges_t = th.tensor(EDGES, device=dev)


def crit(pi):
    return 2 * pi / (1 + pi) + MARGIN


def run_seed(seed):
    ck = f"{CKPT}/cc_paired_small_s{seed}_L20.pt"
    ae = BatchTopKCrossCoder.from_pretrained(ck, device=dev, dtype=th.float32)
    ae.eval()
    # canonical category
    W = ae.decoder.weight.detach()
    n = th.linalg.norm(W, dim=2).transpose(0, 1)
    rel = n / (n.sum(1, keepdim=True) + 1e-12)
    rp, ru, rpr = rel[:, 0], rel[:, 1], rel[:, 2]
    HI, LO = 0.5, 0.2
    cat = th.full((DICT,), 5, dtype=th.int64)
    cat[(rp > LO) & (ru > LO) & (rpr > LO)] = 0
    cat[(rp > HI) & (ru < LO) & (rpr < LO)] = 1
    cat[(rp < LO) & (ru > LO) & (rpr > LO)] = 2
    cat[(ru > HI) & (rpr < LO)] = 4
    cat[(rp < LO) & (ru < LO) & (rpr > HI)] = 3
    cat = cat.cpu().numpy()
    rel_np = rel.cpu().numpy()
    hist = th.zeros(NFOLD, DICT, 7, NBIN, device="cpu", dtype=th.float32)
    regtot = th.zeros(NFOLD, 7, device="cpu", dtype=th.float64)
    fire = th.zeros(DICT, device=dev, dtype=th.float64)
    nres = 0
    BLK = 200000
    ENC = 4096
    t0 = time.time()
    with h5py.File(h5, "r") as h:
        acts = h["acts"]
        for s in range(0, len(idx), BLK):
            e = min(s + BLK, len(idx))
            gi = idx[s:e]
            lo = gi.min()
            x = acts[lo : gi.max() + 1]
            xb_all = x[gi - lo]
            reg = region_all[gi]
            fo = res_fold[gi]
            rg = th.tensor(reg.astype(np.int64), device=dev)
            fd = th.tensor(fo.astype(np.int64), device=dev)
            for b0 in range(0, e - s, ENC):
                b1 = min(b0 + ENC, e - s)
                xb = th.tensor(xb_all[b0:b1].astype(np.float32), device=dev)
                with th.no_grad():
                    f = ae.encode(xb, use_threshold=True, normalize_activations=True).float()
                nz = (f > 0).float()
                fire += nz.sum(0).double()
                nres += xb.shape[0]
                rgb = rg[b0:b1]
                fdb = fd[b0:b1]
                valid = (rgb >= 0) & (rgb < 7) & (fdb >= 0)
                if valid.any():
                    fv = f[valid]
                    rv = rgb[valid]
                    dv = fdb[valid]
                    for fk in range(NFOLD):
                        fm = dv == fk
                        if fm.any():
                            rr = rv[fm]
                            oh = th.zeros(int(fm.sum()), 7, device=dev, dtype=th.float64)
                            oh[th.arange(int(fm.sum()), device=dev), rr] = 1.0
                            regtot[fk] += oh.sum(0).cpu()
                            sub = fv[fm]
                            binidx = th.bucketize(sub, edges_t, right=False).clamp(0, NBIN - 1)
                            nzmask = sub > 0
                            for r_ in th.unique(rr):
                                rm = rr == r_
                                bi = binidx[rm]
                                nm = nzmask[rm]
                                if not nm.any():
                                    continue
                                flat = (
                                    th.arange(DICT, device=dev)[None, :].expand(bi.shape[0], DICT)
                                ) * NBIN + bi
                                add = th.zeros(DICT * NBIN, device=dev, dtype=th.float32)
                                add.index_add_(
                                    0,
                                    (flat[nm]).reshape(-1),
                                    th.ones(int(nm.sum()), device=dev, dtype=th.float32),
                                )
                                hist[fk, :, int(r_), :] += add.reshape(DICT, NBIN).cpu()
            if (s // BLK) % 10 == 0:
                log(f"  seed{seed} encode {e}/{len(idx)} {time.time()-t0:.0f}s")
    # ---- fold summaries (verbatim math) ----
    region_best = np.zeros((NFOLD, 7))
    region_ndet = np.zeros((NFOLD, 7), dtype=int)
    region_floor = np.zeros((NFOLD, 7))
    for fk in range(NFOLD):
        H = hist[fk].to(dev).double()
        suf = th.flip(th.cumsum(th.flip(H, [2]), dim=2), [2])
        predpos = suf.sum(1)
        pos_r = regtot[fk].to(dev)
        Ntot = float(pos_r.sum())
        tp = suf
        prec = tp / th.clamp(predpos[:, None, :], min=1e-9)
        rec = tp / th.clamp(pos_r[None, :, None], min=1e-9)
        f1 = 2 * prec * rec / th.clamp(prec + rec, min=1e-9)
        f1max = f1.max(2).values.cpu().numpy()
        pi_r = (pos_r / Ntot).cpu().numpy()
        fl = crit(pi_r)
        region_floor[fk] = fl
        for r_ in range(7):
            region_best[fk, r_] = f1max[:, r_].max()
            region_ndet[fk, r_] = int((f1max[:, r_] > fl[r_]).sum())
    # ---- whole-test per-latent f1max (folds merged) ----
    Hall = hist.sum(0).to(dev).double()
    suf = th.flip(th.cumsum(th.flip(Hall, [2]), dim=2), [2])
    predpos = suf.sum(1)
    pos_r = regtot.sum(0).to(dev)
    Ntot = float(pos_r.sum())
    tp = suf
    prec = tp / th.clamp(predpos[:, None, :], min=1e-9)
    rec = tp / th.clamp(pos_r[None, :, None], min=1e-9)
    f1 = 2 * prec * rec / th.clamp(prec + rec, min=1e-9)
    f1max_wt = f1.max(2).values.cpu().numpy()  # (DICT,7)
    floor_wt = crit((pos_r / Ntot).cpu().numpy())
    ab_freq = (fire / max(nres, 1)).cpu().numpy()
    np.savez(
        f"fig3_perlatent_paired_s{seed}.npz",
        f1max_wholetest=f1max_wt,
        region_floor_wholetest=floor_wt,
        cat=cat,
        rel=rel_np,
        ab_freq=ab_freq,
        region_names=np.array(REGION_NAMES),
        region_best_fold=region_best,
        region_floor_fold=region_floor,
        region_ndet_fold=region_ndet,
    )
    det = [int((f1max_wt[:, r] > floor_wt[r]).sum()) for r in range(7)]
    log(
        f"seed{seed} DONE wholetest region maxF1 {dict(zip(REGION_NAMES,f1max_wt.max(0).round(3)))} ndet {dict(zip(REGION_NAMES,det))}"
    )


for sd in [0, 1, 2]:
    run_seed(sd)
log("ALL DONE")
