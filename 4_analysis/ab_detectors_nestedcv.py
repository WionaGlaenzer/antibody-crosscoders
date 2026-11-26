#!/usr/bin/env python3
"""Nested-CV (unbiased) vs naive (same-fold, best-case) max-F1 for antibody detectors on the
rebuilt PAIRED crosscoder (cc_paired_small_s0_L20), held-out TEST split of exp_paired_small_L20.h5.

Motivation: the original ab_f1_folds_rebuild95.py reports max-F1 where BOTH the detector latent
AND its firing threshold are chosen to maximise F1, and selection+scoring happen on the SAME fold
=> optimistically biased (winner's-curse). This script adds a proper NESTED-CV estimate:
  for each held-out fold k, SELECT the best (latent, threshold) on the other 4 folds (train),
  then EVALUATE that fixed choice on fold k (held out). Reports naive AND nested side-by-side.

Reuses the original's encode, per-sequence mean-of-nonzero V-gene score, per-region residue
histogram, and prevalence floor 2*pi/(1+pi)+0.2, all verbatim. Region histograms are additive
across folds, so train = sum over the 4 non-held-out folds. V-gene threshold is a per-sequence
score cutoff recovered from the train-optimal rank and applied numerically to the held-out fold.

Writes ab_detectors_nestedcv_rebuild95.json (regions + vgenes, naive vs nested) to D/results/.
"""

import json, time, numpy as np, h5py, torch as th, pyarrow.parquet as pq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

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

ae = BatchTopKCrossCoder.from_pretrained(
    f"{CKPT}/cc_paired_small_s0_L20.pt", device=dev, dtype=th.float32
)
ae.eval()
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
s2f = {int(s): int(seqfold[i]) for i, s in enumerate(seqs_test)}
log(f"test residues {len(idx)}, test seqs {len(seqs_test)}")
vtab = pq.read_table(f"{BASE}/data/exp_paired_small.parquet", columns=["v_call_heavy"]).to_pydict()[
    "v_call_heavy"
]


def gene(v):
    return None if v is None else v.split(",")[0].split("*")[0]


seq_gene = np.array([gene(v) or "NA" for v in vtab])
res_fold = np.full(N, -1, dtype=np.int8)
pos = np.searchsorted(seqs_test, seqid)
pos = np.clip(pos, 0, len(seqs_test) - 1)
is_t = test & (seqs_test[pos] == seqid)
res_fold[is_t] = seqfold[pos[is_t]]
sid2row = {int(s): i for i, s in enumerate(seqs_test)}
nS = len(seqs_test)
sumH = th.zeros(nS, DICT, device=dev, dtype=th.float32)
cntH = th.zeros(nS, DICT, device=dev, dtype=th.float32)
EDGES = np.concatenate([[0], np.logspace(-3, 1.0, NBIN - 1)]).astype(np.float32)
edges_t = th.tensor(EDGES, device=dev)
hist = th.zeros(NFOLD, DICT, 7, NBIN, device="cpu", dtype=th.float32)
regtot = th.zeros(NFOLD, 7, device="cpu", dtype=th.float64)
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
        comp = np.array([sid2row[int(v)] for v in seqid[gi]])
        cp = th.tensor(comp, device=dev)
        rg = th.tensor(reg.astype(np.int64), device=dev)
        fd = th.tensor(fo.astype(np.int64), device=dev)
        for b0 in range(0, e - s, ENC):
            b1 = min(b0 + ENC, e - s)
            xb = th.tensor(xb_all[b0:b1].astype(np.float32), device=dev)
            with th.no_grad():
                f = ae.encode(xb, use_threshold=True, normalize_activations=True).float()
            cpb = cp[b0:b1]
            nz = (f > 0).float()
            sumH.index_add_(0, cpb, f * nz)
            cntH.index_add_(0, cpb, nz)
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
            log(f"  encode {e}/{len(idx)} {time.time()-t0:.0f}s")
log(f"encode done {time.time()-t0:.0f}s")
scoreH = (sumH / th.clamp(cntH, min=1.0)).cpu().numpy()
gene_of_seq = np.array([seq_gene[int(s)] for s in seqs_test])
fold_of_seq = np.array([s2f[int(s)] for s in seqs_test])


def crit(pi):
    return 2 * pi / (1 + pi) + MARGIN


# ---------- REGION: naive (same-fold) vs nested-CV ----------
def region_suf(H):  # H (DICT,7,NBIN) -> suffix-sum over bins (TP at each threshold)
    return th.flip(th.cumsum(th.flip(H, [2]), dim=2), [2])


region_naive = np.zeros((NFOLD, 7))
region_nested = np.full((NFOLD, 7), np.nan)
region_floor = np.zeros((NFOLD, 7))
for fk in range(NFOLD):
    # --- naive: select+score on fold fk ---
    Hk = hist[fk].to(dev).double()
    suf = region_suf(Hk)
    predpos = suf.sum(1)
    pos_r = regtot[fk].to(dev)
    Ntot = float(pos_r.sum())
    prec = suf / th.clamp(predpos[:, None, :], min=1e-9)
    rec = suf / th.clamp(pos_r[None, :, None], min=1e-9)
    f1 = 2 * prec * rec / th.clamp(prec + rec, min=1e-9)
    f1max = f1.max(2).values  # (DICT,7)
    pi_r = (pos_r / Ntot).cpu().numpy()
    fl = crit(pi_r)
    region_floor[fk] = fl
    for r_ in range(7):
        region_naive[fk, r_] = float(f1max[:, r_].max().cpu())
    # --- nested: SELECT (latent,bin) on train (folds != fk), EVAL on fold fk ---
    Htr = sum(hist[f] for f in range(NFOLD) if f != fk).to(dev).double()
    suf_tr = region_suf(Htr)
    predpos_tr = suf_tr.sum(1)
    pos_tr = sum(regtot[f] for f in range(NFOLD) if f != fk).to(dev)
    prec_tr = suf_tr / th.clamp(predpos_tr[:, None, :], min=1e-9)
    rec_tr = suf_tr / th.clamp(pos_tr[None, :, None], min=1e-9)
    f1_tr = 2 * prec_tr * rec_tr / th.clamp(prec_tr + rec_tr, min=1e-9)  # (DICT,7,NBIN)
    predpos_k = suf.sum(1)  # fold-fk predpos per (latent,bin)
    for r_ in range(7):
        f1r = f1_tr[:, r_, :]  # (DICT,NBIN) train F1
        flat = int(th.argmax(f1r).cpu())
        dstar = flat // NBIN
        bstar = flat % NBIN  # best (latent,bin) on train
        tp = suf[dstar, r_, bstar]
        pp = predpos_k[dstar, bstar]
        posr = pos_r[r_]
        pr = tp / th.clamp(pp, min=1e-9)
        rc = tp / th.clamp(posr, min=1e-9)
        region_nested[fk, r_] = float((2 * pr * rc / th.clamp(pr + rc, min=1e-9)).cpu())

# ---------- V-GENE: naive (same-fold) vs nested-CV ----------
import collections

gc = collections.Counter(gene_of_seq)
genes = sorted([g for g, n in gc.items() if g != "NA" and n / nS >= MIN_PREV])


def per_latent_bestf1_and_rank(score, y):
    """score (n,DICT), y (n,). Returns best_f1_per_latent (DICT,), rank_per_latent (DICT,) (0-based)."""
    P = int(y.sum())
    sc = th.tensor(score, device=dev)
    yb = th.tensor(y.astype(np.float32), device=dev)
    order = th.argsort(sc, dim=0, descending=True)
    ys = th.gather(yb[:, None].expand(-1, sc.shape[1]), 0, order)
    cumTP = th.cumsum(ys, 0)
    k = th.arange(1, sc.shape[0] + 1, device=dev, dtype=th.float32)[:, None]
    prec = cumTP / k
    rec = cumTP / max(P, 1)
    f1 = 2 * prec * rec / th.clamp(prec + rec, min=1e-9)
    bestf1, bestrank = f1.max(0)  # per latent
    return bestf1, bestrank, order, sc


gene_naive = np.full((NFOLD, len(genes)), np.nan)
gene_nested = np.full((NFOLD, len(genes)), np.nan)
gene_floor = np.zeros((NFOLD, len(genes)))
for fk in range(NFOLD):
    tr = fold_of_seq != fk
    te = fold_of_seq == fk
    sc_tr = scoreH[tr]
    sc_te = scoreH[te]
    gof_tr = gene_of_seq[tr]
    gof_te = gene_of_seq[te]
    for j, g in enumerate(genes):
        y_all = gene_of_seq == g
        pi = y_all.mean()
        fl = crit(pi)
        gene_floor[fk, j] = fl
        y_tr = gof_tr == g
        y_te = gof_te == g
        # naive: select+score on held-out fold fk (reproduces original)
        y_fk = gene_of_seq[te] == g
        if int(y_fk.sum()) >= MIN_POS:
            bf, _, _, _ = per_latent_bestf1_and_rank(sc_te, y_fk)
            gene_naive[fk, j] = float(bf.max().cpu())
        # nested: select (latent,threshold) on TRAIN, evaluate on held-out fold fk
        if int(y_tr.sum()) < MIN_POS or int(y_te.sum()) < 1:
            continue
        bf, br, order, sc_tr_t = per_latent_bestf1_and_rank(sc_tr, y_tr)
        dstar = int(th.argmax(bf).cpu())
        rstar = int(br[dstar].cpu())
        col = sc_tr_t[:, dstar]
        sorted_desc, _ = th.sort(col, descending=True)
        cutoff = float(sorted_desc[rstar].cpu())
        # evaluate fixed (dstar, cutoff) on held-out fold
        pred = sc_te[:, dstar] >= cutoff
        P = int(y_te.sum())
        tp = int((pred & y_te).sum())
        pp = int(pred.sum())
        prec = tp / pp if pp > 0 else 0.0
        rec = tp / P if P > 0 else 0.0
        gene_nested[fk, j] = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    log(f"  vgene fold {fk} done")


def summ(arr):
    v = arr[~np.isnan(arr)]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if v.size > 1 else 0.0, int(v.size))


out = {
    "n_folds": NFOLD,
    "checkpoint": "cc_paired_small_s0_L20.pt",
    "store": "exp_paired_small_L20.h5",
    "note": "naive=same-fold best-case max-F1 (matches ab_f1_folds_rebuild95); nested=select latent+threshold on 4 train folds, evaluate on held-out fold (unbiased). floor=2pi/(1+pi)+0.2.",
    "regions": {},
    "vgenes": {},
}
for r_, rn in enumerate(REGION_NAMES):
    nm = summ(region_naive[:, r_])
    ne = summ(region_nested[:, r_])
    out["regions"][rn] = {
        "naive_f1_mean": nm[0],
        "naive_f1_sd": nm[1],
        "nested_f1_mean": ne[0],
        "nested_f1_sd": ne[1],
        "floor_mean": float(np.mean(region_floor[:, r_])),
        "nested_detect_in_k_of_5": int(
            np.nansum([(region_nested[fk, r_] > region_floor[fk, r_]) for fk in range(NFOLD)])
        ),
    }
for j, g in enumerate(genes):
    if np.all(np.isnan(gene_naive[:, j])) and np.all(np.isnan(gene_nested[:, j])):
        continue
    nm = summ(gene_naive[:, j])
    ne = summ(gene_nested[:, j])
    out["vgenes"][g] = {
        "naive_f1_mean": nm[0],
        "naive_f1_sd": nm[1],
        "nested_f1_mean": ne[0],
        "nested_f1_sd": ne[1],
        "n_folds_valid": ne[2],
        "floor_mean": float(np.mean(gene_floor[:, j])),
        "nested_detect_in_k_of_5": int(
            np.nansum(
                [
                    (gene_nested[fk, j] > gene_floor[fk, j])
                    for fk in range(NFOLD)
                    if not np.isnan(gene_nested[fk, j])
                ]
            )
        ),
    }
# aggregate
vg_naive = np.array([out["vgenes"][g]["naive_f1_mean"] for g in out["vgenes"]])
vg_nested = np.array([out["vgenes"][g]["nested_f1_mean"] for g in out["vgenes"]])
out["vgene_aggregate"] = {
    "n_genes": len(out["vgenes"]),
    "median_naive_f1": float(np.median(vg_naive)),
    "median_nested_f1": float(np.median(vg_nested)),
    "mean_naive_f1": float(np.mean(vg_naive)),
    "mean_nested_f1": float(np.mean(vg_nested)),
    "mean_drop": float(np.mean(vg_naive - vg_nested)),
}
json.dump(out, open(f"{RES}/ab_detectors_nestedcv_rebuild95.json", "w"), indent=1)
log("DONE")
print(json.dumps({"regions": out["regions"], "vgene_aggregate": out["vgene_aggregate"]}, indent=1))
