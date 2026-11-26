#!/usr/bin/env python3
"""5-fold cross-validated confidence for the antibody F1 detector claims (paired crosscoder,
cc_paired_L20, held-out TEST split). Splits test SEQUENCES into 5 equal folds (by sequence),
and per fold computes:
  (1) Region detectors: per-residue swept-threshold max-F1 per antibody region (7-class), floor 2pi/(1+pi)+0.2
  (2) V-gene detectors: per-sequence mean-of-nonzero latent score, one-vs-rest max-F1 per heavy V-gene
Reports per-claim: best-latent F1 mean/sd across folds and #folds (0-5) clearing the floor,
plus detector-count mean/sd. Writes ab_f1_folds.json."""

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

ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_paired_L20.pt", device=dev, dtype=th.float32)
ae.eval()
h5 = f"{ACTS}/full_paired_L20.h5"
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
# per-seq V gene labels
vtab = pq.read_table(
    f"{BASE}/data/oas_full_annotated.parquet", columns=["v_call_heavy"]
).to_pydict()["v_call_heavy"]


def gene(v):
    return None if v is None else v.split(",")[0].split("*")[0]


seq_gene = np.array([gene(v) or "NA" for v in vtab])
# ---- ENCODE PASS: collect per-residue (fold, region, latent-nonzero-act) for regions,
#      and accumulate per-sequence mean-of-nonzero scores for genes ----
# region: per-fold swept histogram  hist[fold, latent, region, bin]  is too big (5*32768*7*40).
# Instead accumulate per fold separately using activation-bin edges from a first quick pass max.
# Simpler exact approach: per fold, per region, we need max-F1 over thresholds. Use fixed log bins.
# activation upper bound: probe max on a sample.
seqids_all = seqid
# map residue -> fold
res_fold = np.full(N, -1, dtype=np.int8)
pos = np.searchsorted(seqs_test, seqid)
pos = np.clip(pos, 0, len(seqs_test) - 1)
is_t = test & (seqs_test[pos] == seqid)
res_fold[is_t] = seqfold[pos[is_t]]
# per-seq compact index for gene scoring
sid2row = {int(s): i for i, s in enumerate(seqs_test)}
nS = len(seqs_test)
sumH = th.zeros(nS, DICT, device=dev, dtype=th.float32)
cntH = th.zeros(nS, DICT, device=dev, dtype=th.float32)
# region swept histogram per fold: counts of (latent act in bin b, region r) and region totals per fold
EDGES = np.concatenate([[0], np.logspace(-3, 1.0, NBIN - 1)]).astype(
    np.float32
)  # bin edges for activation
edges_t = th.tensor(EDGES, device=dev)
hist = th.zeros(
    NFOLD, DICT, 7, NBIN, device="cpu", dtype=th.float32
)  # CPU to save GPU mem; reverse-cumsum later
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
            # region: for each residue, its fold & region; bin its per-latent activations
            rgb = rg[b0:b1]
            fdb = fd[b0:b1]
            valid = (rgb >= 0) & (rgb < 7) & (fdb >= 0)
            if valid.any():
                fv = f[valid]
                rv = rgb[valid]
                dv = fdb[valid]
                # region totals per fold
                for fk in range(NFOLD):
                    fm = dv == fk
                    if fm.any():
                        rr = rv[fm]
                        # region totals per fold
                        oh = th.zeros(int(fm.sum()), 7, device=dev, dtype=th.float64)
                        oh[th.arange(int(fm.sum()), device=dev), rr] = 1.0
                        regtot[fk] += oh.sum(0).cpu()
                        sub = fv[fm]  # (nr, DICT)
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
                            # build the per-(latent,bin) counts on GPU for this slice, then add to CPU hist
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


# ---- REGION per-fold swept max-F1 ----
def crit(pi):
    return 2 * pi / (1 + pi) + MARGIN


region_best = np.zeros((NFOLD, 7))
region_ndet = np.zeros((NFOLD, 7), dtype=int)
region_floor = np.zeros((NFOLD, 7))
for fk in range(NFOLD):
    H = (
        hist[fk].to(dev).double()
    )  # (DICT,7,NBIN) counts of nonzero acts per bin (move fold slice to GPU)
    suf = th.flip(th.cumsum(th.flip(H, [2]), dim=2), [2])  # residues with act>=edge  (DICT,7,NBIN)
    predpos = suf.sum(1)  # (DICT,NBIN) latent fires>=t on any region
    pos_r = regtot[fk].to(dev)  # (7,)
    Ntot = float(pos_r.sum())
    tp = suf  # (DICT,7,NBIN)
    prec = tp / th.clamp(predpos[:, None, :], min=1e-9)
    rec = tp / th.clamp(pos_r[None, :, None], min=1e-9)
    f1 = 2 * prec * rec / th.clamp(prec + rec, min=1e-9)
    f1max = f1.max(2).values.cpu().numpy()  # (DICT,7)
    pi_r = (pos_r / Ntot).cpu().numpy()
    fl = crit(pi_r)
    region_floor[fk] = fl
    for r_ in range(7):
        region_best[fk, r_] = f1max[:, r_].max()
        region_ndet[fk, r_] = int((f1max[:, r_] > fl[r_]).sum())

# ---- V-GENE per-fold per-sequence max-F1 ----
import collections

gc = collections.Counter(gene_of_seq)
genes = sorted([g for g, n in gc.items() if g != "NA" and n / nS >= MIN_PREV])


def maxf1_fold(score, y):
    P = int(y.sum())
    if P < MIN_POS:
        return None
    sc = th.tensor(score, device=dev)
    yb = th.tensor(y.astype(np.float32), device=dev)
    order = th.argsort(sc, dim=0, descending=True)
    ys = yb[order]
    cumTP = th.cumsum(ys, 0)
    k = th.arange(1, score.shape[0] + 1, device=dev, dtype=th.float32)[:, None]
    prec = cumTP / k
    rec = cumTP / P
    f1 = 2 * prec * rec / th.clamp(prec + rec, min=1e-9)
    return f1.max(0).values.cpu().numpy()


gene_best = np.full((NFOLD, len(genes)), np.nan)
gene_ndet = np.full((NFOLD, len(genes)), np.nan)
gene_floor = np.zeros((NFOLD, len(genes)))
for fk in range(NFOLD):
    fm = fold_of_seq == fk
    sc = scoreH[fm]
    gof = gene_of_seq[fm]
    for j, g in enumerate(genes):
        y = gof == g
        pi = y.mean()
        fl = crit(pi)
        gene_floor[fk, j] = fl
        r = maxf1_fold(sc, y)
        if r is None:
            continue
        gene_best[fk, j] = r.max()
        gene_ndet[fk, j] = int((r > fl).sum())


def stab(best, floor):  # #folds clearing floor
    return int(np.sum([best[fk] > floor[fk] for fk in range(NFOLD)]))


out = {
    "n_folds": NFOLD,
    "fold_sizes_seq": [int((fold_of_seq == fk).sum()) for fk in range(NFOLD)],
    "regions": {},
    "vgenes": {},
}
for r_, rn in enumerate(REGION_NAMES):
    out["regions"][rn] = {
        "best_f1_mean": float(np.mean(region_best[:, r_])),
        "best_f1_sd": float(np.std(region_best[:, r_], ddof=1)),
        "floor_mean": float(np.mean(region_floor[:, r_])),
        "detect_in_k_of_5": stab(region_best[:, r_], region_floor[:, r_]),
        "ndet_mean": float(np.mean(region_ndet[:, r_])),
        "ndet_sd": float(np.std(region_ndet[:, r_], ddof=1)),
    }
for j, g in enumerate(genes):
    b = gene_best[:, j]
    fl = gene_floor[:, j]
    if np.all(np.isnan(b)):
        continue
    valid = ~np.isnan(b)
    out["vgenes"][g] = {
        "best_f1_mean": float(np.nanmean(b)),
        "best_f1_sd": float(np.nanstd(b, ddof=1)) if valid.sum() > 1 else 0.0,
        "floor_mean": float(np.mean(fl)),
        "detect_in_k_of_5": int(
            np.nansum([(b[fk] > fl[fk]) for fk in range(NFOLD) if not np.isnan(b[fk])])
        ),
        "n_folds_valid": int(valid.sum()),
        "ndet_mean": float(np.nanmean(gene_ndet[:, j])),
        "ndet_sd": float(np.nanstd(gene_ndet[:, j], ddof=1)) if valid.sum() > 1 else 0.0,
    }
# V-gene aggregate: median best-F1 across genes per fold, and how many genes detected per fold
gene_detected_per_fold = [
    int(np.nansum([(gene_best[fk, j] > gene_floor[fk, j]) for j in range(len(genes))]))
    for fk in range(NFOLD)
]
gene_median_f1_per_fold = [float(np.nanmedian(gene_best[fk])) for fk in range(NFOLD)]
out["vgene_aggregate"] = {
    "n_genes": len(genes),
    "genes_detected_per_fold": gene_detected_per_fold,
    "median_bestF1_per_fold": gene_median_f1_per_fold,
    "median_bestF1_mean": float(np.mean(gene_median_f1_per_fold)),
    "median_bestF1_sd": float(np.std(gene_median_f1_per_fold, ddof=1)),
}
json.dump(out, open(f"{RES}/ab_f1_folds.json", "w"), indent=1)
log("DONE")
print(json.dumps({"regions": out["regions"], "vgene_aggregate": out["vgene_aggregate"]}, indent=1))
