#!/usr/bin/env python3
"""Gene-level heavy V detection: does cc_paired_L20 resolve individual V-genes (IGHV3-23) or
only families (IGHV3)? Per-sequence mean-of-nonzero latent scores, one-vs-rest max-F1 sweep per
gene, Haque prevalence-adjusted floor (F1 > 2pi/(1+pi)+0.2). Writes vgene_level_detection.json."""

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


MARGIN = 0.2
MIN_PREV = 0.005
MIN_POS = 15
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_paired_L20.pt", device=dev, dtype=th.float32)
ae.eval()
DICT = 32768

# --- per-sequence gene labels (heavy V gene, allele stripped) ---
tab = pq.read_table(
    f"{BASE}/data/oas_full_annotated.parquet", columns=["v_call_heavy"]
).to_pydict()["v_call_heavy"]


def gene(v):
    if v is None:
        return None
    return v.split(",")[0].split("*")[0]


seq_gene = np.array([gene(v) or "NA" for v in tab])  # per parquet-row (=seq_id)

# --- encode: per-sequence mean-of-nonzero latent score on TEST split ---
h5 = f"{ACTS}/full_paired_L20.h5"
with h5py.File(h5, "r") as h:
    split = h["split"][:]
    seqid = h["seq_id"][:]
    N = split.shape[0]
test = split == 2
idx = np.nonzero(test)[0]
seqids_test = np.unique(seqid[test])
sid2row = {int(s): i for i, s in enumerate(seqids_test)}
n_seq = len(seqids_test)
log(f"test residues {len(idx)}, test sequences {n_seq}")
sum_H = th.zeros(n_seq, DICT, device=dev, dtype=th.float64)
cnt_H = th.zeros(n_seq, DICT, device=dev, dtype=th.float64)
BLK = 200000
ENC = 16384
t0 = time.time()
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    for s in range(0, len(idx), BLK):
        e = min(s + BLK, len(idx))
        gi = idx[s:e]
        x = acts[gi.min() : gi.max() + 1]
        take = gi - gi.min()
        xb_all = x[take]
        comp = np.array([sid2row[int(v)] for v in seqid[gi]])
        cp = th.tensor(comp, device=dev)
        for b0 in range(0, e - s, ENC):
            b1 = min(b0 + ENC, e - s)
            xb = th.tensor(xb_all[b0:b1].astype(np.float32), device=dev)
            with th.no_grad():
                f = ae.encode(xb, use_threshold=True, normalize_activations=True).double()
            cpb = cp[b0:b1]
            nz = (f > 0).double()
            sum_H.index_add_(0, cpb, f * nz)
            cnt_H.index_add_(0, cpb, nz)
        if (s // BLK) % 10 == 0:
            log(f"  encode {e}/{len(idx)} {time.time()-t0:.0f}s")
scoreH = (sum_H / th.clamp(cnt_H, min=1.0)).cpu().numpy()  # (n_seq, DICT)
log(f"encode done {time.time()-t0:.0f}s")

# --- gene labels aligned to test sequences ---
gene_of_seq = np.array([seq_gene[int(s)] for s in seqids_test])
import collections

gc = collections.Counter(gene_of_seq)
genes = [g for g, n in gc.items() if g != "NA" and n / n_seq >= MIN_PREV]
genes = sorted(genes)
log(f"{len(genes)} genes with >= {MIN_PREV} prevalence")


# --- max-F1 one-vs-rest per gene (vectorized over latents) ---
def max_f1_all(score, y):
    # score (n_seq, DICT), y bool (n_seq,)
    P = int(y.sum())
    if P < MIN_POS:
        return np.zeros(DICT), 0
    sc = th.tensor(score, device=dev)
    order = th.argsort(sc, dim=0, descending=True)  # (n_seq,DICT)
    yb = th.tensor(y.astype(np.float32), device=dev)
    ys = yb[order]  # (n_seq,DICT)
    cumTP = th.cumsum(ys, dim=0)
    k = th.arange(1, score.shape[0] + 1, device=dev, dtype=th.float32)[:, None]
    prec = cumTP / k
    rec = cumTP / P
    f1 = 2 * prec * rec / th.clamp(prec + rec, min=1e-9)
    return f1.max(0).values.cpu().numpy(), P


# latent categories from classification CSV (paired_L20)
import csv as _csv

catmap = {}
try:
    for row in _csv.DictReader(open(f"{RES}/classification_paired_L20.csv")):
        catmap[int(row["latent_id"])] = row["category"]
except Exception as e:
    log("no classification csv:", e)
# mask of gained-at-unpaired latents (exclude to test: can SHARED latents alone resolve genes?)
gap_mask = np.array([catmap.get(i, "") == "gained-at-unpaired" for i in range(DICT)])
log(f"gained-at-unpaired latents excluded: {int(gap_mask.sum())}")


def cat_counts(det_idx):
    from collections import Counter

    cc = Counter(catmap.get(int(i), "unknown") for i in det_idx)
    return dict(cc)


results = {}
family_of = {}
gene_det_union = set()
maxf1_per_lat = np.zeros(DICT)  # max F1 across all genes, per latent (threshold-free)
for gi_, g in enumerate(genes):
    y = gene_of_seq == g
    pi = y.mean()
    floor = 2 * pi / (1 + pi) + MARGIN
    f1, P = max_f1_all(scoreH, y)
    maxf1_per_lat = np.maximum(maxf1_per_lat, f1)
    det_idx = np.nonzero(f1 > floor)[0]
    gene_det_union.update(int(i) for i in det_idx)
    ndet = int(len(det_idx))
    best = float(f1.max())
    bestlat = int(f1.argmax())
    # exclude gained-at-unpaired latents -> shared-only performance
    f1_excl = f1.copy()
    f1_excl[gap_mask] = -1
    best_excl = float(f1_excl.max())
    bestlat_excl = int(f1_excl.argmax())
    ndet_excl = int((f1_excl > floor).sum())
    # ONLY gained-at-unpaired latents -> gained-only performance
    f1_gonly = f1.copy()
    f1_gonly[~gap_mask] = -1
    best_gonly = float(f1_gonly.max())
    bestlat_gonly = int(f1_gonly.argmax())
    ndet_gonly = int((f1_gonly > floor).sum())
    fam = g.split("-")[0]
    results[g] = {
        "prevalence": float(pi),
        "floor": float(floor),
        "n_detectors": ndet,
        "best_F1": best,
        "best_latent": bestlat,
        "family": fam,
        "n_pos": P,
        "best_F1_shared_only": best_excl,
        "best_latent_shared_only": bestlat_excl,
        "n_detectors_shared_only": ndet_excl,
        "best_F1_gained_only": best_gonly,
        "best_latent_gained_only": bestlat_gonly,
        "n_detectors_gained_only": ndet_gonly,
        "detector_categories": cat_counts(det_idx),
    }
    family_of[g] = fam
    if gi_ % 5 == 0:
        log(f"  gene {gi_}/{len(genes)} {g} ndet={ndet} bestF1={best:.2f}")

# --- IGHV3 spotlight: within-family resolution ---
ighv3 = [g for g in genes if g.startswith("IGHV3-")]
spotlight = {}
for g in ighv3:
    spotlight[g] = results[g]
out = {
    "n_genes_tested": len(genes),
    "margin": MARGIN,
    "results": results,
    "gene_detector_latents": sorted(gene_det_union),
    "maxf1_per_latent": maxf1_per_lat.tolist(),
    "ighv3_genes": ighv3,
    "n_seq_test": n_seq,
    "summary": {
        "genes_with_detectors": sum(1 for g in results if results[g]["n_detectors"] > 0),
        "total_gene_detectors": sum(results[g]["n_detectors"] for g in results),
    },
}
json.dump(out, open(f"{RES}/vgene_level_detection.json", "w"))
log("DONE")
print(
    json.dumps(
        {
            "n_genes": len(genes),
            "genes_with_detectors": out["summary"]["genes_with_detectors"],
            "ighv3_spotlight": {
                g: (results[g]["n_detectors"], round(results[g]["best_F1"], 2)) for g in ighv3
            },
        },
        indent=1,
    )
)
