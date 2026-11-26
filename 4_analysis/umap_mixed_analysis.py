#!/usr/bin/env python3
"""Fig3 e/f/g inputs on rebuild95 MIXED crosscoder (cc_mixed_small_s0_L20) — CORRECTED.
active = fires on ANY test residue (matches training-summary n_active).
concept F1: positional alignment — the k-th general-test residue in STORE ORDER maps to
concept_labels_general95.npz labels[k] (that npz was built as gt_idx=nonzero(gen_test), same order).
"""

import numpy as np, h5py, torch as th, json, time, os, sys

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


DICT = 32768
NBIN = 40
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(
    f"{CKPT}/cc_mixed_small_s0_L20.pt", device=dev, dtype=th.float32
)
ae.eval()
log("loaded mixed s0")

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

Z = np.load(f"{RES}/concept_labels_general95.npz")
labels = Z["labels"]
concepts = [str(x) for x in Z["concepts"]]
NCON = len(concepts)

h5 = f"{ACTS}/exp_mixed_small_L20.h5"
with h5py.File(h5, "r") as h:
    sp = h["split"][:]
    kd = h["kind"][:]
test = np.nonzero(sp == 2)[0]
gen_test_rows = np.nonzero((sp == 2) & (kd == 1))[0]  # STORE ORDER == labels order
assert len(gen_test_rows) == len(labels), (len(gen_test_rows), len(labels))
# map store-row -> label-row index (position in gen_test_rows)
genrow_to_labidx = {int(r): i for i, r in enumerate(gen_test_rows)}

any_cnt = th.zeros(DICT, device=dev)
ab_cnt = th.zeros(DICT, device=dev)
gen_cnt = th.zeros(DICT, device=dev)
nany = 0
nab = 0
ngen = 0
edges = np.concatenate([[0], np.logspace(-3, 1.0, NBIN - 1)]).astype(np.float32)
edges_t = th.tensor(edges, device=dev)
hist = th.zeros(DICT, NCON, NBIN, device="cpu", dtype=th.float32)
contot = th.zeros(NCON, device="cpu", dtype=th.float64)
hist_all = th.zeros(DICT, NBIN, device="cpu", dtype=th.float32)
n_labeled_res = 0
BLK = 200000
ENC = 8192
t0 = time.time()
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    for s in range(0, len(test), BLK):
        gi = test[s : s + BLK]
        lo = gi.min()
        x = acts[lo : gi.max() + 1]
        xb_all = x[gi - lo]
        kk = kd[gi]
        # label rows for the general residues in this block (or -1)
        labidx = np.array([genrow_to_labidx.get(int(r), -1) for r in gi], dtype=np.int64)
        for b0 in range(0, len(gi), ENC):
            b1 = min(b0 + ENC, len(gi))
            xb = th.tensor(xb_all[b0:b1].astype(np.float32), device=dev)
            with th.no_grad():
                f = ae.encode(xb, use_threshold=True, normalize_activations=True).float()
            nz = (f > 0).float()
            any_cnt += nz.sum(0)
            nany += xb.shape[0]
            kb = kk[b0:b1]
            abm = th.tensor(kb == 0, device=dev)
            genm = th.tensor(kb == 1, device=dev)
            if abm.any():
                ab_cnt += nz[abm].sum(0)
                nab += int(abm.sum())
            if genm.any():
                gen_cnt += nz[genm].sum(0)
                ngen += int(genm.sum())
            li = labidx[b0:b1]
            has = li >= 0
            if has.any():
                idxb = np.nonzero(has)[0]
                lab_b = labels[li[idxb]]
                fb = f[th.tensor(idxb, device=dev)]
                binidx = th.bucketize(fb, edges_t, right=False).clamp(0, NBIN - 1)
                nzmask = fb > 0
                labt = th.tensor(lab_b, device=dev)
                contot += labt.double().sum(0).cpu()
                # all-labeled-residue firing histogram (concept-independent denominator)
                bi_all = binidx
                nm_all = nzmask
                if nm_all.any():
                    flat = (
                        th.arange(DICT, device=dev)[None, :].expand(bi_all.shape[0], DICT)
                    ) * NBIN + bi_all
                    add = th.zeros(DICT * NBIN, device=dev, dtype=th.float32)
                    add.index_add_(
                        0,
                        (flat[nm_all]).reshape(-1),
                        th.ones(int(nm_all.sum()), device=dev, dtype=th.float32),
                    )
                    hist_all += add.reshape(DICT, NBIN).cpu()
                    n_labeled_res += int(bi_all.shape[0])
                for cix in range(NCON):
                    cm = labt[:, cix]
                    if not cm.any():
                        continue
                    bi = binidx[cm]
                    nm = nzmask[cm]
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
                    hist[:, cix, :] += add.reshape(DICT, NBIN).cpu()
        if (s // BLK) % 10 == 0:
            log(f"  encode {s+len(gi)}/{len(test)} {time.time()-t0:.0f}s")
any_freq = (any_cnt / max(nany, 1)).cpu().numpy()
ab_freq = (ab_cnt / max(nab, 1)).cpu().numpy()
gen_freq = (gen_cnt / max(ngen, 1)).cpu().numpy()
H = hist.to(dev).double()
suf = th.flip(th.cumsum(th.flip(H, [2]), dim=2), [2])
predpos = suf.sum(1)  # sum over concepts -> per-latent
pos_c = contot.to(dev)
tp = suf
prec = tp / th.clamp(predpos[:, None, :], min=1e-9)
rec = tp / th.clamp(pos_c[None, :, None], min=1e-9)
f1 = 2 * prec * rec / th.clamp(prec + rec, min=1e-9)
f1max_con = f1.max(2).values.cpu().numpy()
best_lat = np.array([int(np.argmax(f1max_con[:, c])) for c in range(NCON)])
bestf1 = np.array([float(f1max_con[:, c].max()) for c in range(NCON)])
pi_c = pos_c.cpu().numpy() / float(pos_c.sum())
floor_c = 2 * pi_c / (1 + pi_c) + 0.2

active = np.nonzero(any_freq > 1e-4)[0]
import umap
from sklearn.cluster import HDBSCAN

D0 = W[0].cpu().numpy()
Xn = D0[active] / (np.linalg.norm(D0[active], axis=1, keepdims=True) + 1e-9)
log("UMAP fit", len(active))
emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0).fit_transform(Xn)
clus = HDBSCAN(min_cluster_size=15, min_samples=5).fit_predict(Xn)
np.savez(
    "umap_mixed_rebuild95.npz",
    active=active,
    x=emb[:, 0],
    y=emb[:, 1],
    cluster=clus,
    cat=cat[active],
    rel=rel_np[active],
    any_freq=any_freq[active],
    ab_freq=ab_freq[active],
    gen_freq=gen_freq[active],
    concepts=np.array(concepts),
    best_lat_per_concept=best_lat,
    bestf1_per_concept=bestf1,
    floor_concept=floor_c,
    f1max_con_active=f1max_con[active],
)  # (n_active, NCON) per-latent concept F1 for ALL detectors
log("DONE n_active", len(active))
log("concept bestF1:", {concepts[c]: round(float(bestf1[c]), 3) for c in range(NCON)})
