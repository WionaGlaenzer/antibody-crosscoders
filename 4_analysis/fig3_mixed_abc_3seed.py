#!/usr/bin/env python3
"""Fig3 a/b/c on the MIXED crosscoder (rebuild95/95%-NR), 3 seeds.
Reuses the verified region-label reconstruction (orig_id->OAS 95nr pool) and the
verified whole-test + per-fold F1 math from fig3c_region_firing.py / fig3_region_reuse.py.

Per seed dumps fig3_perlatent_mixed_s{seed}.npz with:
  f1max_wholetest (DICT,7), region_floor_wholetest(7), cat (DICT,), rel(DICT,3),
  ab_freq(DICT,), gen_freq(DICT,), region_names,
  region_best_fold(5,7), region_floor_fold(5,7), region_ndet_fold(5,7)
Panels: a = region_ndet (counts), b = per-latent f1max scatter, c = ab/gen firing density.
"""

import os, sys, json, time, numpy as np, h5py, torch as th, pyarrow.parquet as pq

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
os.environ["HF_HOME"] = f"{BASE}/caches/hf"
ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
DATA = f"{BASE}/data"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


DICT = 32768
NBIN = 40
MARGIN = 0.2
NFOLD = 5
REGION_MAP = {"FR1": 0, "CDR1": 1, "FR2": 2, "CDR2": 3, "FR3": 4, "CDR3": 5, "FR4": 6}
RN = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]


def crit(pi):
    return 2 * pi / (1 + pi) + MARGIN


from dictionary_learning.dictionary import BatchTopKCrossCoder

# ---- region labels for mixed ab residues via orig_id -> OAS 95nr pool (once) ----
mt = pq.read_table(
    f"{DATA}/exp_mixed_small.parquet", columns=["seq_id", "orig_id", "kind"]
).to_pydict()
op = pq.read_table(
    f"{DATA}/oas_pool_95nr.parquet", columns=["id", "region_labels_heavy", "region_labels_light"]
).to_pydict()
idmap = {op["id"][i]: i for i in range(len(op["id"]))}
seqreg = {}
nmiss = 0
for row in range(len(mt["kind"])):
    if mt["kind"][row] != 0:
        continue
    sid = int(mt["seq_id"][row])
    oid = mt["orig_id"][row]
    j = idmap.get(oid)
    if j is None:
        nmiss += 1
        continue
    rh = op["region_labels_heavy"][j]
    rl = op["region_labels_light"][j]
    codes = [REGION_MAP.get(x, -1) for x in (list(rh) if rh is not None else [])]
    if rl is not None:
        codes += [REGION_MAP.get(x, -1) for x in list(rl)]
    seqreg[sid] = np.array(codes, dtype=np.int8)
log(f"built region labels for {len(seqreg)} ab seqs, {nmiss} missing")

# ---- store metadata + per-residue regcode + fold assignment (test split) ----
with h5py.File(f"{ACTS}/exp_mixed_small_L20.h5", "r") as h:
    sp = h["split"][:]
    kd = h["kind"][:]
    sid = h["seq_id"][:]
    N = sp.shape[0]
test = np.nonzero(sp == 2)[0]
regcode = np.full(N, -1, dtype=np.int8)
ab_all = np.nonzero(kd == 0)[0]
order = np.argsort(sid[ab_all], kind="stable")
ab_sorted = ab_all[order]
s_sorted = sid[ab_sorted]
uniq, starts = np.unique(s_sorted, return_index=True)
starts = list(starts) + [len(ab_sorted)]
for u_i, u in enumerate(uniq):
    rows = ab_sorted[starts[u_i] : starts[u_i + 1]]
    rc = seqreg.get(int(u))
    if rc is None:
        continue
    m = min(len(rows), len(rc))
    regcode[rows[:m]] = rc[:m]
log("assigned regcode to ab residues")
# 5-fold by ab test SEQUENCE (region F1 folds) — only ab test seqs carry regions
ab_test_seqs = np.unique(sid[test][kd[test] == 0])
rng = np.random.default_rng(0)
perm = rng.permutation(len(ab_test_seqs))
seqfold = (perm % NFOLD).astype(np.int8)
sf_map = {int(s): int(f) for s, f in zip(ab_test_seqs, seqfold)}
res_fold = np.full(N, -1, dtype=np.int8)
for gi_ in test:
    if kd[gi_] == 0:
        f = sf_map.get(int(sid[gi_]), -1)
        res_fold[gi_] = f
log("fold assignment done, ab test seqs", len(ab_test_seqs))
EDGES = np.concatenate([[0], np.logspace(-3, 1.0, NBIN - 1)]).astype(np.float32)
edges_t = th.tensor(EDGES, device=dev)


def run_seed(seed):
    ck = f"{CKPT}/cc_mixed_small_s{seed}_L20.pt"
    ae = BatchTopKCrossCoder.from_pretrained(ck, device=dev, dtype=th.float32)
    ae.eval()
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
    fire_ab = th.zeros(DICT, device=dev, dtype=th.float64)
    fire_gen = th.zeros(DICT, device=dev, dtype=th.float64)
    nab = 0
    ngen = 0
    BLK = 200000
    ENC = 8192
    t0 = time.time()
    with h5py.File(f"{ACTS}/exp_mixed_small_L20.h5", "r") as h:
        acts = h["acts"]
        for s in range(0, len(test), BLK):
            gi = test[s : s + BLK]
            lo = gi.min()
            x = acts[lo : gi.max() + 1]
            xb_all = x[gi - lo]
            kk = kd[gi]
            rc = regcode[gi]
            fo = res_fold[gi]
            for b0 in range(0, len(gi), ENC):
                b1 = min(b0 + ENC, len(gi))
                xb = th.tensor(xb_all[b0:b1].astype(np.float32), device=dev)
                with th.no_grad():
                    f = ae.encode(xb, use_threshold=True, normalize_activations=True).float()
                nz = (f > 0).double()
                kb = kk[b0:b1]
                rcb = rc[b0:b1]
                fob = fo[b0:b1]
                am = th.tensor(kb == 0, device=dev)
                gm = th.tensor(kb == 1, device=dev)
                if am.any():
                    fire_ab += nz[am].sum(0)
                    nab += int(am.sum())
                if gm.any():
                    fire_gen += nz[gm].sum(0)
                    ngen += int(gm.sum())
                rct = th.tensor(rcb.astype(np.int64), device=dev)
                fdt = th.tensor(fob.astype(np.int64), device=dev)
                valid = (rct >= 0) & (rct < 7) & am & (fdt >= 0)
                if valid.any():
                    fv = f[valid]
                    rv = rct[valid]
                    dv = fdt[valid]
                    for fk in range(NFOLD):
                        fm = dv == fk
                        if not fm.any():
                            continue
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
                log(f"  seed{seed} encode {s+len(gi)}/{len(test)} {time.time()-t0:.0f}s")
    # per-fold summaries
    region_best = np.zeros((NFOLD, 7))
    region_ndet = np.zeros((NFOLD, 7), dtype=int)
    region_floor = np.zeros((NFOLD, 7))
    for fk in range(NFOLD):
        H = hist[fk].to(dev).double()
        suf = th.flip(th.cumsum(th.flip(H, [2]), dim=2), [2])
        predpos = suf.sum(1)
        pos_r = regtot[fk].to(dev)
        Ntot = float(pos_r.sum())
        prec = suf / th.clamp(predpos[:, None, :], min=1e-9)
        rec = suf / th.clamp(pos_r[None, :, None], min=1e-9)
        f1 = 2 * prec * rec / th.clamp(prec + rec, min=1e-9)
        f1max = f1.max(2).values.cpu().numpy()
        pi_r = (pos_r / Ntot).cpu().numpy()
        fl = crit(pi_r)
        region_floor[fk] = fl
        for r_ in range(7):
            region_best[fk, r_] = f1max[:, r_].max()
            region_ndet[fk, r_] = int((f1max[:, r_] > fl[r_]).sum())
    # whole-test per-latent f1max
    Hall = hist.sum(0).to(dev).double()
    suf = th.flip(th.cumsum(th.flip(Hall, [2]), dim=2), [2])
    predpos = suf.sum(1)
    pos_r = regtot.sum(0).to(dev)
    Ntot = float(pos_r.sum())
    prec = suf / th.clamp(predpos[:, None, :], min=1e-9)
    rec = suf / th.clamp(pos_r[None, :, None], min=1e-9)
    f1 = 2 * prec * rec / th.clamp(prec + rec, min=1e-9)
    f1max_wt = f1.max(2).values.cpu().numpy()
    floor_wt = crit((pos_r / Ntot).cpu().numpy())
    ab_freq = (fire_ab / max(nab, 1)).cpu().numpy()
    gen_freq = (fire_gen / max(ngen, 1)).cpu().numpy()
    np.savez(
        f"fig3_perlatent_mixed_s{seed}.npz",
        f1max_wholetest=f1max_wt,
        region_floor_wholetest=floor_wt,
        cat=cat,
        rel=rel_np,
        ab_freq=ab_freq,
        gen_freq=gen_freq,
        region_names=np.array(RN),
        region_best_fold=region_best,
        region_floor_fold=region_floor,
        region_ndet_fold=region_ndet,
        nab=nab,
        ngen=ngen,
    )
    det = [int((f1max_wt[:, r] > floor_wt[r]).sum()) for r in range(7)]
    log(
        f"seed{seed} DONE region ndet {dict(zip(RN,det))} maxF1 {dict(zip(RN,f1max_wt.max(0).round(3)))}"
    )


for sd in [0, 1, 2]:
    run_seed(sd)
log("ALL DONE")
