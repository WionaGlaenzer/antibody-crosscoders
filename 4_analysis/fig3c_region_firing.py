#!/usr/bin/env python3
"""Fig3c on rebuild95 MIXED crosscoder (cc_mixed_small_s0_L20), self-consistent:
 - build per-residue region labels for the mixed store's ANTIBODY residues via orig_id -> OAS pool
   region_labels_heavy/light (heavy residues then light residues, matching extractor order).
 - identify region detectors IN the mixed model (verified F1 method, predpos=suf.sum(1)).
 - compute those detectors' log2(antibody firing / general firing) in the SAME mixed model.
Writes fig3c_region_firing.npz."""

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

ae = BatchTopKCrossCoder.from_pretrained(
    f"{CKPT}/cc_mixed_small_s0_L20.pt", device=dev, dtype=th.float32
)
ae.eval()

# ---- category (canonical) ----
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

# ---- build region labels for mixed ab residues via orig_id -> OAS pool ----
mt = pq.read_table(
    f"{DATA}/exp_mixed_small.parquet", columns=["seq_id", "orig_id", "kind", "heavy_aa", "light_aa"]
).to_pydict()
op = pq.read_table(
    f"{DATA}/oas_pool_95nr.parquet", columns=["id", "region_labels_heavy", "region_labels_light"]
).to_pydict()
idmap = {op["id"][i]: i for i in range(len(op["id"]))}
# per antibody seq_id (=store seq_id), the concatenated region-code list (heavy then light)
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

# ---- encode mixed test residues; region hist on ab residues (with region label), firing on ab & gen ----
with h5py.File(f"{ACTS}/exp_mixed_small_L20.h5", "r") as h:
    sp = h["split"][:]
    kd = h["kind"][:]
    sid = h["seq_id"][:]
test = np.nonzero(sp == 2)[0]
# assign per-residue region code for ab test residues by walking each seq's block (store is seq-contiguous)
# build residue->regioncode for ab test residues; -1 for general
regcode = np.full(len(sp), -1, dtype=np.int8)
# per ab sequence, its residues in store order = its heavy+light residues in order
ab_all = np.nonzero(kd == 0)[0]
# group ab_all by seq_id preserving order
order = np.argsort(sid[ab_all], kind="stable")
ab_sorted = ab_all[order]
s_sorted = sid[ab_sorted]
# boundaries
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

fire_ab = th.zeros(DICT, device=dev, dtype=th.float64)
fire_gen = th.zeros(DICT, device=dev, dtype=th.float64)
nab = 0
ngen = 0
hist = th.zeros(DICT, 7, NBIN, device="cpu", dtype=th.float32)
regtot = th.zeros(7, device="cpu", dtype=th.float64)
EDGES = np.concatenate([[0], np.logspace(-3, 1.0, NBIN - 1)]).astype(np.float32)
edges_t = th.tensor(EDGES, device=dev)
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
        for b0 in range(0, len(gi), ENC):
            b1 = min(b0 + ENC, len(gi))
            xb = th.tensor(xb_all[b0:b1].astype(np.float32), device=dev)
            with th.no_grad():
                f = ae.encode(xb, use_threshold=True, normalize_activations=True).float()
            nz = (f > 0).double()
            kb = kk[b0:b1]
            rcb = rc[b0:b1]
            am = th.tensor(kb == 0, device=dev)
            gm = th.tensor(kb == 1, device=dev)
            if am.any():
                fire_ab += nz[am].sum(0)
                nab += int(am.sum())
            if gm.any():
                fire_gen += nz[gm].sum(0)
                ngen += int(gm.sum())
            # region hist over ab residues with valid region code
            rct = th.tensor(rcb.astype(np.int64), device=dev)
            valid = (rct >= 0) & (rct < 7) & am
            if valid.any():
                fv = f[valid]
                rv = rct[valid]
                oh = th.zeros(int(valid.sum()), 7, device=dev, dtype=th.float64)
                oh[th.arange(int(valid.sum()), device=dev), rv] = 1.0
                regtot += oh.sum(0).cpu()
                binidx = th.bucketize(fv, edges_t, right=False).clamp(0, NBIN - 1)
                nzmask = fv > 0
                for r_ in th.unique(rv):
                    rm = rv == r_
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
                    hist[:, int(r_), :] += add.reshape(DICT, NBIN).cpu()
        if (s // BLK) % 10 == 0:
            log(f"  encode {s+len(gi)}/{len(test)} {time.time()-t0:.0f}s")
ab_freq = (fire_ab / max(nab, 1)).cpu().numpy()
gen_freq = (fire_gen / max(ngen, 1)).cpu().numpy()
# per-latent whole-test region F1 (verified math)
H = hist.to(dev).double()
suf = th.flip(th.cumsum(th.flip(H, [2]), dim=2), [2])
predpos = suf.sum(1)
pos_r = regtot.to(dev)
Ntot = float(pos_r.sum())
tp = suf
prec = tp / th.clamp(predpos[:, None, :], min=1e-9)
rec = tp / th.clamp(pos_r[None, :, None], min=1e-9)
f1 = 2 * prec * rec / th.clamp(prec + rec, min=1e-9)
f1max = f1.max(2).values.cpu().numpy()
floor = crit((pos_r / Ntot).cpu().numpy())
det = [int((f1max[:, r] > floor[r]).sum()) for r in range(7)]
np.savez(
    "fig3c_region_firing.npz",
    f1max=f1max,
    floor=floor,
    ab_freq=ab_freq,
    gen_freq=gen_freq,
    cat=cat,
    region_names=np.array(RN),
    nab=nab,
    ngen=ngen,
)
log(f"DONE mixed region det {dict(zip(RN,det))} maxF1 {dict(zip(RN,f1max.max(0).round(3)))}")
