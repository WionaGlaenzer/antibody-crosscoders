#!/usr/bin/env python3
"""Do latents detect antibody's conserved residues? The canonical invariant positions are the two
framework cysteines (intradomain disulfide), the conserved FR2 tryptophan, and the FR4 WGxG glycines.
We define per-residue conserved-residue concepts by (amino acid identity + region), then run the
InterPLM max-F1 sweep (Haque floor) to see if dedicated latents fire on them.
Writes conserved_residues.json."""

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
MIN_POS = 15
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_paired_L20.pt", device=dev, dtype=th.float32)
ae.eval()
DICT = 32768
REGION_MAP = {0: "FR1", 1: "CDR1", 2: "FR2", 3: "CDR2", 4: "FR3", 5: "CDR3", 6: "FR4"}

# --- per-sequence amino acids from parquet (heavy + light) ---
tab = pq.read_table(
    f"{BASE}/data/oas_full_annotated.parquet", columns=["heavy_aa", "light_aa"]
).to_pydict()
H = tab["heavy_aa"]
L = tab["light_aa"]

h5 = f"{ACTS}/full_paired_L20.h5"
with h5py.File(h5, "r") as h:
    split = h["split"][:]
    seqid = h["seq_id"][:]
    region = h["region_label"][:]
    chain = h["chain"][:]
N = split.shape[0]
test = split == 2
idx = np.nonzero(test)[0]
Nt = len(idx)
log(f"test residues {Nt}")

# --- reconstruct per-residue amino acid: store is grouped by seq_id, heavy block then light block ---
# within a sequence, heavy residues (chain b'H') come first in order, then light (b'L')
aa = np.full(Nt, "X", dtype="U1")
# build per-seq residue index: for each test residue, its ordinal within its (seq,chain) block
order = np.lexsort((np.arange(N), seqid))  # not needed; store is already ordered
# process by contiguous seq blocks within test
t0 = time.time()
# map global index -> aa by walking each seq_id's H then L residues
uind = idx
# group test indices by seq_id preserving order
seq_of = seqid[uind]
ch_of = chain[uind]
# per (seq) counters
from collections import defaultdict

hcount = defaultdict(int)
lcount = defaultdict(int)
for j in range(Nt):
    s = int(seq_of[j])
    c = ch_of[j]
    if c == b"H":
        seqaa = H[s]
        p = hcount[s]
        hcount[s] += 1
    else:
        seqaa = L[s]
        p = lcount[s]
        lcount[s] += 1
    if seqaa is not None and p < len(seqaa):
        aa[j] = seqaa[p]
    if j % 1000000 == 0:
        log(f"  aa decode {j}/{Nt} {time.time()-t0:.0f}s")
reg = region[uind]
ch = ch_of
log("aa decoded")

# --- define conserved-residue concept masks ---
# Cys in framework (FR1..FR4) = the conserved disulfide cysteines
cys_fw = (aa == "C") & np.isin(reg, [0, 2, 4, 6])
# conserved FR2 Trp
trp_fr2 = (aa == "W") & (reg == 2)
# FR4 WGxG glycines: Gly in FR4
gly_fr4 = (aa == "G") & (reg == 6)
# also all Cys anywhere (for comparison)
cys_all = aa == "C"
concepts = {"Cys_framework": cys_fw, "Trp_FR2": trp_fr2, "Gly_FR4": gly_fr4, "Cys_all": cys_all}
for k, m in concepts.items():
    log(f"  {k}: {int(m.sum())} residues ({100*m.mean():.3f}%)")

# --- encode + per-residue max-F1 sweep (sparse, like region null) ---
# collect nonzero (latent, act, residue) then exact cumulative-F1 sweep per concept
lat_b = []
act_b = []
res_b = []
BLK = 200000
ENC = 16384
tot = 0
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    for s in range(0, Nt, BLK):
        e = min(s + BLK, Nt)
        gi = uind[s:e]
        x = acts[gi.min() : gi.max() + 1]
        take = gi - gi.min()
        xb_all = x[take]
        for b0 in range(0, e - s, ENC):
            b1 = min(b0 + ENC, e - s)
            xb = th.tensor(xb_all[b0:b1].astype(np.float32), device=dev)
            with th.no_grad():
                f = ae.encode(xb, use_threshold=True, normalize_activations=True)
            nz = f.nonzero(as_tuple=True)
            ri, li = nz
            lat_b.append(li.to(th.int32).cpu().numpy())
            act_b.append(f[ri, li].cpu().numpy())
            res_b.append((s + b0 + ri).cpu().numpy().astype(np.int64))
            tot += len(ri)
        if (s // BLK) % 15 == 0:
            log(f"  encode {e}/{Nt} nz={tot} {time.time()-t0:.0f}s")
lat = np.concatenate(lat_b)
act = np.concatenate(act_b)
res = np.concatenate(res_b)
del lat_b, act_b, res_b
log(f"nonzeros {len(lat)}")
# sort by latent then act desc
o = np.lexsort((-act, lat))
lat_s = lat[o]
res_s = res[o]
la_u, la_first, la_cnt = np.unique(lat_s, return_index=True, return_counts=True)


def sweep(mask):
    P = int(mask.sum())
    if P < MIN_POS:
        return np.zeros(DICT), 0.0
    pi = mask.mean()
    floor = 2 * pi / (1 + pi) + MARGIN
    bestf1 = np.zeros(DICT)
    for Lt, f0, cc in zip(la_u, la_first, la_cnt):
        r_res = res_s[f0 : f0 + cc]
        y = mask[r_res].astype(np.float64)
        tp = np.cumsum(y)
        k = np.arange(1, cc + 1)
        prec = tp / k
        rec = tp / P
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
        bestf1[int(Lt)] = f1.max()
    return bestf1, floor


# latent categories from classification
import csv as _csv

catmap = {}
try:
    for row in _csv.DictReader(open(f"{RES}/classification_paired_L20.csv")):
        catmap[int(row["latent_id"])] = row["category"]
except Exception as e:
    log("no cls csv:", e)
from collections import Counter

out = {}
for name, mask in concepts.items():
    bf, floor = sweep(mask)
    det = np.nonzero(bf > floor)[0]
    ndet = int(len(det))
    best = float(bf.max())
    bestlat = int(bf.argmax())
    catc = dict(Counter(catmap.get(int(i), "unknown") for i in det))
    out[name] = {
        "n_residues": int(mask.sum()),
        "prevalence": float(mask.mean()),
        "floor": float(floor),
        "n_detectors": ndet,
        "best_F1": best,
        "best_latent": bestlat,
        "best_latent_category": catmap.get(bestlat, "?"),
        "detector_categories": catc,
        "detector_latents": [int(i) for i in det],
        "top5": [[int(i), float(bf[i]), catmap.get(int(i), "?")] for i in np.argsort(-bf)[:5]],
    }
    log(
        f"{name}: ndet={ndet} bestF1={best:.3f} bestlat={bestlat}({catmap.get(bestlat,'?')}) cats={catc}"
    )
json.dump(out, open(f"{RES}/conserved_residues.json", "w"))
log("DONE")
print(
    json.dumps(
        {
            k: {
                "n_det": v["n_detectors"],
                "bestF1": round(v["best_F1"], 3),
                "bestlat": v["best_latent"],
            }
            for k, v in out.items()
        },
        indent=1,
    )
)
