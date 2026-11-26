#!/usr/bin/env python
"""Explicit per-residue SHM test: mark each HEAVY residue as germline-match(0) or
mutated(1) by ANARCI-IMGT-aligning to the assigned germline V allele, then measure
each latent's firing rate on mutated vs germline-matched residues (FR1-FR3+CDR1-2;
CDR3/FR4 excluded - not V-germline-encoded). Enrichment = firing_on_mutated /
firing_on_germline. Identifies latents that specifically detect mutated positions."""

import json, time, numpy as np, h5py, torch as th, pandas as pd, sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


from anarci import run_anarci, germlines as G

AG = G.all_germlines["V"]["H"]["human"]


def germ_for(vc):
    if not isinstance(vc, str) or not vc:
        return None
    g = vc.split(",")[0].strip()
    if g in AG:
        return AG[g]
    base = g.split("*")[0]
    for a in (base + "*01", base + "*02"):
        if a in AG:
            return AG[a]
    for k in AG:
        if k.split("*")[0] == base:
            return AG[k]
    return None


df = pd.read_parquet(
    f"{BASE}/data/oas_full_annotated.parquet", columns=["id", "heavy_aa", "v_call_heavy", "split"]
).reset_index(drop=True)
# only need test-split sequences (seq_id = row index)
# Build per-sequence heavy-residue mutation mask (list aligned to heavy_aa position order)
# ANARCI gives numbering; we walk the numbered residues in order and mark mutated vs germline.
test_rows = df.index[df["split"] == "test"].to_numpy()
log(f"test sequences {len(test_rows)}")
seqs = [(str(i), df.at[i, "heavy_aa"]) for i in test_rows if isinstance(df.at[i, "heavy_aa"], str)]
log("ANARCI numbering test heavy chains")
_anarci_res = run_anarci(seqs, scheme="imgt", ncpu=4)
numbering = _anarci_res[1]
# For each sequence, produce a mask over heavy_aa positions: 0=germline match,1=mutated,-1=not-V-scored
mut_mask = {}  # row_index -> np.int8 array length len(heavy_aa)
for (sid_s, seq), num in zip(seqs, numbering):
    ri = int(sid_s)
    L = len(seq)
    mask = np.full(L, -1, np.int8)
    germ = germ_for(df.at[ri, "v_call_heavy"])
    if num is None or germ is None:
        mut_mask[ri] = mask
        continue
    dom = num[0][0]  # [((pos,ins),aa),...] in sequence order for aligned region
    # walk numbered residues; they correspond to consecutive seq positions (skip '-')
    seqpos = 0
    # find start offset: ANARCI aligns from first matched residue; map by counting non-gap aa
    # Build ordered list of (aa) from numbering to align against seq via matching
    numbered = [(pos, ins, aa) for (pos, ins), aa in dom if aa != "-"]
    # numbered aa string should be a substring-in-order of seq; align greedily
    si = 0
    for pos, ins, aa in numbered:
        # advance si to match aa
        while si < L and seq[si] != aa:
            si += 1
        if si >= L:
            break
        if ins == " " and 1 <= pos < 105:
            g = germ[pos - 1] if pos - 1 < len(germ) else "-"
            if g != "-":
                mask[si] = 0 if seq[si] == g else 1
        si += 1
    mut_mask[ri] = mask
log(f"built masks for {len(mut_mask)} sequences")

# ---- encode store heavy residues, accumulate firing on mutated vs germline positions ----
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(
    f"{BASE}/checkpoints/cc_single_L20.pt", device="cuda", dtype=th.float32
)
ae.eval()
cls = pd.read_csv(f"{BASE}/results/classification_single_L20.csv")
active = cls["latent_id"].to_numpy().astype(np.int64)
A = len(active)
cat = cls["category"].to_numpy()
active_t = th.tensor(active, device="cuda")
H5 = f"{BASE}/activations/full_single_L20.h5"
with h5py.File(H5, "r") as f:
    tmean = th.tensor(f["train_mean"][:], device="cuda", dtype=th.float32)
    tstd = th.tensor(f["train_std"][:], device="cuda", dtype=th.float32)
    sp = f["split"][:]
    seqid = f["seq_id"][:]
    chain = f["chain"][:]
    acts = f["acts"]
    idx = np.nonzero((sp == 2) & (chain == b"H"))[0]  # heavy residues, test split
    log(f"heavy test residues {len(idx)}")
    # per-residue mutation label: need position within each sequence's heavy block
    # store residues are in order; for each seq_id the heavy residues appear in sequence order.
    # compute intra-sequence heavy index by cumulative counting
    sid_test = seqid[idx]
    # position within heavy block: for contiguous equal seq_id runs, enumerate
    heavy_pos = np.zeros(len(idx), np.int32)
    # since idx is sorted and heavy residues per seq are contiguous, compute run positions
    _, starts = np.unique(sid_test, return_index=True)
    order = np.argsort(starts)
    # simpler: iterate
    last = -1
    cnt = 0
    for i in range(len(idx)):
        s = sid_test[i]
        if s != last:
            cnt = 0
            last = s
        heavy_pos[i] = cnt
        cnt += 1
    # residue mutation label
    res_mut = np.full(len(idx), -1, np.int8)
    for i in range(len(idx)):
        m = mut_mask.get(int(sid_test[i]))
        if m is not None and heavy_pos[i] < len(m):
            res_mut[i] = m[heavy_pos[i]]
    fire_mut = np.zeros(A, np.int64)
    fire_germ = np.zeros(A, np.int64)
    n_mut = int((res_mut == 1).sum())
    n_germ = int((res_mut == 0).sum())
    log(
        f"mutated residues {n_mut} | germline residues {n_germ} | unscored {int((res_mut==-1).sum())}"
    )
    BLK = 8192
    for b0 in range(0, len(idx), BLK):
        bidx = idx[b0 : b0 + BLK]
        bm = res_mut[b0 : b0 + BLK]
        x = th.tensor(acts[bidx].astype(np.float32), device="cuda")
        xn = (x - tmean) / tstd
        fa = (ae.encode(xn, use_threshold=False)[:, active_t] > 0).cpu().numpy()
        fire_mut += fa[bm == 1].sum(0)
        fire_germ += fa[bm == 0].sum(0)
        if (b0 // BLK) % 40 == 0:
            log(f"  {b0+len(bidx)}/{len(idx)}")
freq_mut = fire_mut / max(n_mut, 1)
freq_germ = fire_germ / max(n_germ, 1)
enr = np.log2((freq_mut + 1e-6) / (freq_germ + 1e-6))
out = pd.DataFrame(
    {
        "latent_id": active,
        "category": cat,
        "freq_mutated": freq_mut,
        "freq_germline": freq_germ,
        "log2_mut_over_germ": enr,
    }
)
out.to_parquet(f"{BASE}/results/residue_mutation_firing.parquet")
summ = dict(n_mutated_res=n_mut, n_germline_res=n_germ)
for c in ["shared-all", "gained-at-unpaired"]:
    d = out[out.category == c]
    summ[c] = dict(
        n=int(len(d)),
        median_log2_mut_over_germ=float(d["log2_mut_over_germ"].median()),
        frac_mut_selective=float((d["log2_mut_over_germ"] > 1).mean()),
        frac_germ_selective=float((d["log2_mut_over_germ"] < -1).mean()),
    )
top = out.nlargest(25, "log2_mut_over_germ")[
    ["latent_id", "category", "freq_germline", "freq_mutated", "log2_mut_over_germ"]
]
summ["top25_mutation_selective"] = top.to_dict("records")
json.dump(summ, open(f"{BASE}/results/residue_mutation_summary.json", "w"), indent=2, default=float)
log("DONE")
print(
    json.dumps(
        {
            k: summ[k]
            for k in ["n_mutated_res", "n_germline_res", "shared-all", "gained-at-unpaired"]
        },
        indent=2,
        default=float,
    )
)
