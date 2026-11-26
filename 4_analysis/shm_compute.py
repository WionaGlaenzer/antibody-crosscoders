#!/usr/bin/env python
"""Stage 1: compute somatic hypermutation (SHM) level per heavy chain.
Number each heavy_aa with ANARCI (IMGT), align position-by-position to the assigned
germline V gene (anarci offline germlines, IMGT-gapped 128), count AA mismatches over
V-covered positions (FR1-FR3+CDR1-2; CDR3/FR4 excluded - not V-germline-encoded).
Output per-sequence: v_identity (%), n_mismatch, n_aligned, shm_bin."""

import json, time, numpy as np, pandas as pd
from anarci import run_anarci, germlines as G


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


AG = G.all_germlines["V"]["H"]["human"]  # gene -> IMGT-gapped 128 AA

DF = f"{BASE}/data/oas_full_annotated.parquet"
df = pd.read_parquet(DF, columns=["id", "heavy_aa", "v_call_heavy", "split"])
log("rows", len(df))


def germ_for(vcall):
    """Resolve the assigned V allele to a germline seq; fall back to *01 of the gene."""
    if not isinstance(vcall, str) or not vcall:
        return None
    g = vcall.split(",")[0].strip()
    if g in AG:
        return AG[g]
    base = g.split("*")[0]
    for allele in (base + "*01", base + "*02"):
        if allele in AG:
            return AG[allele]
    # any allele of this gene
    for k in AG:
        if k.split("*")[0] == base:
            return AG[k]
    return None


# ANARCI number all heavy chains in one batch call
seqs = [(str(i), s) for i, s in zip(df["id"], df["heavy_aa"]) if isinstance(s, str) and len(s) > 50]
log("numbering", len(seqs), "heavy chains with ANARCI")
results = run_anarci(seqs, scheme="imgt", ncpu=4)
numbering = results[
    1
]  # list aligned to input order; each is [(domain_numbering, start, end)] or None

# IMGT position -> index in the gapped-128 germline: germlines are indexed by IMGT position 1..128
# ANARCI numbering gives [((pos,ins),aa),...]; germline gapped string is positional 1..128 (index 0..127)
id2num = {}
for (nm, s), res in zip(seqs, numbering):
    id2num[nm] = res

rows = []
miss_germ = 0
miss_num = 0
for _, r in df.iterrows():
    sid = str(r["id"])
    germ = germ_for(r["v_call_heavy"])
    num = id2num.get(sid)
    if germ is None:
        miss_germ += 1
    if num is None or germ is None:
        if num is None:
            miss_num += 1
        rows.append((r["id"], np.nan, -1, 0))
        continue
    dom = num[0][0]  # first domain numbering: list of ((pos,ins),aa)
    nm = 0
    nmis = 0
    for (pos, ins), aa in dom:
        if aa == "-":
            continue
        if ins != " ":
            continue  # skip insertions (not in germline frame)
        if pos < 1 or pos > 128:
            continue
        if pos >= 105:
            continue  # exclude CDR3 region (IMGT CDR3 starts ~105) + FR4
        g_aa = germ[pos - 1]
        if g_aa == "-":
            continue  # germline gap
        nm += 1
        if aa != g_aa:
            nmis += 1
    ident = 100.0 * (1 - nmis / nm) if nm > 0 else np.nan
    rows.append((r["id"], ident, nmis, nm))

out = pd.DataFrame(rows, columns=["id", "v_identity", "n_mismatch", "n_aligned"])
out = out.merge(df[["id", "split"]], on="id")


# SHM bins by V-gene AA identity to germline (standard cutoffs)
def binf(x):
    if not np.isfinite(x):
        return "unknown"
    if x >= 98:
        return "near-germline (>=98%)"
    if x >= 95:
        return "low-SHM (95-98%)"
    if x >= 90:
        return "mid-SHM (90-95%)"
    return "high-SHM (<90%)"


out["shm_bin"] = out["v_identity"].map(binf)
out.to_parquet(f"{BASE}/results/shm_per_sequence.parquet")
log("missing germline", miss_germ, "| missing numbering", miss_num)
log("v_identity describe:\n" + out["v_identity"].describe().to_string())
log("bin counts:\n" + out["shm_bin"].value_counts().to_string())
log("test-split bin counts:\n" + out[out.split == 2]["shm_bin"].value_counts().to_string())
summ = dict(
    n=len(out),
    n_valid=int(out["v_identity"].notna().sum()),
    median_identity=float(out["v_identity"].median()),
    bins={k: int(v) for k, v in out["shm_bin"].value_counts().items()},
    test_bins={k: int(v) for k, v in out[out.split == 2]["shm_bin"].value_counts().items()},
)
json.dump(
    summ,
    open(f"{BASE}/results/shm_summary.json", "w"),
    indent=2,
)
log("DONE")
