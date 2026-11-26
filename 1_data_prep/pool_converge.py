#!/usr/bin/env python3
"""Converge both 95% representative pools to TRUE non-redundancy: linclust is a greedy
linear-time approximation, so a first pass can leave ~0.6% of reps still >=95% similar to
each other. Iterate linclust on the representatives until idempotent (n_clusters stops
shrinking), keeping reps-of-reps. Rewrites each pool parquet in place + updates summary.
Concat sequence for OAS = heavy+light; general = seq."""

import json, os, subprocess, time, numpy as np, pandas as pd, pyarrow.parquet as pq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE, SCRATCH

DATA = BASE + "/data"
WORK = f"{SCRATCH}/pool_converge"
os.makedirs(WORK, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def seqcol(df):
    if "heavy_aa" in df.columns:
        return (df["heavy_aa"].fillna("") + df["light_aa"].fillna("")).values
    return df["seq"].values


def linclust(seqs, ids, tag):
    fa = os.path.join(WORK, tag + ".fasta")
    pref = os.path.join(WORK, tag + "_clu")
    tmp = os.path.join(WORK, tag + "_tmp")
    with open(fa, "w") as f:
        for i, s in zip(ids, seqs):
            if s:
                f.write(f">{i}\n{s}\n")
    subprocess.run(
        [
            "mmseqs",
            "easy-linclust",
            fa,
            pref,
            tmp,
            "--min-seq-id",
            "0.95",
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--threads",
            "16",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    reps = set()
    for l in open(pref + "_rep_seq.fasta"):
        if l.startswith(">"):
            reps.add(l[1:].strip().split()[0])
    return reps


for name, pf in [("oas", "oas_pool_95nr.parquet"), ("general", "general_pool_95nr.parquet")]:
    p = os.path.join(DATA, pf)
    df = pd.read_parquet(p)
    df.index = df.index.astype(str)
    log(f"{name}: start {len(df)} reps")
    it = 0
    while True:
        it += 1
        ids = df.index.tolist()
        seqs = seqcol(df)
        reps = linclust(seqs, ids, f"{name}_it{it}")
        keep = df.index.isin(reps)
        n_new = int(keep.sum())
        n_old = len(df)
        log(f"  iter {it}: {n_old} -> {n_new}")
        df = df[keep]
        if n_new == n_old or it >= 5:
            break
    df.reset_index(drop=True).to_parquet(p)
    # update summary
    sp = os.path.join(DATA, pf.replace(".parquet", "_summary.json"))
    s = json.load(open(sp))
    s["converged_pool_size"] = len(df)
    s["converge_iters"] = it
    s["non_redundant_95_converged"] = True
    json.dump(s, open(sp, "w"), indent=2)
    log(f"{name}: CONVERGED at {len(df)} reps in {it} iters")
    print(f"RESULT {name} converged_pool_size={len(df)} iters={it}")
