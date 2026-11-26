#!/usr/bin/env python3
"""Timing probe: cluster a 200k subsample of the OAS-paired corpus on the CONCATENATED
heavy+light sequence with mmseqs easy-linclust --min-seq-id 0.95 -c 0.8 --cov-mode 0.
Reports wall time + cluster count so we can extrapolate the full 2.48M run. Reuses the
already-built oas_full_annotated + scaleup parquet? No -- we need the full 2.48M source.
We read the existing oas_scaleup_annotated.parquet (2.48M, already parsed/QC'd/deduped)
and take its heavy+light concat, subsample 200k, cluster, time it."""

import time, os, subprocess, numpy as np, pyarrow.parquet as pq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE, SCRATCH

WORK = f"{SCRATCH}/oas_linclust_probe"
os.makedirs(WORK, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


SRC = BASE + "/data/oas_scaleup_annotated.parquet"
t = pq.read_table(SRC, columns=["heavy_aa", "light_aa"]).to_pandas()
log(f"loaded {len(t)} seqs from scaleup parquet")
rng = np.random.default_rng(0)
idx = rng.choice(len(t), 200000, replace=False)
sub = t.iloc[idx].reset_index(drop=True)
fasta = os.path.join(WORK, "concat200k.fasta")
with open(fasta, "w") as f:
    for i, r in sub.iterrows():
        f.write(f">{i}\n{r['heavy_aa']}{r['light_aa']}\n")
log(f"wrote FASTA ({os.path.getsize(fasta)/1e6:.1f} MB)")
pref = os.path.join(WORK, "clu")
tmp = os.path.join(WORK, "tmp")
t0 = time.time()
subprocess.run(
    [
        "mmseqs",
        "easy-linclust",
        fasta,
        pref,
        tmp,
        "--min-seq-id",
        "0.95",
        "-c",
        "0.8",
        "--cov-mode",
        "0",
        "--threads",
        str(os.cpu_count() or 8),
    ],
    check=True,
)
dt = time.time() - t0
# count clusters from rep_seq.fasta
nrep = sum(1 for l in open(pref + "_rep_seq.fasta") if l.startswith(">"))
log(f"PROBE DONE: 200k seqs clustered in {dt:.1f}s -> {nrep} clusters")
# linear extrapolation to full corpus
full = len(t)
log(
    f"linclust is ~linear: extrapolated {full} seqs ~= {dt*full/200000:.0f}s ({dt*full/200000/60:.1f} min)"
)
print(
    f"RESULT probe_seconds={dt:.1f} probe_clusters={nrep} full_seqs={full} extrap_min={dt*full/200000/60:.1f}"
)
