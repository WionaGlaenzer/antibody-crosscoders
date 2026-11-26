#!/usr/bin/env python3
"""
build_oas_pool.py — Build the OAS antibody scale-up pool (95% non-redundant,
whole-sequence representatives).

Pipeline:
  1. Read D/data/oas_scaleup_annotated.parquet (2,479,273 paired antibody seqs).
  2. Write a FASTA of the CONCATENATED heavy+light AA sequence per row.
     FASTA id = parquet row index (0-based).
  3. mmseqs easy-linclust --min-seq-id 0.95 -c 0.8 --cov-mode 0 --threads 16
     (linear-time clustering at 95% identity / 80% coverage).
  4. Keep ONE representative per cluster (ids in <pref>_rep_seq.fasta).
  5. Subset the original parquet to those representative rows -> oas_pool_95nr.parquet
     (ALL original columns preserved) + a `cluster_id` provenance column (the
     mmseqs representative id each row maps to).
  6. Write a summary JSON (n_input, n_clusters, cluster-size distribution, etc.).

The resulting pool is 95%-non-redundant by construction: one representative per
95%-identity / 80%-coverage cluster, so no two members share >=95% whole-sequence
identity over >=80% coverage. Any downstream split is therefore leakage-controlled.
"""

import os
import sys
import json
import subprocess
from collections import Counter

import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

# --- Paths (absolute, project-dir; scratch is purged) ---
D = BASE
SRC_PARQUET = os.path.join(D, "data", "oas_scaleup_annotated.parquet")
OUT_PARQUET = os.path.join(D, "data", "oas_pool_95nr.parquet")
OUT_SUMMARY = os.path.join(D, "data", "oas_pool_95nr_summary.json")

# Work in a job-local scratch tmp for mmseqs I/O (fast, purgeable is fine here)
WORKDIR = os.environ.get("OAS_WORKDIR", os.getcwd())
FASTA = os.path.join(WORKDIR, "oas_concat.fasta")
PREF = os.path.join(WORKDIR, "oas_clust")
TMP = os.path.join(WORKDIR, "mmseqs_tmp")

HEAVY_COL = "heavy_aa"
LIGHT_COL = "light_aa"


def log(msg):
    print(msg, flush=True)


def main():
    log(f"[1] Reading {SRC_PARQUET}")
    df = pd.read_parquet(SRC_PARQUET)
    n_input = len(df)
    log(f"    n_input = {n_input}, columns = {list(df.columns)}")
    assert HEAVY_COL in df.columns, f"missing {HEAVY_COL}"
    assert LIGHT_COL in df.columns, f"missing {LIGHT_COL}"

    # --- [2] Write concatenated heavy+light FASTA, id = row index ---
    log(f"[2] Writing concatenated heavy+light FASTA -> {FASTA}")
    heavy = df[HEAVY_COL].astype(str).values
    light = df[LIGHT_COL].astype(str).values
    n_written = 0
    n_empty = 0
    with open(FASTA, "w") as fh:
        for i in range(n_input):
            h = heavy[i].strip()
            l = light[i].strip()
            seq = (h + l).replace(" ", "")
            if not seq or seq.lower() == "nannan" or len(seq) == 0:
                n_empty += 1
                continue
            fh.write(f">{i}\n{seq}\n")
            n_written += 1
    log(f"    wrote {n_written} sequences ({n_empty} empty/invalid skipped)")

    # --- [3] mmseqs easy-linclust ---
    log("[3] Running mmseqs easy-linclust (--min-seq-id 0.95 -c 0.8 --cov-mode 0)")
    cmd = [
        "mmseqs",
        "easy-linclust",
        FASTA,
        PREF,
        TMP,
        "--min-seq-id",
        "0.95",
        "-c",
        "0.8",
        "--cov-mode",
        "0",
        "--threads",
        "16",
    ]
    log("    " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    # --- [4] Parse cluster assignments ---
    # <pref>_cluster.tsv : two cols per line = <rep_id>\t<member_id>
    cluster_tsv = PREF + "_cluster.tsv"
    rep_fasta = PREF + "_rep_seq.fasta"
    log(f"[4] Parsing {cluster_tsv}")

    member_to_rep = {}
    with open(cluster_tsv) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            rep, member = parts
            member_to_rep[int(member)] = int(rep)

    # Representative ids = those present in rep_seq.fasta
    reps = []
    with open(rep_fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                reps.append(int(line[1:].strip().split()[0]))
    reps_set = set(reps)
    n_clusters = len(reps_set)
    log(f"    n_clusters (pool size) = {n_clusters}")

    # sanity: every member maps to a rep
    all_members = set(member_to_rep.keys())
    log(
        f"    members mapped = {len(all_members)}; unique reps in tsv = {len(set(member_to_rep.values()))}"
    )
    assert (
        set(member_to_rep.values()) == reps_set
    ), "rep set from rep_seq.fasta != rep set from cluster.tsv"

    # --- [5] Cluster-size distribution ---
    cluster_sizes = Counter(member_to_rep.values())  # rep -> size
    size_hist = Counter(cluster_sizes.values())  # size -> #clusters of that size
    max_cluster_size = max(cluster_sizes.values())
    n_singletons = size_hist.get(1, 0)
    singleton_fraction = n_singletons / n_clusters

    # Bucketed distribution: sizes 1..9 explicit, >=10 bucketed
    dist = {}
    for s in range(1, 10):
        dist[str(s)] = size_hist.get(s, 0)
    dist[">=10"] = sum(c for sz, c in size_hist.items() if sz >= 10)

    log(
        f"    singletons = {n_singletons} ({singleton_fraction:.4f}); "
        f"max_cluster_size = {max_cluster_size}"
    )

    # --- [6] Subset original parquet to representatives, add cluster_id ---
    log(f"[6] Subsetting original parquet to {n_clusters} representative rows")
    # We keep ONLY the rep rows. Each rep row IS its own cluster's representative,
    # so its provenance cluster_id == its own (source) row index. This OVERWRITES
    # the stale cluster_id carried from the old CDR3-only build.
    reps_sorted = sorted(reps_set)
    pool = df.iloc[reps_sorted].copy()

    # The source parquet has a duplicated column name ('element'); pyarrow cannot
    # write duplicate column labels. De-duplicate defensively (record the rename).
    renamed_cols = []
    cols = list(pool.columns)
    seen = {}
    new_cols = []
    for cname in cols:
        if cname in seen:
            seen[cname] += 1
            newname = f"{cname}_{seen[cname]}"
            renamed_cols.append((cname, newname))
            new_cols.append(newname)
        else:
            seen[cname] = 0
            new_cols.append(cname)
    if renamed_cols:
        pool.columns = new_cols
        log(f"    de-duplicated column names: {renamed_cols}")

    pool["cluster_id"] = reps_sorted  # rep row's provenance cluster id is itself
    pool.to_parquet(OUT_PARQUET, index=True)
    log(f"    wrote {OUT_PARQUET} ({len(pool)} rows, {len(pool.columns)} cols)")

    # --- Summary JSON ---
    summary = {
        "n_input": n_input,
        "n_written_to_fasta": n_written,
        "n_empty_skipped": n_empty,
        "n_clusters": n_clusters,
        "pool_size": n_clusters,
        "n_singletons": n_singletons,
        "singleton_fraction": singleton_fraction,
        "max_cluster_size": max_cluster_size,
        "cluster_size_distribution": dist,
        "clustering_params": {
            "tool": "mmseqs easy-linclust",
            "min_seq_id": 0.95,
            "coverage": 0.8,
            "cov_mode": 0,
            "sequence": "concatenated heavy_aa + light_aa",
        },
        "source_parquet": SRC_PARQUET,
        "output_parquet": OUT_PARQUET,
        "renamed_duplicate_columns": renamed_cols,
        "note": (
            "cluster_id overwritten with new mmseqs whole-seq rep id; "
            "stale 'split' column from old CDR3 build preserved as-is "
            "(Phase 1 reassigns splits)."
        ),
    }
    with open(OUT_SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"[done] summary -> {OUT_SUMMARY}")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
