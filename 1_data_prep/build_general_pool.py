#!/usr/bin/env python
"""Build the Swiss-Prot general-protein 95%-non-redundant scale-up pool.

Replaces the old 50%-identity clustering (step2_build.py -> 96,477 clusters)
with whole-sequence 95%-identity linclust, matching ProtBERT's near-full-diversity
(UniRef100-like) training regime. Keeps ONE representative per 95% cluster.

Reuses the filtering already done by step1_filter.py (Ig/TCR removal + length 20-512
+ standard-AA-only + exact dedup); its output filtered.fasta is the unique deduped set.

Usage:
  build_general_pool.py FILTERED_FASTA WORKDIR OUT_PARQUET OUT_SUMMARY STEP1_STATS MIN_ID COV
"""

import json, os, sys, subprocess, collections
import numpy as np
import pandas as pd

FILTERED_FASTA = sys.argv[1]  # unique deduped set from step1_filter.py
WORKDIR = sys.argv[2]  # scratch work dir for mmseqs (purgeable)
OUT_PARQUET = sys.argv[3]  # D/data/general_pool_95nr.parquet
OUT_SUMMARY = sys.argv[4]  # D/data/general_pool_95nr_summary.json
STEP1_STATS = sys.argv[5]  # step1_stats_96k.json (funnel counts)
MIN_ID = float(sys.argv[6])  # 0.95
COV = float(sys.argv[7])  # 0.8

OLD_50PCT_CLUSTERS = 96477  # old step2_build.py at min-seq-id=0.5

os.makedirs(WORKDIR, exist_ok=True)


def iter_fasta(path):
    hdr, seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(seq)
                hdr, seq = line[1:], []
            else:
                seq.append(line)
    if hdr is not None:
        yield hdr, "".join(seq)


def run_linclust(in_fasta, pref, tmp):
    cmd = [
        "mmseqs",
        "easy-linclust",
        in_fasta,
        pref,
        tmp,
        "--min-seq-id",
        str(MIN_ID),
        "-c",
        str(COV),
        "--cov-mode",
        "0",
        "--threads",
        "16",
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---- 1. read filtered (already deduped) sequences, assign STABLE INDEX id ----
recs = list(iter_fasta(FILTERED_FASTA))
recs.sort(key=lambda x: x[0])  # sort by original accession for determinism
n_input = len(recs)
seqs = [s for _, s in recs]  # index i -> sequence
idx_fasta = os.path.join(WORKDIR, "input_indexed.fasta")
with open(idx_fasta, "w") as out:
    for i, s in enumerate(seqs):
        out.write(f">{i}\n{s}\n")
print(f"Wrote {n_input} indexed sequences to {idx_fasta}", flush=True)

# ---- 2. cluster whole sequence at 95% identity ----
pref = os.path.join(WORKDIR, "clu95")
tmp = os.path.join(WORKDIR, "tmp95")
run_linclust(idx_fasta, pref, tmp)

# ---- 3. parse clusters: rep_idx -> [member_idx,...] ----
cluster_members = collections.defaultdict(list)
with open(pref + "_cluster.tsv") as fh:
    for line in fh:
        rep, mem = line.rstrip("\n").split("\t")
        cluster_members[int(rep)].append(int(mem))
reps = sorted(cluster_members.keys())
n_clusters = len(reps)
print(f"n_clusters (pool size) = {n_clusters}", flush=True)

# sanity: total members == n_input
total_members = sum(len(v) for v in cluster_members.values())
assert total_members == n_input, (total_members, n_input)

# ---- 4. build parquet: one representative per cluster ----
rows = []
for cid, rep in enumerate(reps):
    s = seqs[rep]
    rows.append((rep, s, len(s), "swissprot", "general_protein", cid))
df = pd.DataFrame(rows, columns=["id", "seq", "length", "source", "kind", "cluster_id"])
df = df.sort_values("id").reset_index(drop=True)
df.to_parquet(OUT_PARQUET, index=False)
print(f"Wrote parquet {OUT_PARQUET} rows={len(df)}", flush=True)

# ---- 5. cluster-size distribution ----
sizes = np.array([len(cluster_members[r]) for r in reps])
n_singletons = int((sizes == 1).sum())
singleton_fraction = float(n_singletons / n_clusters)
size_bins = [1, 2, 3, 5, 10, 20, 50, 100, 10**9]
size_labels = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", "100+"]
hist = {}
for i in range(len(size_labels)):
    lo, hi = size_bins[i], size_bins[i + 1]
    hist[size_labels[i]] = int(((sizes >= lo) & (sizes < hi)).sum())
csd = dict(
    n_clusters=n_clusters,
    n_singletons=n_singletons,
    singleton_fraction=singleton_fraction,
    min_size=int(sizes.min()),
    max_size=int(sizes.max()),
    mean_size=float(sizes.mean()),
    median_size=float(np.median(sizes)),
    histogram=hist,
)

# ---- 5b. VERIFY 95%-non-redundancy: re-cluster the representatives ----
rep_fasta = os.path.join(WORKDIR, "reps.fasta")
with open(rep_fasta, "w") as out:
    for rep in reps:
        out.write(f">{rep}\n{seqs[rep]}\n")
vpref = os.path.join(WORKDIR, "verify95")
vtmp = os.path.join(WORKDIR, "verifytmp95")
run_linclust(rep_fasta, vpref, vtmp)
vreps = set()
with open(vpref + "_cluster.tsv") as fh:
    for line in fh:
        rep, mem = line.rstrip("\n").split("\t")
        vreps.add(int(rep))
n_verify_clusters = len(vreps)
idempotent = n_verify_clusters == n_clusters
print(
    f"VERIFY: re-cluster reps -> {n_verify_clusters} clusters (expect {n_clusters}); idempotent={idempotent}",
    flush=True,
)

# ---- 6. summary JSON ----
step1 = json.load(open(STEP1_STATS))
n_after_length = step1["n_after_ig"] - step1["n_removed_length"]
L = df["length"].to_numpy()
lstats = dict(
    min=int(L.min()),
    max=int(L.max()),
    mean=float(L.mean()),
    median=float(np.median(L)),
    p05=float(np.percentile(L, 5)),
    p25=float(np.percentile(L, 25)),
    p75=float(np.percentile(L, 75)),
    p95=float(np.percentile(L, 95)),
)
summary = dict(
    build="swissprot general-protein 95%-non-redundant scale-up pool",
    source="UniProtKB/Swiss-Prot (reviewed), release 2026_02",
    funnel=dict(
        n_downloaded=step1["n_downloaded"],
        n_removed_ig=step1["n_removed_ig"],
        ig_by_name=step1["ig_by_name"],
        ig_by_gene=step1["ig_by_gene"],
        n_after_ig=step1["n_after_ig"],
        n_removed_length=step1["n_removed_length"],
        n_after_length=n_after_length,
        n_removed_nonstd=step1["n_removed_nonstd"],
        n_after_quality=step1["n_after_quality"],
        n_dupes_removed=step1["n_dupes_removed"],
        n_after_dedup=step1["n_after_dedup"],
    ),
    clustering=dict(
        tool="mmseqs2 easy-linclust",
        min_seq_id=MIN_ID,
        coverage=COV,
        cov_mode=0,
        threads=16,
        whole_sequence=True,
    ),
    n_clusters=n_clusters,
    pool_size=n_clusters,
    cluster_size_distribution=csd,
    verification=dict(
        method="re-cluster representatives at min-seq-id=0.95 -c 0.8; idempotent means no two reps are >=95% identity over 80% coverage",
        n_verify_clusters=n_verify_clusters,
        idempotent=idempotent,
        non_redundant_95=idempotent,
    ),
    length_stats=lstats,
    comparison_old_build=dict(
        old_min_seq_id=0.5,
        old_n_clusters=OLD_50PCT_CLUSTERS,
        new_min_seq_id=MIN_ID,
        new_n_clusters=n_clusters,
        ratio_new_over_old=float(n_clusters / OLD_50PCT_CLUSTERS),
        note=(
            "95%% linclust keeps far more clusters than the old 50%% build "
            "(96,477), approaching the deduped count (383,352), matching "
            "ProtBERT's near-full-diversity UniRef100 regime."
        ),
    ),
    columns=list(df.columns),
    parquet=OUT_PARQUET,
)
json.dump(summary, open(OUT_SUMMARY, "w"), indent=2)
print(
    "SUMMARY:",
    json.dumps(
        {k: summary[k] for k in ["n_clusters", "cluster_size_distribution", "verification"]},
        indent=2,
    ),
    flush=True,
)
print("DONE", flush=True)
