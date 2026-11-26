#!/usr/bin/env python
"""Step 2: take mmseqs cluster representatives, random-sample to target,
build parquet + manifest.json + README.md.
"""

import json, sys, random
import numpy as np
import pandas as pd

REP_FASTA = sys.argv[1]  # mmseqs rep_seq.fasta (one seq per cluster)
STEP1_STATS = sys.argv[2]  # step1_stats.json
OUT_PARQUET = sys.argv[3]
OUT_MANIFEST = sys.argv[4]
OUT_README = sys.argv[5]
SOURCE_URL = sys.argv[6]
RELEASE = sys.argv[7]
TARGET = int(sys.argv[8])
MIN_ID = float(sys.argv[9])  # mmseqs min-seq-id used
COV = float(sys.argv[10])  # mmseqs coverage used

random.seed(42)


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


reps = [(h.split()[0], s) for h, s in iter_fasta(REP_FASTA)]
n_clusters = len(reps)

# random-sample down to target for a manageable, diverse set
if n_clusters > TARGET:
    reps = random.sample(reps, TARGET)
n_final = len(reps)

df = pd.DataFrame(
    {
        "id": [a for a, _ in reps],
        "seq": [s for _, s in reps],
    }
)
df["length"] = df["seq"].str.len()
df["source"] = "swissprot"
df["kind"] = "general_protein"
df = df.sort_values("id").reset_index(drop=True)
df.to_parquet(OUT_PARQUET, index=False)

L = df["length"].to_numpy()
step1 = json.load(open(STEP1_STATS))
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
# histogram bins for the manifest
bins = [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 512]
hist, edges = np.histogram(L, bins=bins)
lhist = {f"{int(edges[i])}-{int(edges[i+1])}": int(hist[i]) for i in range(len(hist))}

manifest = dict(
    source="UniProtKB/Swiss-Prot (reviewed)",
    source_url=SOURCE_URL,
    release=RELEASE,
    n_downloaded=step1["n_downloaded"],
    ig_filter_rule=(
        "drop if protein name matches immunoglobulin/antibody/"
        "Ig[GAMDE]/Ig heavy|kappa|lambda|.../T-cell receptor|TCR/"
        "Fc receptor/V(D)J, OR gene symbol starts IGH|IGK|IGL or "
        "matches TR[ABGD][VDJC] (TCR loci)"
    ),
    n_removed_ig=step1["n_removed_ig"],
    ig_removed_by_name=step1["ig_by_name"],
    ig_removed_by_gene=step1["ig_by_gene"],
    n_after_ig=step1["n_after_ig"],
    length_filter="20 <= len <= 512 residues",
    n_removed_length=step1["n_removed_length"],
    quality_filter="drop any sequence containing non-standard residues (X/B/Z/U/O/* etc.); keep only the 20 standard amino acids",
    n_removed_nonstd=step1["n_removed_nonstd"],
    n_after_quality=step1["n_after_quality"],
    n_dupes_removed=step1["n_dupes_removed"],
    n_after_dedup=step1["n_after_dedup"],
    clustering=dict(
        tool="mmseqs2 easy-cluster",
        min_seq_id=MIN_ID,
        coverage=COV,
        cov_mode=0,
        n_clusters=n_clusters,
    ),
    sampling=dict(
        strategy=(
            "random sample of cluster representatives"
            if n_clusters > TARGET
            else "all cluster representatives"
        ),
        target=TARGET,
        seed=42,
    ),
    n_final=n_final,
    length_stats=lstats,
    length_histogram=lhist,
    columns=list(df.columns),
    parquet=OUT_PARQUET.split("/")[-1],
)
json.dump(manifest, open(OUT_MANIFEST, "w"), indent=2)

readme = f"""# General-protein sequence sample (mixed-data crosscoder companion)

Non-antibody general-protein set for the "mixed-data companion crosscoder" arm.
Mixed later with the antibody set (which gets `kind='antibody'`) so a crosscoder
can reveal which ProtBert general-protein features are DEAD on antibodies
(fire on general proteins, ~0 on antibodies).

## Source
- **{manifest['source']}**
- URL: {SOURCE_URL}
- Release: **{RELEASE}**

## Pipeline
1. Downloaded Swiss-Prot FASTA: **{step1['n_downloaded']:,}** entries.
2. **Ig / antibody / TCR removal** — {manifest['ig_filter_rule']}.
   Removed **{step1['n_removed_ig']:,}** ({step1['ig_by_name']:,} by name, {step1['ig_by_gene']:,} by gene symbol). Remaining: {step1['n_after_ig']:,}.
3. **Length filter** 20-512 aa: removed {step1['n_removed_length']:,}.
4. **Quality filter** (standard 20 aa only): removed {step1['n_removed_nonstd']:,} with non-standard residues.
5. **Exact dedup**: removed {step1['n_dupes_removed']:,}. Unique: {step1['n_after_dedup']:,}.
6. **mmseqs2 clustering** at min-seq-id={MIN_ID}, coverage={COV} (cov-mode 1): {n_clusters:,} clusters.
7. **Sampling**: {manifest['sampling']['strategy']} -> **{n_final:,}** final sequences (target {TARGET:,}, seed 42).

## Final set
- **N = {n_final:,}** sequences
- Length: min {lstats['min']}, median {lstats['median']:.0f}, mean {lstats['mean']:.1f}, max {lstats['max']} (p5={lstats['p05']:.0f}, p95={lstats['p95']:.0f})
- Columns: `{', '.join(df.columns)}`
- File: `{manifest['parquet']}`
"""
open(OUT_README, "w").write(readme)
print("STEP2 done:", dict(n_clusters=n_clusters, n_final=n_final, **lstats))
