#!/usr/bin/env python
"""Build the full annotated OAS-paired dataset (REUSABLE scale-up prep).

Parses every oas_paired/*.csv.gz (OAS AIRR paired format: line 1 = JSON metadata
header, line 2 = CSV header, rows follow), exact-dedups on (heavy_aa, light_aa),
QC-filters (standard AA only, per-chain length bounds), builds per-residue region
labels for both chains, v/d/j calls in full/stripped/family forms, and metadata
labels. Clusters heavy CDR3 with mmseqs2 easy-cluster (--min-seq-id 0.8 -c 0.8)
and splits 70/15/15 by cluster so no cluster spans splits (leakage control).

Reproduces the 47-column schema of oas_full_annotated.parquet exactly so all
downstream analyses (concept/SHM/isotype/steering) run unchanged.

Usage: python build_oas_scaleup.py <out_parquet> <out_summary_json> [max_per_file]
"""

import gzip, json, os, glob, sys, subprocess, hashlib, time
import pandas as pd, numpy as np
import pyarrow as pa, pyarrow.parquet as pq

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import OAS_RAW, SCRATCH

SRC = OAS_RAW
OUT = sys.argv[1]
SUMMARY = sys.argv[2]
MAX_PER_FILE = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # 0 = no cap (entire data)
WORK = f"{SCRATCH}/oas_scaleup_work"
os.makedirs(WORK, exist_ok=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MINLEN, MAXLEN = 20, 200  # per-chain aa length bounds

SEG = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"]
LAB = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def strip_allele(c):
    return c.split("*")[0] if isinstance(c, str) and c else c


def family(c):
    if not isinstance(c, str) or not c:
        return c
    return c.split("*")[0].split("-")[0]


def build_chain(row, chain):
    """Return (aa, region_labels, seg_aa dict) or None if invalid."""
    seg_aa = {}
    parts, labels = [], []
    for s, lab in zip(SEG, LAB):
        v = row.get(f"{s}_aa_{chain}", "")
        v = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
        seg_aa[s] = v
        parts.append(v)
        labels.extend([lab] * len(v))
    aa = "".join(parts)
    if len(aa) < MINLEN or len(aa) > MAXLEN:
        return None
    if not set(aa) <= STD_AA:
        return None
    return aa, labels, seg_aa


files = sorted(glob.glob(os.path.join(SRC, "*_Paired_All.csv.gz")))
log(f"{len(files)} source files")

rows = []
seen = set()
stats = dict(n_files=len(files), n_raw=0, n_bad_chain=0, n_dupes=0, per_file_kept={})
for fi, f in enumerate(files):
    fname = os.path.basename(f)
    with gzip.open(f, "rt") as fh:
        meta_line = fh.readline()
        try:
            meta = json.loads(meta_line.strip().strip('"').replace('""', '"'))
        except Exception:
            meta = {}
        df = pd.read_csv(fh, low_memory=False)
    kept_this = 0
    for _, r in df.iterrows():
        stats["n_raw"] += 1
        if MAX_PER_FILE and kept_this >= MAX_PER_FILE:
            break
        H = build_chain(r, "heavy")
        L = build_chain(r, "light")
        if H is None or L is None:
            stats["n_bad_chain"] += 1
            continue
        h_aa, h_lab, h_seg = H
        l_aa, l_lab, l_seg = L
        key = (h_aa, l_aa)
        if key in seen:
            stats["n_dupes"] += 1
            continue
        seen.add(key)
        rec = dict(
            id=f"{fname}:{r.get('sequence_id_heavy', kept_this)}",
            file=fname,
            heavy_aa=h_aa,
            light_aa=l_aa,
            region_labels_heavy=np.array(h_lab),
            region_labels_light=np.array(l_lab),
            cdr3_aa_heavy=h_seg["cdr3"],
            cdr3_aa_light=l_seg["cdr3"],
            cdr1_aa_heavy=h_seg["cdr1"],
            cdr2_aa_heavy=h_seg["cdr2"],
            fwr1_aa_heavy=h_seg["fwr1"],
            fwr2_aa_heavy=h_seg["fwr2"],
            fwr3_aa_heavy=h_seg["fwr3"],
            fwr4_aa_heavy=h_seg["fwr4"],
            cdr1_aa_light=l_seg["cdr1"],
            cdr2_aa_light=l_seg["cdr2"],
            fwr1_aa_light=l_seg["fwr1"],
            fwr2_aa_light=l_seg["fwr2"],
            fwr3_aa_light=l_seg["fwr3"],
            fwr4_aa_light=l_seg["fwr4"],
            v_call_heavy=strip_allele(r.get("v_call_heavy")),
            d_call_heavy=strip_allele(r.get("d_call_heavy")),
            j_call_heavy=strip_allele(r.get("j_call_heavy")),
            v_call_light=strip_allele(r.get("v_call_light")),
            j_call_light=strip_allele(r.get("j_call_light")),
            v_call_heavy_full=r.get("v_call_heavy"),
            d_call_heavy_full=r.get("d_call_heavy"),
            j_call_heavy_full=r.get("j_call_heavy"),
            v_call_light_full=r.get("v_call_light"),
            j_call_light_full=r.get("j_call_light"),
            v_family_heavy=family(r.get("v_call_heavy")),
            d_family_heavy=family(r.get("d_call_heavy")),
            j_family_heavy=family(r.get("j_call_heavy")),
            v_family_light=family(r.get("v_call_light")),
            j_family_light=family(r.get("j_call_light")),
            locus_heavy=r.get("locus_heavy"),
            locus_light=r.get("locus_light"),
            Isotype=r.get("Isotype_heavy") or meta.get("Isotype"),
            Isotype_light=r.get("locus_light"),
            Species=meta.get("Species"),
            BType=meta.get("BType"),
            Vaccine=meta.get("Vaccine"),
            Disease=meta.get("Disease"),
            Subject=meta.get("Subject"),
            Author=meta.get("Author"),
            BSource=meta.get("BSource"),
        )
        rows.append(rec)
        kept_this += 1
    stats["per_file_kept"][fname] = kept_this
    if fi % 50 == 0:
        log(f"file {fi}/{len(files)} kept_total={len(rows)}")

log(
    f"parsed: raw={stats['n_raw']} kept={len(rows)} dupes={stats['n_dupes']} bad={stats['n_bad_chain']}"
)
df = pd.DataFrame(rows)

# --- cluster heavy CDR3 with mmseqs2 (leakage control) ---
fasta = os.path.join(WORK, "cdr3.fasta")
with open(fasta, "w") as fh:
    for i, s in enumerate(df["cdr3_aa_heavy"].tolist()):
        fh.write(f">{i}\n{s}\n")
pref = os.path.join(WORK, "clu")
subprocess.run(
    [
        "mmseqs",
        "easy-cluster",
        fasta,
        pref,
        os.path.join(WORK, "tmp"),
        "--min-seq-id",
        "0.8",
        "-c",
        "0.8",
        "--cov-mode",
        "1",
    ],
    check=True,
)
clu = pd.read_csv(
    pref + "_cluster.tsv",
    sep="\t",
    header=None,
    names=["rep", "mem"],
    dtype={"rep": str, "mem": str},
)  # fasta headers are row indices as strings
rep_of = dict(zip(clu["mem"], clu["rep"]))
df["cluster_id"] = [rep_of.get(str(i), str(i)) for i in range(len(df))]
n_uniq = df["cluster_id"].nunique()
assert n_uniq < len(df), f"clustering collapsed to singletons: {n_uniq} == {len(df)}"
log(f"cluster_id assigned: {n_uniq} clusters over {len(df)} seqs")

# --- split 70/15/15 by cluster ---
rng = np.random.RandomState(42)
clusters = np.array(list(df["cluster_id"].unique()), dtype=object)
rng.shuffle(clusters)
n = len(clusters)
n_tr = int(0.70 * n)
n_va = int(0.15 * n)
split_map = {}
for j, cid in enumerate(clusters):
    split_map[cid] = "train" if j < n_tr else ("val" if j < n_tr + n_va else "test")
df["split"] = df["cluster_id"].map(split_map)

df = df.sort_values("id").reset_index(drop=True)

# --- FAST parquet write: build list<string> columns natively via pyarrow ---
# pandas.to_parquet on object-dtype ragged arrays is pathologically slow (hours);
# pop the two ragged columns, write scalars fast, append list arrays at C level.
log("building region-label list arrays")
rl_h = [list(x) for x in df.pop("region_labels_heavy")]
rl_l = [list(x) for x in df.pop("region_labels_light")]
log("converting scalar frame to arrow")
table = pa.Table.from_pandas(df, preserve_index=False)
table = table.append_column("region_labels_heavy", pa.array(rl_h, type=pa.list_(pa.string())))
table = table.append_column("region_labels_light", pa.array(rl_l, type=pa.list_(pa.string())))
log("writing parquet")
pq.write_table(table, OUT, compression="zstd")
log("parquet written")

# reconcile per-file counts to post-dedup total
assert sum(stats["per_file_kept"].values()) == len(
    df
), f"{sum(stats['per_file_kept'].values())} != {len(df)}"
stats.update(
    n_sequences=len(df),
    dedup_removed=stats["n_dupes"],
    n_clusters=int(n),
    split_counts=df["split"].value_counts().to_dict(),
    n_heavy_res=int(df["heavy_aa"].str.len().sum()),
    n_light_res=int(df["light_aa"].str.len().sum()),
)
with open(SUMMARY, "w") as fh:
    json.dump(
        {k: v for k, v in stats.items() if k != "per_file_kept"}
        | {"per_file_kept_head": dict(list(stats["per_file_kept"].items())[:5])},
        fh,
        indent=2,
    )
log("DONE", stats["n_sequences"], "seqs", stats["split_counts"])
