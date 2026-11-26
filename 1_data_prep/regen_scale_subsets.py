#!/usr/bin/env python3
"""FOLLOW-UP: regenerate ONLY the three SCALE experiment parquets at a new budget.

B_scale = 94,825,040 residues (exactly 10x the original full_paired_L20 store
= 9,482,504). Same uniform sampling over the 95%-NR pools (seed 42, no study
stratification), same nesting invariants, same 70/15/15 per-rep split, same
seq_id namespacing, same schema. The three SMALL parquets are NOT touched.
"""

import os, json, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

D = f"{BASE}/data"
SEED = 42
B_SCALE = 94_825_040  # 10x original full_paired_L20 store (9,482,504)
OFFSET = 10_000_000  # general-id namespace offset (scale) — unchanged
scale = "scale"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def draw_by_budget(nres, budget, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(nres))
    csum = np.cumsum(nres[perm])
    k = int(np.searchsorted(csum, budget, side="left")) + 1
    k = min(k, len(nres))
    return perm[:k]


def assign_split(n, seed):
    rng = np.random.default_rng(seed)
    r = rng.random(n)
    return np.where(r < 0.70, "train", np.where(r < 0.85, "val", "test"))


def split_counts(splits, nres):
    """per-split {n_seq, residues} given aligned split-label & residue arrays."""
    splits = np.asarray(splits)
    nres = np.asarray(nres)
    out = {}
    for k in ("train", "val", "test"):
        m = splits == k
        out[k] = {"n_seq": int(m.sum()), "residues": int(nres[m].sum())}
    return out


# ---------------------------------------------------------------- load pools
log("reading antibody pool (heavy/light for residue counts)")
ab_min = pq.read_table(f"{D}/oas_pool_95nr.parquet", columns=["heavy_aa", "light_aa"]).to_pandas()
ab_nres = (
    ab_min["heavy_aa"].str.len().fillna(0).astype(int)
    + ab_min["light_aa"].str.len().fillna(0).astype(int)
).to_numpy()
log("ab pool reps", len(ab_nres), "total concat residues", int(ab_nres.sum()))

log("reading full antibody pool table")
ab_full = pq.read_table(f"{D}/oas_pool_95nr.parquet")

log("reading general pool")
gn = pq.read_table(
    f"{D}/general_pool_95nr.parquet", columns=["id", "seq", "length", "source"]
).to_pandas()
gn_nres = gn["seq"].str.len().astype(int).to_numpy()
log("gn pool reps", len(gn), "total residues", int(gn_nres.sum()))

B = B_SCALE
half = B // 2
OFF = OFFSET
log(f"==== scale build B={B} half={half} offset={OFF} ====")

# (1) paired
sel_ab = draw_by_budget(ab_nres, B, SEED)
split_ab = assign_split(len(sel_ab), SEED)
dfp = ab_full.take(pa.array(sel_ab)).to_pandas()
dfp["split"] = split_ab
dfp["regime"] = "paired"
p_paired = f"{D}/exp_paired_scale.parquet"
dfp.to_parquet(p_paired, index=False)
paired_res = int(ab_nres[sel_ab].sum())
log(f"paired_scale: n_ab={len(dfp)} residues={paired_res} -> {p_paired}")

# (2) unpaired = identical rows + split, regime marker only
dfu = dfp.copy()
dfu["regime"] = "unpaired"
p_unpaired = f"{D}/exp_unpaired_scale.parquet"
dfu.to_parquet(p_unpaired, index=False)
log(f"unpaired_scale: n_ab={len(dfu)} -> {p_unpaired}")

# (3) mixed: ab half strict subset of paired rows; general 50/50 by residue
dfp_nres = (dfp["heavy_aa"].str.len() + dfp["light_aa"].str.len()).to_numpy()
sel_mix_ab = draw_by_budget(dfp_nres, half, SEED)
mix_ab = dfp.iloc[sel_mix_ab]
ab_res = int(dfp_nres[sel_mix_ab].sum())
sel_gn = draw_by_budget(gn_nres, half, SEED)
mix_gn = gn.iloc[sel_gn].copy()
gn_split = assign_split(len(mix_gn), SEED)
gn_res = int(gn_nres[sel_gn].sum())
n_ab = len(mix_ab)
n_gn = len(mix_gn)

ab_part = pd.DataFrame(
    {
        "seq_id": np.arange(n_ab, dtype=np.int64),
        "orig_id": mix_ab["id"].to_numpy(),
        "kind": np.int8(0),
        "split": mix_ab["split"].to_numpy(),
        "heavy_aa": mix_ab["heavy_aa"].to_numpy(),
        "light_aa": mix_ab["light_aa"].to_numpy(),
        "seq": pd.Series([None] * n_ab, dtype=object),
        "source": "oas",
        "n_res": dfp_nres[sel_mix_ab],
    }
)
gn_part = pd.DataFrame(
    {
        "seq_id": (mix_gn["id"].to_numpy() + OFF),
        "orig_id": mix_gn["id"].astype(str).to_numpy(),
        "kind": np.int8(1),
        "split": gn_split,
        "heavy_aa": pd.Series([None] * n_gn, dtype=object),
        "light_aa": pd.Series([None] * n_gn, dtype=object),
        "seq": mix_gn["seq"].to_numpy(),
        "source": mix_gn["source"].to_numpy(),
        "n_res": gn_nres[sel_gn],
    }
)
mixed = pd.concat([ab_part, gn_part], ignore_index=True)
p_mixed = f"{D}/exp_mixed_scale.parquet"
mixed.to_parquet(p_mixed, index=False)
log(f"mixed_scale: n_ab={n_ab}({ab_res}) n_gn={n_gn}({gn_res}) total={ab_res+gn_res} -> {p_mixed}")

# ---------------------------------------------------------------- verify (read back)
log("==== verification (reading written parquets back) ====")
dp = pd.read_parquet(p_paired, columns=["id", "heavy_aa", "light_aa", "split"])
du = pd.read_parquet(p_unpaired, columns=["id", "heavy_aa", "light_aa", "split"])
dm = pd.read_parquet(p_mixed)

dp_res = (dp.heavy_aa.str.len() + dp.light_aa.str.len()).to_numpy()
du_res = (du.heavy_aa.str.len() + du.light_aa.str.len()).to_numpy()
p_res = int(dp_res.sum())
u_res = int(du_res.sum())
m_res = int(dm.n_res.sum())

paired_ids = set(dp["id"])
unpaired_ids = set(du["id"])
mix_ab_ids = set(dm.loc[dm.kind == 0, "orig_id"])
subset = mix_ab_ids.issubset(paired_ids)
strict = subset and (len(mix_ab_ids) < len(paired_ids))
same_set = paired_ids == unpaired_ids
sp = dp.set_index("id")["split"]
su = du.set_index("id")["split"]
same_split = bool((sp.sort_index() == su.sort_index()).all()) and same_set


def props(s):
    vc = s.value_counts(normalize=True)
    return {k: round(float(vc.get(k, 0.0)), 4) for k in ("train", "val", "test")}


j = dm.loc[dm.kind == 0, ["orig_id", "split"]].merge(
    dp[["id", "split"]], left_on="orig_id", right_on="id", suffixes=("_mix", "_paired")
)
ab_split_consistent = bool((j.split_mix == j.split_paired).all())
trio = [p_res, u_res, m_res]
size_match = (max(trio) - min(trio)) / max(trio) < 0.05

verify_scale = {
    "B": B,
    "offset": OFF,
    "paired_residues": p_res,
    "unpaired_residues": u_res,
    "mixed_residues": m_res,
    "paired_n_ab": len(dp),
    "mixed_n_ab": int((dm.kind == 0).sum()),
    "mixed_n_gen": int((dm.kind == 1).sum()),
    "mixed_ab_residues": int(dm.loc[dm.kind == 0, "n_res"].sum()),
    "mixed_gen_residues": int(dm.loc[dm.kind == 1, "n_res"].sum()),
    "mixed_ab_strict_subset_of_paired": bool(strict),
    "unpaired_equals_paired_abs": bool(same_split),
    "size_match_within_5pct": bool(size_match),
    "ab_split_consistent_paired_vs_mixed": ab_split_consistent,
    "no_seqid_collision": int(dm.seq_id.nunique()) == len(dm),
    "split_props_paired": props(dp["split"]),
    "split_props_mixed_ab": props(dm.loc[dm.kind == 0, "split"]),
    "split_props_mixed_gen": props(dm.loc[dm.kind == 1, "split"]),
    # per-split sequence & residue counts
    "per_split_paired": split_counts(dp["split"].to_numpy(), dp_res),
    "per_split_unpaired": split_counts(du["split"].to_numpy(), du_res),
    "per_split_mixed_all": split_counts(dm["split"].to_numpy(), dm["n_res"].to_numpy()),
    "per_split_mixed_ab": split_counts(
        dm.loc[dm.kind == 0, "split"].to_numpy(), dm.loc[dm.kind == 0, "n_res"].to_numpy()
    ),
    "per_split_mixed_gen": split_counts(
        dm.loc[dm.kind == 1, "split"].to_numpy(), dm.loc[dm.kind == 1, "n_res"].to_numpy()
    ),
}
log("verify_scale:", json.dumps(verify_scale))

# ---------------------------------------------------------------- merge into existing summary
outp = f"{D}/subsets_summary.json"
summary = json.load(open(outp))
summary["budgets"]["scale"] = B
summary["offsets"]["scale"] = OFF
summary["scales"]["scale"] = {
    "B": B,
    "offset": OFF,
    "paired_n_ab": int(len(dp)),
    "paired_residues": p_res,
    "mixed_n_ab": int(n_ab),
    "mixed_ab_residues": ab_res,
    "mixed_n_gen": int(n_gn),
    "mixed_gen_residues": gn_res,
    "mixed_residues": ab_res + gn_res,
    "note": "regenerated at B_scale=10x full_paired_L20 (94,825,040)",
}
summary["verify"]["scale"] = verify_scale
with open(outp, "w") as f:
    json.dump(summary, f, indent=2)
log("updated", outp)
log("DONE")
