#!/usr/bin/env python3
"""Draw NESTED, SIZE-MATCHED experiment subsets for three training regimes
(unpaired / paired / mixed) from the two 95%-non-redundant representative pools.

Design (content-vs-volume control): all regimes are matched to a FIXED TOTAL
residue budget B by uniform random sampling over representatives (seed 42, no
study stratification). The three regimes therefore differ only in content/format,
not residue budget.

  paired-small   : uniform ab reps until cum(concat H+L residues) >= B  (~40k ab)
  unpaired-small : the SAME ab reps + SAME split (regime marker only; the
                   heavy/light-as-separate-chains difference is applied at
                   extraction time). H+L residues == concat length, so identical B.
  mixed-small    : ab + general split 50/50 BY RESIDUE to B. The antibody half is
                   a STRICT SUBSET of paired-small's rows (drawn from paired rows,
                   not fresh from the pool); each ab seq keeps its paired split;
                   general gets a fresh 70/15/15 split. kind 0=ab 1=general;
                   general ids namespaced +OFFSET to avoid collision.

A size-matched scale-up trio is drawn the same way at B_scale.

Residue facts (verified): concat(H+L) mean ~231.6; general seq mean ~263.
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
B_SMALL = 9_400_000
B_SCALE = 70_000_000
# general-id namespacing offset per scale (must exceed the antibody row count so
# ab seq_id [0..n_ab) never collides with general seq_id [id+OFFSET))
OFFSET = {"small": 100_000, "scale": 10_000_000}
BUDGET = {"small": B_SMALL, "scale": B_SCALE}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def draw_by_budget(nres, budget, seed):
    """Uniform random permutation; take reps until cumulative residues >= budget.
    Returns positional indices into the array `nres`."""
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


# ---------------------------------------------------------------- load pools
log("reading antibody pool (minimal cols for the draw)")
ab_min = pq.read_table(f"{D}/oas_pool_95nr.parquet", columns=["heavy_aa", "light_aa"]).to_pandas()
ab_nres = (
    ab_min["heavy_aa"].str.len().fillna(0).astype(int)
    + ab_min["light_aa"].str.len().fillna(0).astype(int)
).to_numpy()
log("ab pool reps", len(ab_nres), "total concat residues", int(ab_nres.sum()))

log("reading full antibody pool table (all cols)")
ab_full = pq.read_table(f"{D}/oas_pool_95nr.parquet")
AB_COLS = ab_full.schema.names
log("ab_full cols", len(AB_COLS))

log("reading general pool")
gn = pq.read_table(
    f"{D}/general_pool_95nr.parquet", columns=["id", "seq", "length", "source"]
).to_pandas()
gn_nres = gn["seq"].str.len().astype(int).to_numpy()
log("gn pool reps", len(gn), "total residues", int(gn_nres.sum()))

summary = {"seed": SEED, "budgets": BUDGET, "offsets": OFFSET, "scales": {}}
paths = {}

# ---------------------------------------------------------------- per-scale build
for scale in ("small", "scale"):
    B = BUDGET[scale]
    OFF = OFFSET[scale]
    half = B // 2
    log(f"==== building scale={scale} B={B} half={half} offset={OFF} ====")

    # (1) paired: draw antibody reps until cum concat residues >= B
    sel_ab = draw_by_budget(ab_nres, B, SEED)
    split_ab = assign_split(len(sel_ab), SEED)  # per-rep split
    dfp = ab_full.take(pa.array(sel_ab)).to_pandas()
    dfp["split"] = split_ab  # fresh 70/15/15 (overwrites pool split)
    dfp["regime"] = "paired"
    p_paired = f"{D}/exp_paired_{scale}.parquet"
    dfp.to_parquet(p_paired, index=False)
    log(f"paired_{scale}: n_ab={len(dfp)} residues={int(ab_nres[sel_ab].sum())} -> {p_paired}")

    # (2) unpaired: identical rows + identical split, regime marker only
    dfu = dfp.copy()
    dfu["regime"] = "unpaired"
    p_unpaired = f"{D}/exp_unpaired_{scale}.parquet"
    dfu.to_parquet(p_unpaired, index=False)
    log(f"unpaired_{scale}: n_ab={len(dfu)} -> {p_unpaired}")

    # (3) mixed: antibody half is a STRICT SUBSET of paired rows (draw from dfp)
    dfp_nres = (dfp["heavy_aa"].str.len() + dfp["light_aa"].str.len()).to_numpy()
    sel_mix_ab = draw_by_budget(dfp_nres, half, SEED)  # positional into dfp
    mix_ab = dfp.iloc[sel_mix_ab]
    ab_res = int(dfp_nres[sel_mix_ab].sum())

    sel_gn = draw_by_budget(gn_nres, half, SEED)  # general drawn fresh from pool
    mix_gn = gn.iloc[sel_gn].copy()
    gn_split = assign_split(len(mix_gn), SEED)  # fresh split for general
    gn_res = int(gn_nres[sel_gn].sum())

    n_ab = len(mix_ab)
    n_gn = len(mix_gn)
    ab_part = pd.DataFrame(
        {
            "seq_id": np.arange(n_ab, dtype=np.int64),  # 0..n_ab-1
            "orig_id": mix_ab["id"].to_numpy(),  # real ab id (join key to paired)
            "kind": np.int8(0),
            "split": mix_ab["split"].to_numpy(),  # inherited paired split
            "heavy_aa": mix_ab["heavy_aa"].to_numpy(),
            "light_aa": mix_ab["light_aa"].to_numpy(),
            "seq": pd.Series([None] * n_ab, dtype=object),
            "source": "oas",
            "n_res": dfp_nres[sel_mix_ab],
        }
    )
    gn_part = pd.DataFrame(
        {
            "seq_id": (mix_gn["id"].to_numpy() + OFF),  # namespaced
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
    p_mixed = f"{D}/exp_mixed_{scale}.parquet"
    mixed.to_parquet(p_mixed, index=False)
    log(
        f"mixed_{scale}: n_ab={n_ab}({ab_res} res) n_gn={n_gn}({gn_res} res) total={ab_res+gn_res} -> {p_mixed}"
    )

    paths[f"paired_{scale}"] = p_paired
    paths[f"unpaired_{scale}"] = p_unpaired
    paths[f"mixed_{scale}"] = p_mixed
    summary["scales"][scale] = {
        "B": B,
        "offset": OFF,
        "paired_n_ab": int(len(dfp)),
        "paired_residues": int(ab_nres[sel_ab].sum()),
        "mixed_n_ab": int(n_ab),
        "mixed_ab_residues": ab_res,
        "mixed_n_gen": int(n_gn),
        "mixed_gen_residues": gn_res,
        "mixed_residues": ab_res + gn_res,
    }

# ---------------------------------------------------------------- verification (read back)
log("==== verification (reading written parquets back) ====")
verify = {}
for scale in ("small", "scale"):
    dp = pd.read_parquet(paths[f"paired_{scale}"], columns=["id", "heavy_aa", "light_aa", "split"])
    du = pd.read_parquet(
        paths[f"unpaired_{scale}"], columns=["id", "heavy_aa", "light_aa", "split"]
    )
    dm = pd.read_parquet(paths[f"mixed_{scale}"])

    p_res = int((dp.heavy_aa.str.len() + dp.light_aa.str.len()).sum())
    u_res = int((du.heavy_aa.str.len() + du.light_aa.str.len()).sum())
    m_res = int(dm.n_res.sum())

    paired_ids = set(dp["id"])
    unpaired_ids = set(du["id"])
    mix_ab_ids = set(dm.loc[dm.kind == 0, "orig_id"])

    # (a) mixed ab ids strict subset of paired ab ids
    subset = mix_ab_ids.issubset(paired_ids)
    strict = subset and (len(mix_ab_ids) < len(paired_ids))
    # (c) unpaired == paired abs (same set + same split labels per id)
    same_set = paired_ids == unpaired_ids
    sp = dp.set_index("id")["split"]
    su = du.set_index("id")["split"]
    same_split = bool((sp.sort_index() == su.sort_index()).all()) and same_set

    # (d) split proportions + per-seq split consistency
    def props(s):
        vc = s.value_counts(normalize=True)
        return {k: round(float(vc.get(k, 0.0)), 4) for k in ("train", "val", "test")}

    # leakage check: ab seq present in both paired and mixed must share split
    j = dm.loc[dm.kind == 0, ["orig_id", "split"]].merge(
        dp[["id", "split"]], left_on="orig_id", right_on="id", suffixes=("_mix", "_paired")
    )
    ab_split_consistent = bool((j.split_mix == j.split_paired).all())
    # (b) size-match: three regime budgets within a few percent
    trio = [p_res, u_res, m_res]
    size_match = (max(trio) - min(trio)) / max(trio) < 0.05

    verify[scale] = {
        "paired_residues": p_res,
        "unpaired_residues": u_res,
        "mixed_residues": m_res,
        "paired_n_ab": len(dp),
        "mixed_n_ab": int((dm.kind == 0).sum()),
        "mixed_n_gen": int((dm.kind == 1).sum()),
        "mixed_ab_strict_subset_of_paired": bool(strict),
        "unpaired_equals_paired_abs": bool(same_split),
        "size_match_within_5pct": bool(size_match),
        "ab_split_consistent_paired_vs_mixed": ab_split_consistent,
        "split_props_paired": props(dp["split"]),
        "split_props_mixed_ab": props(dm.loc[dm.kind == 0, "split"]),
        "split_props_mixed_gen": props(dm.loc[dm.kind == 1, "split"]),
        "no_seqid_collision": int(dm.seq_id.nunique()) == len(dm),
    }
    log(f"verify[{scale}]:", json.dumps(verify[scale]))

summary["verify"] = verify
summary["paths"] = paths
outp = f"{D}/subsets_summary.json"
with open(outp, "w") as f:
    json.dump(summary, f, indent=2)
log("wrote", outp)
log("DONE")
