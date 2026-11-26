#!/usr/bin/env python
"""Characterize the paired-specific capability: which antibodies rely MOST on correct
heavy-light pairing? Re-runs the pairing-perturbation test (mask 15% of one chain, score
MLM cross-entropy under CORRECT vs SHUFFLED partner) on IgBERT-paired, recording per-
antibody Delta = CE(shuffled) - CE(correct) tied to metadata (V/J family, isotype,
B-cell type, disease). Larger Delta = more partner-dependent."""

import json, time, argparse, numpy as np, torch as th, pandas as pd


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


MODEL = "Exscientia/IgBert"  # paired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--mask_frac", type=float, default=0.15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if th.cuda.is_available() else "cpu"
    log("device", dev)
    rng = np.random.default_rng(args.seed)
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tok = AutoTokenizer.from_pretrained(MODEL, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(MODEL).to(dev).eval()
    mask_id = tok.mask_token_id
    cols = [
        "heavy_aa",
        "light_aa",
        "v_family_heavy",
        "j_family_heavy",
        "d_family_heavy",
        "v_family_light",
        "j_family_light",
        "Isotype",
        "BType",
        "Disease",
    ]
    df = pd.read_parquet(f"{BASE}/data/oas_full_annotated.parquet")
    have = [c for c in cols if c in df.columns]
    log("have cols", have)
    df = df[(df.get("split", "test") == "test") if "split" in df.columns else slice(None)]
    df = df.dropna(subset=["heavy_aa", "light_aa"]).reset_index(drop=True)
    idx = rng.permutation(len(df))[: args.n]
    sub = df.iloc[idx].reset_index(drop=True)
    log("n antibodies", len(sub))
    Hs = sub["heavy_aa"].tolist()
    Ls = sub["light_aa"].tolist()

    def sp(s):
        return " ".join(list(s))

    def pair_text(h, l):
        return sp(h), sp(l)

    def masked_ce(heavy_list, light_list, mask_chain, seeds):
        """mean per-seq CE over `seeds` masks on the given chain; partner=light_list."""
        BS = 16
        ce = np.zeros(len(heavy_list))
        for s in range(seeds):
            g = np.random.default_rng(1000 + s)
            for b0 in range(0, len(heavy_list), BS):
                H = heavy_list[b0 : b0 + BS]
                L = light_list[b0 : b0 + BS]
                enc = tok(
                    [sp(h) for h in H],
                    [sp(l) for l in L],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=400,
                )
                ids = enc["input_ids"].clone()
                att = enc["attention_mask"]
                tt = enc.get("token_type_ids")
                lab = th.full_like(ids, -100)
                for r in range(len(H)):
                    # mask positions within the chain of interest
                    seqtok = (tt[r] == (0 if mask_chain == "H" else 1)) & (att[r] == 1)
                    pos = th.where(seqtok)[0]
                    # exclude special tokens (CLS/SEP) at ends
                    pos = pos[(ids[r, pos] != tok.cls_token_id) & (ids[r, pos] != tok.sep_token_id)]
                    npos = max(1, int(len(pos) * args.mask_frac))
                    sel = pos[th.tensor(g.permutation(len(pos))[:npos])]
                    lab[r, sel] = ids[r, sel]
                    ids[r, sel] = mask_id
                ids = ids.to(dev)
                att = att.to(dev)
                lab = lab.to(dev)
                kw = dict(input_ids=ids, attention_mask=att)
                if tt is not None:
                    kw["token_type_ids"] = tt.to(dev)
                with th.no_grad():
                    out = model(**kw).logits
                lp = th.log_softmax(out.float(), -1)
                for r in range(len(H)):
                    m = lab[r] != -100
                    if m.any():
                        ce[b0 + r] += (-lp[r, m, lab[r][m]]).mean().item()
        return ce / seeds

    # correct vs shuffled partner
    shuf = rng.permutation(len(sub))
    Ls_shuf = [Ls[i] for i in shuf]
    Hs_shuf = [Hs[i] for i in shuf]
    log("scoring heavy-masked correct...")
    ceH_c = masked_ce(Hs, Ls, "H", args.seeds)
    log("scoring heavy-masked shuffled...")
    ceH_s = masked_ce(Hs, Ls_shuf, "H", args.seeds)
    log("scoring light-masked correct...")
    ceL_c = masked_ce(Hs, Ls, "L", args.seeds)
    log("scoring light-masked shuffled...")
    ceL_s = masked_ce(Hs_shuf, Ls, "L", args.seeds)
    sub["dH"] = ceH_s - ceH_c  # heavy partner-dependence
    sub["dL"] = ceL_s - ceL_c  # light partner-dependence
    sub["dpair"] = sub["dH"] + sub["dL"]
    out = {
        "n": len(sub),
        "mask_frac": args.mask_frac,
        "seeds": args.seeds,
        "overall": {
            "dH_mean": float(sub.dH.mean()),
            "dL_mean": float(sub.dL.mean()),
            "dH_fracpos": float((sub.dH > 0).mean()),
            "dL_fracpos": float((sub.dL > 0).mean()),
        },
    }
    # breakdown by metadata group (mean dpair, ranked)
    breakdown = {}
    for col in [
        "Isotype",
        "BType",
        "Disease",
        "v_family_heavy",
        "j_family_heavy",
        "d_family_heavy",
        "v_family_light",
    ]:
        if col not in sub.columns:
            continue
        gp = (
            sub.groupby(sub[col].astype(str))
            .agg(n=("dpair", "size"), dpair=("dpair", "mean"), dH=("dH", "mean"), dL=("dL", "mean"))
            .reset_index()
        )
        gp = gp[gp["n"] >= 30].sort_values("dpair", ascending=False)
        breakdown[col] = gp.to_dict(orient="records")
    out["breakdown"] = breakdown
    json.dump(out, open(f"{BASE}/results/pairing_characterize.json", "w"), indent=2, default=float)
    sub[["dH", "dL", "dpair"] + have].to_parquet(
        f"{BASE}/results/pairing_characterize_perab.parquet", index=False
    )
    log("SUMMARY", json.dumps(out["overall"], default=float))
    log("ALL DONE")


if __name__ == "__main__":
    main()
