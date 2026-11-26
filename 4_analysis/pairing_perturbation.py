#!/usr/bin/env python
"""Pairing-perturbation behavioral test: does IgBERT-paired USE the correct H-L pairing
more than IgBERT-unpaired? For each antibody we build heavy[SEP]light with the CORRECT
light partner vs a SHUFFLED (mismatched) light chain, mask 15% of HEAVY residues (same
positions in both conditions), and compute MLM cross-entropy on the masked heavy tokens.
Delta = loss(shuffled) - loss(correct): positive => the model's heavy-chain predictions
depend on the light partner. If Delta_paired > Delta_unpaired, paired fine-tuning taught
the model to exploit cross-chain context.

Also runs the symmetric test (mask LIGHT residues, correct vs shuffled heavy).
"""

import json, time, argparse, numpy as np, torch as th
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


HF = f"{BASE}/caches/hf"
MODELS = {"unpaired": "Exscientia/IgBert_unpaired", "paired": "Exscientia/IgBert"}


def space(s):
    return " ".join(list(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--mask_frac", type=float, default=0.15)
    ap.add_argument("--n_mask_seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if th.cuda.is_available() else "cpu"
    log("device", dev)
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(
        f"{BASE}/data/oas_full_annotated.parquet", columns=["heavy_aa", "light_aa", "split"]
    )
    df = df[df["split"] == "test"].reset_index(drop=True)
    df = df[
        df["heavy_aa"].str.len().between(90, 140) & df["light_aa"].str.len().between(90, 120)
    ].reset_index(drop=True)
    if len(df) > args.n:
        df = df.iloc[rng.choice(len(df), args.n, replace=False)].reset_index(drop=True)
    n = len(df)
    log(f"n antibodies {n}")
    heavy = df["heavy_aa"].tolist()
    light = df["light_aa"].tolist()
    # shuffled light: derangement (each paired with a different one's light)
    perm = rng.permutation(n)
    for i in range(n):
        if perm[i] == i:
            perm[i] = (i + 1) % n
    light_shuf = [light[perm[i]] for i in range(n)]
    heavy_shuf = [heavy[perm[i]] for i in range(n)]

    tok = AutoTokenizer.from_pretrained(MODELS["paired"], do_lower_case=False)
    mask_id = tok.mask_token_id
    sep_id = tok.sep_token_id
    cls_id = tok.cls_token_id
    results = {}
    for mname, mpath in MODELS.items():
        log(f"loading {mname}")
        model = AutoModelForMaskedLM.from_pretrained(mpath).to(dev).eval()

        # accumulate per-antibody masked-heavy CE under correct vs shuffled light
        def masked_ce(heavy_list, light_list, mask_target="heavy"):
            """Return per-seq mean CE over masked residues of mask_target chain,
            averaged over n_mask_seeds. Correct vs shuffled controlled by caller via light_list."""
            per = np.zeros(n)
            BS = 32
            for ms in range(args.n_mask_seeds):
                mrng = np.random.default_rng(1000 + ms)
                for b0 in range(0, n, BS):
                    hs = heavy_list[b0 : b0 + BS]
                    ls = light_list[b0 : b0 + BS]
                    enc = tok(
                        [space(h) for h in hs],
                        [space(l) for l in ls],
                        return_tensors="pt",
                        padding=True,
                    )
                    ids = enc["input_ids"].clone()
                    att = enc["attention_mask"]
                    labels = th.full_like(ids, -100)
                    # heavy tokens occupy positions 1..len(h); light after first SEP
                    for r, (h, l) in enumerate(zip(hs, ls)):
                        nh = len(h)
                        if mask_target == "heavy":
                            cand = np.arange(1, 1 + nh)  # token positions of heavy residues
                        else:
                            cand = np.arange(1 + nh + 1, 1 + nh + 1 + len(l))  # after [SEP]
                        k = max(1, int(len(cand) * args.mask_frac))
                        sel = mrng.choice(cand, k, replace=False)
                        labels[r, sel] = ids[r, sel]
                        ids[r, sel] = mask_id
                    ids = ids.to(dev)
                    att = att.to(dev)
                    labels = labels.to(dev)
                    with th.no_grad():
                        out = model(input_ids=ids, attention_mask=att).logits
                    # per-seq mean CE over masked positions
                    lp = th.log_softmax(out.float(), -1)
                    for r in range(len(hs)):
                        m = labels[r] != -100
                        if m.any():
                            ce = -lp[r, m, labels[r, m]].mean().item()
                            per[b0 + r] += ce
            per /= args.n_mask_seeds
            return per

        log(f"  {mname}: heavy-masked, correct light")
        ce_h_correct = masked_ce(heavy, light, "heavy")
        log(f"  {mname}: heavy-masked, shuffled light")
        ce_h_shuf = masked_ce(heavy, light_shuf, "heavy")
        log(f"  {mname}: light-masked, correct heavy")
        ce_l_correct = masked_ce(heavy, light, "light")
        log(f"  {mname}: light-masked, shuffled heavy")
        ce_l_shuf = masked_ce(heavy_shuf, light, "light")
        results[mname] = dict(
            heavy_ce_correct=float(ce_h_correct.mean()),
            heavy_ce_shuffled=float(ce_h_shuf.mean()),
            heavy_delta=float((ce_h_shuf - ce_h_correct).mean()),
            heavy_delta_median=float(np.median(ce_h_shuf - ce_h_correct)),
            heavy_frac_pos=float((ce_h_shuf > ce_h_correct).mean()),
            light_ce_correct=float(ce_l_correct.mean()),
            light_ce_shuffled=float(ce_l_shuf.mean()),
            light_delta=float((ce_l_shuf - ce_l_correct).mean()),
            light_frac_pos=float((ce_l_shuf > ce_l_correct).mean()),
        )
        np.savez(
            f"{BASE}/results/pairing_pert_{mname}.npz",
            ce_h_correct=ce_h_correct,
            ce_h_shuf=ce_h_shuf,
            ce_l_correct=ce_l_correct,
            ce_l_shuf=ce_l_shuf,
        )
        log(
            f"  {mname} heavy_delta {results[mname]['heavy_delta']:.4f} (frac_pos {results[mname]['heavy_frac_pos']:.2f}) "
            f"light_delta {results[mname]['light_delta']:.4f}"
        )
        del model
        th.cuda.empty_cache()
    # paired-effect: is the shuffle penalty larger for paired than unpaired?
    results["paired_minus_unpaired"] = dict(
        heavy_delta_diff=results["paired"]["heavy_delta"] - results["unpaired"]["heavy_delta"],
        light_delta_diff=results["paired"]["light_delta"] - results["unpaired"]["light_delta"],
    )
    results["config"] = dict(n=n, mask_frac=args.mask_frac, n_mask_seeds=args.n_mask_seeds)
    json.dump(
        results,
        open(f"{BASE}/results/pairing_perturbation_summary.json", "w"),
        indent=2,
        default=float,
    )
    log("SUMMARY", json.dumps(results, default=float)[:900])
    log("ALL DONE")


if __name__ == "__main__":
    main()
