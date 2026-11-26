#!/usr/bin/env python
"""Causal steering of IGHJ6 latent at a chosen layer (L6/L12/L20), IgBERT-paired.
Injects alpha*d_hat (crosscoder paired-decoder direction, denormalized) into the chosen
layer's hidden state; measures logP(MDV[J6]) vs logP(FDY[J4]) at masked pre-WGxG triples
on non-J6 heavy chains. Controls: unrelated IGHV latent + random direction."""

import json, time, re, argparse, numpy as np, torch as th, pandas as pd


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


MODEL = "Exscientia/IgBert"
COL_PAIRED = 2


def space(s):
    return " ".join(list(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--j6_latent", type=int, required=True)
    ap.add_argument("--ctrl_latent", type=int, required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = "cuda" if th.cuda.is_available() else "cpu"
    log("device", dev, "layer", args.layer)
    rng = np.random.default_rng(args.seed)
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tok = AutoTokenizer.from_pretrained(MODEL, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(MODEL).to(dev).eval()
    mask_id = tok.mask_token_id
    sd = th.load(args.ckpt, map_location="cpu")
    Wdec = sd["decoder.weight"]
    astd = sd["activation_std"]

    def dir_for(l):
        d = (Wdec[COL_PAIRED, l, :] * (astd[COL_PAIRED] + 1e-8)).float()
        return (d / d.norm()).to(dev)

    d_j6 = dir_for(args.j6_latent)
    d_ctrl = dir_for(args.ctrl_latent)
    d_rand = th.randn(1024, generator=th.Generator().manual_seed(1))
    d_rand = (d_rand / d_rand.norm()).to(dev)
    df = pd.read_parquet(
        f"{BASE}/data/oas_full_annotated.parquet", columns=["heavy_aa", "j_call_heavy", "split"]
    )
    df = df[
        (df["split"] == "test") & ~df["j_call_heavy"].astype(str).str.contains("IGHJ6")
    ].reset_index(drop=True)
    recs = []
    for h in df["heavy_aa"].tolist():
        m = list(re.finditer(r"WG.G", h))
        if not m:
            continue
        pos = m[-1].start()
        if pos < 3 or pos > len(h) - 1:
            continue
        recs.append((h, pos))
    rng.shuffle(recs)
    recs = recs[: args.n]
    log(f"n {len(recs)}")
    j6_ids = [tok.convert_tokens_to_ids(x) for x in "MDV"]
    j4_ids = [tok.convert_tokens_to_ids(x) for x in "FDY"]
    alphas = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    target = model.bert.encoder.layer[args.layer - 1]

    def mkhook(vec, alpha):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + alpha * vec
            return (h,) + out[1:] if isinstance(out, tuple) else h

        return hook

    def run(vec, alpha):
        handle = target.register_forward_hook(mkhook(vec, alpha)) if alpha != 0 else None
        lj6 = []
        lj4 = []
        aj = 0
        nt = 0
        BS = 16
        with th.no_grad():
            for b0 in range(0, len(recs), BS):
                batch = recs[b0 : b0 + BS]
                enc = tok([space(h) for h, _ in batch], return_tensors="pt", padding=True)
                ids = enc["input_ids"].clone()
                att = enc["attention_mask"]
                mp = []
                for r, (h, pos) in enumerate(batch):
                    tp = [pos - 2, pos - 1, pos]  # token positions = residue(pos-3..pos-1)+1
                    for t in tp:
                        ids[r, t] = mask_id
                    mp.append(tp)
                ids = ids.to(dev)
                att = att.to(dev)
                out = model(input_ids=ids, attention_mask=att).logits
                lp = th.log_softmax(out.float(), -1)
                for r, tp in enumerate(mp):
                    for k, t in enumerate(tp):
                        lj6.append(lp[r, t, j6_ids[k]].item())
                        lj4.append(lp[r, t, j4_ids[k]].item())
                        if lp[r, t].argmax().item() == j6_ids[k]:
                            aj += 1
                        nt += 1
        if handle:
            handle.remove()
        return (
            float(np.mean(lj6)),
            float(np.mean(lj4)),
            float(np.mean(np.array(lj6) - np.array(lj4))),
            aj / nt,
        )

    results = {
        "alphas": alphas,
        "layer": args.layer,
        "j6_latent": args.j6_latent,
        "ctrl_latent": args.ctrl_latent,
    }
    for name, vec in [("j6", d_j6), ("ctrl_ighv", d_ctrl), ("random", d_rand)]:
        curve = []
        for a in alphas:
            j6, j4, diff, fa = run(vec, a)
            curve.append(dict(alpha=a, logp_j6=j6, logp_j4=j4, diff=diff, frac_argmax_j6=fa))
            log(f"{name} a={a}: diff={diff:+.3f} argmaxJ6={fa:.3f}")
        results[name] = curve
    json.dump(
        results,
        open(f"{BASE}/results/steering_ighj6_{args.tag}.json", "w"),
        indent=2,
        default=float,
    )
    log("ALL DONE")


if __name__ == "__main__":
    main()
