#!/usr/bin/env python
"""Causal steering validation of the IGHJ6 concept latent (paired_L20 latent 356).

Method (Haque et al. 2026 style, adapted to a BERT MLM crosscoder latent):
  - Take the crosscoder's paired-model decoder direction for latent 356 (a 1024-d
    direction in IgBERT-paired's L20 residual stream), denormalize by activation_std.
  - Add alpha * d_hat to the layer-20 hidden state (hook on encoder.layer[19] output)
    during a forward pass of IgBERT-paired.
  - Test on non-IGHJ6 heavy chains (true germline J4/J1/J2/J3/J5). Mask the 3 residues
    immediately preceding the 'WGxG' J-framework motif (the J-discriminative triple:
    IGHJ6='MDV', IGHJ4='FDY', etc.). Measure the model's predicted log-prob of the
    IGHJ6 triple 'MDV' vs the true-germline/J4 triple 'FDY' at those masked positions,
    as a function of steering strength alpha.
  - CAUSAL claim: if clamping the J6 latent up shifts predictions toward MDV
    monotonically with alpha, the latent causally controls IGHJ6 identity.
  - CONTROLS: (1) a random unit direction; (2) an unrelated concept latent (an IGHV
    detector). Neither should produce the J6 shift.
"""

import json, time, re, argparse, numpy as np, torch as th, pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


MODEL = "Exscientia/IgBert"  # paired
CKPT = f"{BASE}/checkpoints/cc_paired_L20.pt"
J6_LATENT = 356  # paired_L20 IGHJ6 detector, F1 0.9965
COL_PAIRED = 2  # decoder model column order [prot, unp, paired]
LAYER = 20  # hidden_states[20] = output of encoder.layer[19]


def space(s):
    return " ".join(list(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if th.cuda.is_available() else "cpu"
    log("device", dev)
    rng = np.random.default_rng(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL, do_lower_case=False)
    model = AutoModelForMaskedLM.from_pretrained(MODEL).to(dev).eval()
    mask_id = tok.mask_token_id
    # decoder directions
    sd = th.load(CKPT, map_location="cpu")
    Wdec = sd["decoder.weight"]  # (3, dict, 1024)
    astd = sd["activation_std"]  # (3,1024)

    def dir_for(latent):
        d = (Wdec[COL_PAIRED, latent, :] * (astd[COL_PAIRED] + 1e-8)).float()
        return (d / d.norm()).to(dev)

    d_j6 = dir_for(J6_LATENT)
    # control latent: paired_L20 latent 2 = perfect IGHV6 detector (F1 1.0), unrelated to J
    d_ctrl_latent = dir_for(2)
    d_rand = th.randn(1024, generator=th.Generator().manual_seed(1))
    d_rand = (d_rand / d_rand.norm()).to(dev)
    # typical L20 hidden norm for scaling alpha into interpretable units
    # data
    df = pd.read_parquet(
        f"{BASE}/data/oas_full_annotated.parquet", columns=["heavy_aa", "j_call_heavy", "split"]
    )
    df = df[
        (df["split"] == "test") & ~df["j_call_heavy"].astype(str).str.contains("IGHJ6")
    ].reset_index(drop=True)
    # find WGxG motif and the 3 residues before it
    recs = []
    for h in df["heavy_aa"].tolist():
        m = list(re.finditer(r"WG.G", h))
        if not m:
            continue
        pos = m[-1].start()  # index of 'W'
        if pos < 3 or pos > len(h) - 1:
            continue
        recs.append((h, pos))  # triple = h[pos-3:pos]
    rng.shuffle(recs)
    recs = recs[: args.n]
    log(f"n test seqs {len(recs)}")
    # target triples
    J6 = "MDV"
    J4 = "FDY"
    j6_ids = [tok.convert_tokens_to_ids(c) for c in J6]
    j4_ids = [tok.convert_tokens_to_ids(c) for c in J4]
    log("J6 ids", j6_ids, "J4 ids", j4_ids)

    alphas = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    hidden = {}

    def make_hook(vec, alpha):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + alpha * vec
            return (h,) + out[1:] if isinstance(out, tuple) else h

        return hook

    target_layer = model.bert.encoder.layer[LAYER - 1]  # hidden_states[20] = layer[19] output

    def run(vec, alpha):
        """Return mean logP(J6 triple) - logP(J4 triple) over masked positions, + frac argmax==J6."""
        handle = (
            target_layer.register_forward_hook(make_hook(vec, alpha))
            if alpha != 0 or vec is not None
            else None
        )
        lp_j6 = []
        lp_j4 = []
        arg_j6 = 0
        ntot = 0
        BS = 16
        with th.no_grad():
            for b0 in range(0, len(recs), BS):
                batch = recs[b0 : b0 + BS]
                enc = tok([space(h) for h, _ in batch], return_tensors="pt", padding=True)
                ids = enc["input_ids"].clone()
                att = enc["attention_mask"]
                # mask the 3 pre-WGxG residues: token position = residue_index+1 (CLS at 0)
                maskpos = []
                for r, (h, pos) in enumerate(batch):
                    tp = [pos - 3 + 1, pos - 2 + 1, pos - 1 + 1]  # token positions
                    for t in tp:
                        ids[r, t] = mask_id
                    maskpos.append(tp)
                ids = ids.to(dev)
                att = att.to(dev)
                out = model(input_ids=ids, attention_mask=att).logits
                lp = th.log_softmax(out.float(), -1)
                for r, tp in enumerate(maskpos):
                    for k, t in enumerate(tp):
                        lp_j6.append(lp[r, t, j6_ids[k]].item())
                        lp_j4.append(lp[r, t, j4_ids[k]].item())
                        if lp[r, t].argmax().item() == j6_ids[k]:
                            arg_j6 += 1
                        ntot += 1
        if handle:
            handle.remove()
        return (
            float(np.mean(lp_j6)),
            float(np.mean(lp_j4)),
            float(np.mean(np.array(lp_j6) - np.array(lp_j4))),
            arg_j6 / ntot,
        )

    results = {"alphas": alphas, "j6_latent": J6_LATENT}
    for name, vec in [("j6", d_j6), ("ctrl_ighv", d_ctrl_latent), ("random", d_rand)]:
        curve = []
        for a in alphas:
            j6, j4, diff, fa = run(vec, a)
            curve.append(dict(alpha=a, logp_j6=j6, logp_j4=j4, diff=diff, frac_argmax_j6=fa))
            log(
                f"{name} a={a}: logP(MDV)={j6:.3f} logP(FDY)={j4:.3f} diff={diff:+.3f} argmaxJ6={fa:.3f}"
            )
        results[name] = curve
    # also negative steering on j6
    negcurve = []
    for a in [-8.0, -4.0, -2.0]:
        j6, j4, diff, fa = run(d_j6, a)
        negcurve.append(dict(alpha=a, diff=diff, frac_argmax_j6=fa))
        log(f"j6 NEG a={a}: diff={diff:+.3f} argmaxJ6={fa:.3f}")
    results["j6_negative"] = negcurve
    json.dump(results, open(f"{BASE}/results/steering_ighj6.json", "w"), indent=2, default=float)
    log("SUMMARY", json.dumps(results, default=float)[:800])
    log("ALL DONE")


if __name__ == "__main__":
    main()
