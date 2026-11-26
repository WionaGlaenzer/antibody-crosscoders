#!/usr/bin/env python
"""All-layer per-residue activation extraction, PAIRED regime, rebuild95 small pool.
Reproduces exp_paired_small_L20.h5 format for EVERY layer 1..30 in a single GPU pass.
Metadata (seq_id/chain/region_label/split) is copied from the reference L20 store to
guarantee row-for-row consistency; only per-layer `acts` are recomputed + per-layer
train_mean/std over split==0. Models: [prot_bert, igbert_unpaired, igbert_paired].
Paired regime: heavy [SEP] light in one forward pass; drop CLS + both SEP; keep both chains.
"""

import argparse, os, sys, json, time
import numpy as np, h5py, torch as th, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractor_core import (
    MODELS,
    MODEL_KEYS,
    HID,
    NM,
    log,
    space_tok as space,
    load_tokenizers,
    load_models,
    RunningStats,
)

LAYERS = list(range(1, 31))  # hidden_states[1..30] (0 = embeddings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument(
        "--ref-l20",
        required=True,
        help="reference exp_paired_small_L20.h5 for metadata + row order",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--only-layers", default="", help="comma list to restrict (else 1..30)")
    ap.add_argument(
        "--limit", type=int, default=0, help="0=all seqs; else first N seqs (smoke test)"
    )
    args = ap.parse_args()
    layers = [int(x) for x in args.only_layers.split(",")] if args.only_layers else LAYERS
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if th.cuda.is_available() else "cpu"

    # --- metadata from reference L20 store (row order is parquet order, positional seq_id) ---
    ref = h5py.File(args.ref_l20, "r")
    ref_sid = ref["seq_id"][:]
    ref_chain = ref["chain"][:]
    ref_reg = ref["region_label"][:]
    ref_split = ref["split"][:]
    N_res = ref_sid.shape[0]
    log("ref residues", N_res)
    region_map = json.loads(ref.attrs["region_map"])
    split_map = json.loads(ref.attrs["split_map"])
    ref.close()

    df = pd.read_parquet(args.data, columns=["heavy_aa", "light_aa"])
    n_seq = len(df)
    if args.limit:
        n_seq = min(args.limit, n_seq)
        df = df.iloc[:n_seq].reset_index(drop=True)
        keep_res = int(
            np.searchsorted(ref_sid, n_seq)
        )  # ref_sid positional 0..; rows with seq_id < n_seq
        ref_sid = ref_sid[:keep_res]
        ref_chain = ref_chain[:keep_res]
        ref_reg = ref_reg[:keep_res]
        ref_split = ref_split[:keep_res]
        N_res = keep_res
        log("LIMIT", n_seq, "seqs ->", N_res, "residues")
    log("sequences", n_seq)

    toks, _ = load_tokenizers()
    mdls = load_models(device)

    # preallocate per-layer output h5 (acts only; metadata copied at end)
    files = {}
    for L in layers:
        p = os.path.join(args.out_dir, f"exp_paired_small_L{L}.h5")
        f = h5py.File(p, "w")
        f.create_dataset("acts", shape=(N_res, NM, HID), dtype="float16", chunks=(1024, NM, HID))
        files[L] = f
    stats = {L: RunningStats(nm=NM, hid=HID) for L in layers}

    write_ptr = 0
    for start in range(0, n_seq, args.batch):
        sub = df.iloc[start : start + args.batch]
        heavies = list(sub["heavy_aa"])
        lights = list(sub["light_aa"])
        # tokenize paired (text-pair) per model; identical vocab -> identical token layout
        per_model_hs = []  # list over models of {L: (B, T, HID)}
        for ti, mdl in enumerate(mdls):
            enc = toks[ti](
                [space(h) for h in heavies],
                [space(l) for l in lights],
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with th.no_grad():
                out = mdl(**enc, output_hidden_states=True)
            hs = {L: out.hidden_states[L].to(th.float16).cpu().numpy() for L in layers}
            per_model_hs.append(
                (
                    hs,
                    enc["attention_mask"].cpu().numpy(),
                    enc["input_ids"].cpu().numpy(),
                    enc["token_type_ids"].cpu().numpy() if "token_type_ids" in enc else None,
                )
            )
        # per sequence in batch: identify residue positions (drop CLS + both SEP), keep H then L
        B = len(heavies)
        for b in range(B):
            am = per_model_hs[0][1][b]
            ids = per_model_hs[0][2][b]
            valid = np.nonzero(am > 0)[0]  # non-pad positions
            # special tokens: CLS at valid[0]; two SEP tokens. drop CLS and SEPs.
            sep_id = toks[0].sep_token_id
            cls_id = toks[0].cls_token_id
            keep = [p for p in valid if ids[p] != cls_id and ids[p] != sep_id]
            n_h = len(space(heavies[b]).split())
            n_l = len(space(lights[b]).split())
            assert len(keep) == n_h + n_l, (len(keep), n_h, n_l, start + b)
            # stack acts (n_res, NM, HID) for each layer
            for L in layers:
                arr = np.stack(
                    [per_model_hs[m][0][L][b][keep] for m in range(NM)], axis=1
                )  # (n_res, NM, HID)
                files[L]["acts"][write_ptr : write_ptr + arr.shape[0]] = arr
                m0 = ref_split[write_ptr : write_ptr + arr.shape[0]] == split_map["train"]
                if m0.any():
                    stats[L].update(arr[m0])
            write_ptr += n_h + n_l
        if (start // args.batch) % 20 == 0:
            log(f"  {write_ptr}/{N_res} residues ({100*write_ptr/N_res:.0f}%)")
    assert write_ptr == N_res, (write_ptr, N_res)

    # finalize: copy metadata + write per-layer train_mean/std + attrs
    for L in layers:
        f = files[L]
        f.create_dataset("seq_id", data=ref_sid)
        f.create_dataset("chain", data=ref_chain)
        f.create_dataset("region_label", data=ref_reg)
        f.create_dataset("split", data=ref_split)
        mean, std = stats[L].finalize(var_floor=1e-12)
        f.create_dataset("train_mean", data=mean)
        f.create_dataset("train_std", data=std)
        f.attrs["model_order"] = json.dumps(MODEL_KEYS)
        f.attrs["region_map"] = json.dumps(region_map)
        f.attrs["split_map"] = json.dumps(split_map)
        f.attrs["layer"] = L
        f.close()
        log(f"wrote L{L} (train residues {stats[L].cnt})")
    log("DONE all layers", layers)


if __name__ == "__main__":
    main()
