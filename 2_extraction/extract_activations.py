#!/usr/bin/env python
"""Token-aligned per-residue activation extraction for the 3-model crosscoder.

Three models (prot_bert, IgBert_unpaired, IgBert_paired) share an identical
BertTokenizerFast (vocab 30) -> per-residue token positions are identical across
models, so residue r aligns to the same token row in all three.

Two regimes:
  single  : heavy_aa alone AND light_aa alone (two forward passes), per-residue
            acts concatenated heavy-then-light.
  paired  : "heavy [SEP] light" in ONE forward pass (tokenizer text-pair API),
            keeping both chains' residues (CLS + both SEP dropped).

For each (layer, regime) writes an h5 {tag}_{regime}_L{L}.h5 with datasets:
  acts (N_res,3,1024) float16 stacked [prot_bert, igbert_unpaired, igbert_paired]
  seq_id int32, chain S1 ('H'/'L'), region_label int8 code, split int8 code.
Per-file per-model train-split mean/std (3,1024) stored as datasets + sidecar npz.

CONSOLIDATED extractor. --layers and --regime select the layer set and regimes;
the defaults (6,12,20,28 / single+paired) reproduce the original 4-layer x 2-regime
sweep. This one script supersedes the former near-identical copies:
  extract_scaleup.py    == --layers 20 --regime paired
  extract_single_96k.py == --layers 20 --regime single
  extract_L20.py        == --layers 20 --regime both
See REPRODUCE.md (stage 2) for the arg mapping that reproduces each store.
"""

import argparse, os, sys, json, time
import numpy as np
import pandas as pd
import torch
import h5py

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import HF_HOME

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractor_core import (
    MODELS,
    MODEL_KEYS,
    HID,
    NM,
    space_tok,
    load_tokenizers,
    load_models,
    fwd_layers,
    RunningStats,
)

os.environ.setdefault("HF_HOME", HF_HOME)

REGION_MAP = {"FR1": 0, "CDR1": 1, "FR2": 2, "CDR2": 3, "FR3": 4, "CDR3": 5, "FR4": 6}
REGION_INV = {v: k for k, v in REGION_MAP.items()}
SPLIT_MAP = {"train": 0, "val": 1, "test": 2}

LOGF = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + "\n")
        LOGF.flush()


def new_h5(path):
    f = h5py.File(path, "w")
    f.create_dataset(
        "acts",
        shape=(0, NM, HID),
        maxshape=(None, NM, HID),
        dtype="float16",
        chunks=(1024, NM, HID),
    )
    f.create_dataset("seq_id", shape=(0,), maxshape=(None,), dtype="int32", chunks=(65536,))
    f.create_dataset("chain", shape=(0,), maxshape=(None,), dtype="S1", chunks=(65536,))
    f.create_dataset("region_label", shape=(0,), maxshape=(None,), dtype="int8", chunks=(65536,))
    f.create_dataset("split", shape=(0,), maxshape=(None,), dtype="int8", chunks=(65536,))
    f.attrs["region_map"] = json.dumps(REGION_MAP)
    f.attrs["split_map"] = json.dumps(SPLIT_MAP)
    f.attrs["model_order"] = json.dumps(MODEL_KEYS)
    return f


def append(f, acts, seq_id, chain, region, split):
    n0 = f["acts"].shape[0]
    n = acts.shape[0]
    for name, arr in (
        ("acts", acts),
        ("seq_id", seq_id),
        ("chain", chain),
        ("region_label", region),
        ("split", split),
    ):
        d = f[name]
        d.resize(n0 + n, axis=0)
        d[n0 : n0 + n] = arr


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument(
        "--limit", type=int, default=0, help="0 = all sequences (smoke test uses small N)"
    )
    ap.add_argument("--tag", default="full")
    ap.add_argument(
        "--layers",
        default="6,12,20,28",
        help="comma-separated hidden-state layers (default 6,12,20,28 reproduces the sweep)",
    )
    ap.add_argument(
        "--regime",
        choices=["single", "paired", "both"],
        default="both",
        help="single, paired, or both (default) — 'both' reproduces the original 2-regime run",
    )
    args = ap.parse_args()

    layers = [int(x) for x in str(args.layers).split(",") if x != ""]
    regimes = ("single", "paired") if args.regime == "both" else (args.regime,)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    LOGF = open(args.log, "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(
        f"device={device} torch={torch.__version__} tag={args.tag} batch={args.batch} "
        f"limit={args.limit} layers={layers} regimes={regimes}"
    )

    # ---- data ----
    df = pd.read_parquet(
        args.data,
        columns=[
            "id",
            "heavy_aa",
            "light_aa",
            "region_labels_heavy",
            "region_labels_light",
            "split",
        ],
    )
    if args.limit:
        # stratified-ish small sample across splits for a representative smoke test
        df = pd.concat(
            [g.head(max(1, args.limit // 3)) for _, g in df.groupby("split")]
        ).reset_index(drop=True)
    df = df.reset_index(drop=True)
    N = len(df)
    log(f"loaded {N} sequences from {args.data}")

    # size estimate
    tot_res = int((df["heavy_aa"].str.len() + df["light_aa"].str.len()).sum())
    est_gb_file = tot_res * NM * HID * 2 / 1e9
    n_files = len(layers) * len(regimes)
    log(
        f"residues/regime={tot_res} -> per-file ~{est_gb_file:.1f} GB, "
        f"{n_files} files ~{n_files*est_gb_file:.0f} GB"
    )

    # ---- tokenizer (assert all three identical) + models (shared core) ----
    toks, tok = load_tokenizers()
    log(
        f"tokenizer OK: vocab={len(tok.get_vocab())} cls={tok.cls_token_id} "
        f"sep={tok.sep_token_id} pad={tok.pad_token_id}"
    )
    models = load_models(device)
    for m in MODELS:
        log(f"loaded {m}")

    manifest = {
        "tag": args.tag,
        "data": args.data,
        "models": MODEL_KEYS,
        "layers": layers,
        "region_map": REGION_MAP,
        "split_map": SPLIT_MAP,
        "dtype": "float16",
        "files": {},
    }
    npz_store = {}

    for regime in regimes:
        t_reg = time.time()
        files = {
            L: new_h5(os.path.join(args.out_dir, f"{args.tag}_{regime}_L{L}.h5")) for L in layers
        }
        stats = {L: RunningStats() for L in layers}
        res_written = 0
        skipped = 0
        seqs_done = 0
        next_log = 2000

        for start in range(0, N, args.batch):
            sub = df.iloc[start : start + args.batch]
            heavy = list(sub["heavy_aa"])
            light = list(sub["light_aa"])
            rl_h = list(sub["region_labels_heavy"])
            rl_l = list(sub["region_labels_light"])
            sids = [start + i for i in range(len(sub))]  # positional seq id
            splt = [SPLIT_MAP[s] for s in sub["split"]]
            nh = [len(h) for h in heavy]
            nl = [len(l) for l in light]

            if regime == "single":
                enc_h = tok([space_tok(h) for h in heavy], return_tensors="pt", padding=True)
                enc_l = tok([space_tok(l) for l in light], return_tensors="pt", padding=True)
                hs_h = [fwd_layers(mdl, enc_h, device, layers) for mdl in models]
                hs_l = [fwd_layers(mdl, enc_l, device, layers) for mdl in models]
                lens_h = enc_h["attention_mask"].sum(1).numpy()
                lens_l = enc_l["attention_mask"].sum(1).numpy()
            else:
                enc_p = tok(
                    [space_tok(h) for h in heavy],
                    [space_tok(l) for l in light],
                    return_tensors="pt",
                    padding=True,
                )
                hs_p = [fwd_layers(mdl, enc_p, device, layers) for mdl in models]
                lens_p = enc_p["attention_mask"].sum(1).numpy()

            # per-layer batch buffers
            buf = {L: [] for L in layers}
            b_sid = []
            b_chain = []
            b_reg = []
            b_split = []
            for i in range(len(sub)):
                n_h, n_l = nh[i], nl[i]
                # validate region-label alignment
                if len(rl_h[i]) != n_h or len(rl_l[i]) != n_l:
                    skipped += 1
                    continue
                if regime == "single":
                    if lens_h[i] != n_h + 2 or lens_l[i] != n_l + 2:
                        skipped += 1
                        continue
                    sl_h = slice(1, 1 + n_h)
                    sl_l = slice(1, 1 + n_l)
                    for L in layers:
                        ah = np.stack(
                            [hs_h[m][L][i, sl_h] for m in range(NM)], axis=1
                        )  # (n_h,3,1024)
                        al = np.stack([hs_l[m][L][i, sl_l] for m in range(NM)], axis=1)
                        buf[L].append(np.concatenate([ah, al], axis=0))
                else:
                    # [CLS] h(1..n_h) [SEP] l(n_h+2 .. n_h+1+n_l) [SEP]
                    if lens_p[i] != n_h + n_l + 3:
                        skipped += 1
                        continue
                    sl_h = slice(1, 1 + n_h)
                    sl_l = slice(n_h + 2, n_h + 2 + n_l)
                    for L in layers:
                        ah = np.stack([hs_p[m][L][i, sl_h] for m in range(NM)], axis=1)
                        al = np.stack([hs_p[m][L][i, sl_l] for m in range(NM)], axis=1)
                        buf[L].append(np.concatenate([ah, al], axis=0))
                # metadata for this seq (same across layers)
                b_sid.extend([sids[i]] * (n_h + n_l))
                b_chain.extend([b"H"] * n_h + [b"L"] * n_l)
                b_reg.extend(
                    [REGION_MAP.get(x, -1) for x in rl_h[i]]
                    + [REGION_MAP.get(x, -1) for x in rl_l[i]]
                )
                b_split.extend([splt[i]] * (n_h + n_l))

            if b_sid:
                a_sid = np.asarray(b_sid, np.int32)
                a_chain = np.asarray(b_chain, "S1")
                a_reg = np.asarray(b_reg, np.int8)
                a_split = np.asarray(b_split, np.int8)
                train_mask = a_split == SPLIT_MAP["train"]
                for L in layers:
                    acts = np.concatenate(buf[L], axis=0).astype(np.float16)  # (n_res,3,1024)
                    append(files[L], acts, a_sid, a_chain, a_reg, a_split)
                    if train_mask.any():
                        stats[L].update(acts[train_mask])
                res_written += a_sid.shape[0]
            seqs_done += len(sub)
            if seqs_done >= next_log:
                log(
                    f"  [{regime}] {seqs_done}/{N} seqs, {res_written} residues written, skipped={skipped}"
                )
                next_log += 2000

        # finalize norm stats + write into each file
        for L in layers:
            mean, std = stats[L].finalize(var_floor=0.0)
            f = files[L]
            f.create_dataset("train_mean", data=mean)  # (3,1024)
            f.create_dataset("train_std", data=std)
            f.attrs["train_residue_count"] = stats[L].cnt
            npz_store[f"{regime}_L{L}_mean"] = mean
            npz_store[f"{regime}_L{L}_std"] = std

            # per-file manifest entry
            n_res = f["acts"].shape[0]
            sp = f["split"][:]
            ch = f["chain"][:]
            rg = f["region_label"][:]
            split_counts = {k: int((sp == v).sum()) for k, v in SPLIT_MAP.items()}
            chain_counts = {"H": int((ch == b"H").sum()), "L": int((ch == b"L").sum())}
            region_hist = {
                REGION_INV.get(int(code), str(code)): int((rg == code).sum())
                for code in sorted(set(rg.tolist()))
            }
            path = os.path.join(args.out_dir, f"{args.tag}_{regime}_L{L}.h5")
            manifest["files"][f"{args.tag}_{regime}_L{L}.h5"] = {
                "path": path,
                "layer": L,
                "regime": regime,
                "n_residues": int(n_res),
                "n_sequences": int(seqs_done - skipped),
                "split_residue_counts": split_counts,
                "chain_counts": chain_counts,
                "region_hist": region_hist,
                "bytes": int(n_res) * NM * HID * 2,
                "train_mean_norm": [float(np.linalg.norm(mean[m])) for m in range(NM)],
                "train_std_norm": [float(np.linalg.norm(std[m])) for m in range(NM)],
                "dtype": "float16",
            }
            f.close()
        log(f"[{regime}] done: {res_written} residues, skipped={skipped}, {time.time()-t_reg:.0f}s")

    np.savez(args.npz, **npz_store)
    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, indent=1)
    log(f"wrote manifest {args.manifest} and npz {args.npz}")

    # ---- SANITY: reload each file ----
    log("=== SANITY ===")
    for fn, meta in manifest["files"].items():
        with h5py.File(meta["path"], "r") as f:
            acts = f["acts"]
            n = acts.shape[0]
            samp = acts[: min(4096, n)].astype(np.float32)
            nan = int(np.isnan(samp).sum())
            inf = int(np.isinf(samp).sum())
            pm = [float(samp[:, m].mean()) for m in range(NM)]
            ps = [float(samp[:, m].std()) for m in range(NM)]
            log(
                f"{fn}: shape={acts.shape} nan={nan} inf={inf} "
                f"sample_mean={['%.3f'%x for x in pm]} sample_std={['%.3f'%x for x in ps]} "
                f"chain={meta['chain_counts']} splits={meta['split_residue_counts']}"
            )
            assert nan == 0 and inf == 0, f"NaN/Inf in {fn}"
    log("SANITY OK")


if __name__ == "__main__":
    main()
