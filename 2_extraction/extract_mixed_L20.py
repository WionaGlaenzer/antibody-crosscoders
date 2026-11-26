#!/usr/bin/env python
"""L20 per-residue activation extraction for the MIXED regime from a pre-balanced
exp_mixed_*.parquet (schema: seq_id, orig_id, kind, split, heavy_aa, light_aa, seq).

The mixed parquet is already 50/50 antibody/general at residue level with train/val/
test splits assigned in Phase 1, so activations are extracted directly (no post-hoc
resampling from separate antibody/general stores as the old build_mixed.py did).

  kind 0 = antibody : heavy_aa alone AND light_aa alone (two single-chain forwards),
                      per-residue acts concatenated heavy-then-light.
  kind 1 = general  : seq alone (single-chain forward), truncated to 510 residues so
                      CLS+SEP stay within the models' 512-position limit.

Three models (prot_bert, IgBert_unpaired, IgBert_paired) share one BertTokenizerFast
(vocab 30) -> identical token positions, so residue r aligns across all three.

Writes an h5 matching the antibody single-store schema (+ kind):
  acts (N_res,3,1024) float16 [prot_bert, igbert_unpaired, igbert_paired],
  seq_id int32 (namespaced in-parquet), split int8, kind int8,
  train_mean/train_std (3,1024) fp32 over split==0 only.
"""

import argparse, os, sys, json, time
import numpy as np
import pandas as pd
import torch
import h5py

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import HF_HOME

os.environ.setdefault("HF_HOME", HF_HOME)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractor_core import (
    MODELS,
    MODEL_KEYS,
    HID,
    NM,
    space_tok,
    load_tokenizers,
    load_models,
    RunningStats,
    sanity_check,
)

LAYER = 20
MAXRES = 510
SPLIT_MAP = {"train": 0, "val": 1, "test": 2}

LOGF = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + "\n")
        LOGF.flush()


@torch.no_grad()
def fwd_layer(model, enc, device):
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    return out.hidden_states[LAYER].to(torch.float16).cpu().numpy()


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)  # full output h5 path
    ap.add_argument("--log", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag", default="exp_mixed_small")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    LOGF = open(args.log, "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(
        f"device={device} torch={torch.__version__} tag={args.tag} batch={args.batch} layer={LAYER}"
    )

    df = pd.read_parquet(
        args.data, columns=["seq_id", "kind", "split", "heavy_aa", "light_aa", "seq"]
    )
    log(
        f"loaded {len(df)} rows from {args.data}; kind counts {df['kind'].value_counts().to_dict()}"
    )

    # flatten to a chain stream: (seq_str, seq_id, split_code, kind)
    chains = []
    for r in df.itertuples(index=False):
        sc = SPLIT_MAP[r.split]
        if int(r.kind) == 0:
            chains.append((r.heavy_aa, int(r.seq_id), sc, 0))
            chains.append((r.light_aa, int(r.seq_id), sc, 0))
        else:
            chains.append((r.seq[:MAXRES], int(r.seq_id), sc, 1))
    log(f"flattened to {len(chains)} single chains")

    toks, tok = load_tokenizers()
    log(f"tokenizer OK vocab={len(tok.get_vocab())} cls={tok.cls_token_id} sep={tok.sep_token_id}")

    models = load_models(device)
    log("models loaded")

    hf = h5py.File(args.out, "w")
    acts_ds = hf.create_dataset(
        "acts",
        shape=(0, NM, HID),
        maxshape=(None, NM, HID),
        dtype="float16",
        chunks=(8192, NM, HID),
    )
    sid_ds = hf.create_dataset(
        "seq_id", shape=(0,), maxshape=(None,), dtype="int32", chunks=(65536,)
    )
    split_ds = hf.create_dataset(
        "split", shape=(0,), maxshape=(None,), dtype="int8", chunks=(65536,)
    )
    kind_ds = hf.create_dataset("kind", shape=(0,), maxshape=(None,), dtype="int8", chunks=(65536,))
    hf.attrs["model_order"] = json.dumps(MODEL_KEYS)
    hf.attrs["split_map"] = json.dumps(SPLIT_MAP)
    hf.attrs["kind_map"] = json.dumps({"antibody": 0, "general": 1})

    def append(name, arr):
        d = hf[name]
        n0 = d.shape[0]
        n = arr.shape[0]
        d.resize(n0 + n, axis=0)
        d[n0 : n0 + n] = arr

    stats = RunningStats(nm=NM, hid=HID)
    res_written = 0
    skipped = 0
    done = 0
    next_log = 20000
    t0 = time.time()

    for start in range(0, len(chains), args.batch):
        batch = chains[start : start + args.batch]
        texts = [space_tok(x[0]) for x in batch]
        enc = tok(texts, return_tensors="pt", padding=True)
        lens = enc["attention_mask"].sum(1).numpy()
        hs = [fwd_layer(mdl, enc, device) for mdl in models]

        buf = []
        b_sid = []
        b_split = []
        b_kind = []
        for i, (seq, sid, sc, kd) in enumerate(batch):
            n = len(seq)
            if n == 0 or lens[i] != n + 2:
                skipped += 1
                continue
            sl = slice(1, 1 + n)
            x = np.stack([hs[m][i, sl] for m in range(NM)], axis=1)  # (n,3,1024)
            buf.append(x.astype(np.float16))
            b_sid.extend([sid] * n)
            b_split.extend([sc] * n)
            b_kind.extend([kd] * n)

        if buf:
            acts = np.concatenate(buf, axis=0)
            a_sid = np.asarray(b_sid, np.int32)
            a_split = np.asarray(b_split, np.int8)
            a_kind = np.asarray(b_kind, np.int8)
            append("acts", acts)
            append("seq_id", a_sid)
            append("split", a_split)
            append("kind", a_kind)
            tr = a_split == 0
            if tr.any():
                stats.update(acts[tr])
            res_written += acts.shape[0]
        done += len(batch)
        if done >= next_log:
            log(
                f"  {done}/{len(chains)} chains, {res_written} residues, skipped={skipped}, {time.time()-t0:.0f}s"
            )
            next_log += 20000

    acc_cnt = stats.cnt
    mean, std = stats.finalize(var_floor=0.0)
    hf.create_dataset("train_mean", data=mean)
    hf.create_dataset("train_std", data=std)
    hf.attrs["n_residues"] = res_written
    hf.attrs["train_residue_count"] = acc_cnt

    sp = hf["split"][:]
    kd = hf["kind"][:]
    split_counts = {k: int((sp == v).sum()) for k, v in SPLIT_MAP.items()}
    kind_counts = {"antibody": int((kd == 0).sum()), "general": int((kd == 1).sum())}
    manifest = {
        "tag": args.tag,
        "data": args.data,
        "out": args.out,
        "models": MODEL_KEYS,
        "layer": LAYER,
        "n_residues": int(res_written),
        "n_chains": len(chains),
        "skipped": skipped,
        "split_residue_counts": split_counts,
        "kind_residue_counts": kind_counts,
        "train_residue_count": acc_cnt,
        "dtype": "float16",
        "train_mean_norm": [float(np.linalg.norm(mean[m])) for m in range(NM)],
        "train_std_norm": [float(np.linalg.norm(std[m])) for m in range(NM)],
    }
    hf.close()
    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, indent=1)
    log(
        f"wrote {args.out} residues={res_written} train={acc_cnt} "
        f"splits={split_counts} kinds={kind_counts}"
    )

    log("=== SANITY ===")
    with h5py.File(args.out, "r") as f:
        acts = f["acts"]
        n = acts.shape[0]
        samp = acts[: min(8192, n)].astype(np.float32)
        nan = int(np.isnan(samp).sum())
        inf = int(np.isinf(samp).sum())
        tm = f["train_mean"][:]
        ts = f["train_std"][:]
        mnan = int(np.isnan(tm).sum()) + int(np.isnan(ts).sum())
        log(
            f"shape={acts.shape} nan={nan} inf={inf} norm_nan={mnan} "
            f"mean_norms={[round(float(np.linalg.norm(tm[m])),2) for m in range(NM)]}"
        )
        assert nan == 0 and inf == 0 and mnan == 0, "NaN/Inf in mixed store"
    log("SANITY OK")


if __name__ == "__main__":
    main()
