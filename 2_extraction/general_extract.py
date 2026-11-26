"""Extract general-protein L20 activations for the 3 models (single-sequence regime),
writing an h5 matching the antibody single store schema so it can be mixed for a
combined crosscoder. All 40k Swiss-Prot proteins, split 70/15/15 by protein.

Uses extractor_core for the model/tokenizer load (with vocab-identity assertion)
and the train mean/std accumulator; the per-protein streaming + split-by-protein
logic is specific to this general-protein store and stays here.
"""

import os, sys, time
import numpy as np, h5py, torch as th, pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE, HF_HOME

ACTS = f"{BASE}/activations"
GP = f"{BASE}/data/general_protein_sample.parquet"
OUT = f"{ACTS}/general_single_L20.h5"
os.environ["HF_HOME"] = HF_HOME

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extractor_core import log, space_tok, load_tokenizers, load_models, RunningStats

LAYER = 20
HID = 1024


def main(gp=GP, out=OUT):
    dev = "cuda" if th.cuda.is_available() else "cpu"
    log(f"device {dev}")

    df = pd.read_parquet(gp).reset_index(drop=True)
    seqs = [s[:512] for s in df["seq"].tolist()]
    n_prot = len(seqs)
    log(f"general proteins: {n_prot}")
    # split by protein 70/15/15
    rng = np.random.default_rng(0)
    order = rng.permutation(n_prot)
    sp_by_prot = np.zeros(n_prot, dtype=np.int8)
    n_tr = int(0.70 * n_prot)
    n_va = int(0.15 * n_prot)
    sp_by_prot[order[:n_tr]] = 0
    sp_by_prot[order[n_tr : n_tr + n_va]] = 1
    sp_by_prot[order[n_tr + n_va :]] = 2

    toks, _ = load_tokenizers()
    mdls = load_models(dev)

    hf = h5py.File(out, "w")
    acts_ds = hf.create_dataset(
        "acts", shape=(0, 3, HID), maxshape=(None, 3, HID), dtype=np.float16, chunks=(8192, 3, HID)
    )
    split_ds = hf.create_dataset(
        "split", shape=(0,), maxshape=(None,), dtype=np.int8, chunks=(65536,)
    )
    sid_ds = hf.create_dataset(
        "seq_id", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(65536,)
    )
    kind_ds = hf.create_dataset(
        "kind", shape=(0,), maxshape=(None,), dtype=np.int8, chunks=(65536,)
    )
    stats = RunningStats(nm=3, hid=HID)
    B = 16
    t0 = time.time()
    total = 0
    for s in range(0, n_prot, B):
        idx = list(range(s, min(s + B, n_prot)))
        batch = [seqs[i] for i in idx]
        per_model = []
        for tok, mdl in zip(toks, mdls):
            enc = tok([space_tok(x) for x in batch], return_tensors="pt", padding=True).to(dev)
            with th.no_grad():
                o = mdl(**enc, output_hidden_states=True)
            hs = o.hidden_states[LAYER].to(th.float16)
            mask = enc["attention_mask"].bool()
            rows = [hs[bi, 1 : int(mask[bi].sum().item()) - 1, :] for bi in range(len(batch))]
            per_model.append(rows)
        for j, pi in enumerate(idx):
            x = th.stack([per_model[mi][j] for mi in range(3)], dim=1)  # (n_res,3,1024) fp16
            nr = x.shape[0]
            if nr == 0:
                continue
            xn = x.cpu().numpy()
            a0 = acts_ds.shape[0]
            acts_ds.resize(a0 + nr, axis=0)
            acts_ds[a0 : a0 + nr] = xn
            for ds, val in [(split_ds, sp_by_prot[pi]), (sid_ds, pi), (kind_ds, 1)]:
                d0 = ds.shape[0]
                ds.resize(d0 + nr, axis=0)
                ds[d0 : d0 + nr] = val
            if sp_by_prot[pi] == 0:
                stats.update(xn)
            total += nr
        if (s // B) % 80 == 0:
            log(f"  {min(s+B,n_prot)}/{n_prot} prot, {total} res, {time.time()-t0:.0f}s")
    tmean, tstd = stats.finalize(var_floor=1e-8)
    hf.create_dataset("train_mean", data=tmean)
    hf.create_dataset("train_std", data=tstd)
    hf.attrs["n_residues"] = total
    hf.attrs["n_proteins"] = n_prot
    hf.attrs["train_residues"] = stats.cnt
    hf.close()
    log(f"DONE {out} total_res={total} train_res={stats.cnt} mean_norm={np.linalg.norm(tmean):.2f}")


if __name__ == "__main__":
    main()
