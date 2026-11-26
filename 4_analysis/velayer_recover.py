#!/usr/bin/env python3
"""Recover VE + decnorm for velayer crosscoders from their _last.pt checkpoints
(walltime-killed before summary write). Replicates the trainer's exact val eval:
VE = 1 - sum(err^2)/sum(var) per model on normalized val activations (use_threshold=False),
n_active = #latents with val firing freq > 1e-4, decoder-norm categories.
Dumps summary_{regime}_velayer_L{L}.json and decnorm_{regime}_velayer_L{L}.npz per available ckpt.
"""

import os, sys, json, glob, time, numpy as np, h5py, torch as th
import types

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.modules["nnsight"] = types.ModuleType("nnsight")
sys.modules["nnsight"].LanguageModel = object

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
os.environ["HF_HOME"] = f"{BASE}/caches/hf"
ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


from dictionary_learning.dictionary import BatchTopKCrossCoder

MNAMES = ["prot_bert", "igbert_unpaired", "igbert_paired"]


def classify(Wdec):
    n = th.linalg.norm(Wdec, dim=2).transpose(0, 1)
    rel = n / (n.sum(1, keepdim=True) + 1e-12)
    rp, ru, rpr = rel[:, 0], rel[:, 1], rel[:, 2]
    HI, LO = 0.5, 0.2
    cat = th.full((rel.shape[0],), 5, dtype=th.int64, device=rel.device)
    cat[(rp > LO) & (ru > LO) & (rpr > LO)] = 0
    cat[(rp > HI) & (ru < LO) & (rpr < LO)] = 1
    cat[(rp < LO) & (ru > LO) & (rpr > LO)] = 2
    cat[(ru > HI) & (rpr < LO)] = 4
    cat[(rp < LO) & (ru < LO) & (rpr > HI)] = 3
    return n, rel, cat


def do(regime, L):
    ck = f"{CKPT}/cc_{regime}_velayer_L{L}_last.pt"
    if not os.path.exists(ck):
        log(f"skip {regime} L{L} (no ckpt)")
        return
    h5 = f"{ACTS}/exp_{regime}_small_L{L}.h5"
    ae = BatchTopKCrossCoder.from_pretrained(ck, device=dev, dtype=th.float32)
    ae.eval()
    with h5py.File(h5, "r") as h:
        sp = h["split"][:]
        acts = h["acts"]
        vidx = np.nonzero(sp == 1)[0]
        # cap val eval like trainer subset (use up to 2M residues for speed, deterministic)
        if len(vidx) > 2_000_000:
            vidx = vidx[np.linspace(0, len(vidx) - 1, 2_000_000).astype(int)]
        tot_e = th.zeros(3, device=dev)
        tot_var = th.zeros(3, device=dev)
        act_any = th.zeros(ae.dict_size, device=dev)
        nb = 0
        EB = 16384
        with th.no_grad():
            for s in range(0, len(vidx), EB):
                ii = vidx[s : s + EB]
                xb = th.tensor(
                    acts[ii.min() : ii.max() + 1][ii - ii.min()].astype(np.float32), device=dev
                )
                xn = ae.normalize_activations(xb, inplace=False)
                f = ae.encode(xn, normalize_activations=False, use_threshold=False)
                xhat = ae.decode(f, denormalize_activations=False)
                e = xn - xhat
                tot_e += e.pow(2).sum(-1).sum(0)
                tot_var += (xn - xn.mean(0, keepdim=True)).pow(2).sum(-1).sum(0)
                act_any += (f > 0).float().sum(0)
                nb += xb.shape[0]
                if (s // EB) % 40 == 0:
                    log(f"  {regime} L{L} eval {s+len(ii)}/{len(vidx)}")
        ve = (1.0 - tot_e / (tot_var + 1e-12)).cpu().tolist()
        freq = (act_any / max(nb, 1)).cpu().numpy()
        n_active = int((freq > 1e-4).sum())
    n, rel, cat = classify(ae.decoder.weight.detach())
    rel_np = rel.cpu().numpy()
    cat_np = cat.cpu().numpy()
    n_np = n.cpu().numpy()
    from collections import Counter

    cc = Counter(cat_np[freq > 1e-4].tolist())
    summ = dict(
        regime=f"{regime}_velayer",
        layer=L,
        seed=0,
        n_active=n_active,
        n_val=nb,
        final_var_expl={MNAMES[i]: ve[i] for i in range(3)},
        category_counts={int(k): int(v) for k, v in cc.items()},
        recovered_from="_last.pt walltime-killed",
    )
    json.dump(summ, open(f"{RES}/summary_{regime}_velayer_L{L}.json", "w"))
    np.savez(
        f"{RES}/decnorm_{regime}_velayer_L{L}.npz",
        rel=rel_np,
        dnorm=n_np,
        cat=cat_np,
        val_act_freq=freq,
    )
    log(f"{regime} L{L} DONE VE {[round(v,3) for v in ve]} n_active {n_active} cats {dict(cc)}")


for regime in ["paired", "unpaired", "mixed"]:
    for L in [4, 12, 28]:
        try:
            do(regime, L)
        except Exception as ex:
            log(f"ERR {regime} L{L}: {str(ex)[:120]}")
log("ALL DONE")
