"""General-protein pass-through, CORRECTED to use true batch-top-k (use_threshold=False).
Earlier run used use_threshold=True which gave dense pre-topk codes (median freq 0.48 - a bug).
Compares per-latent firing freq general vs antibody, by shared/gained category."""

import os, sys, json, time
import numpy as np, h5py, torch as th, pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE, HF_HOME

RES = f"{BASE}/results"
ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
CODE = f"{BASE}/code/dictionary_learning_jkminder"
sys.path.insert(0, CODE)
GP = f"{BASE}/data/general_protein_sample.parquet"
os.environ["HF_HOME"] = HF_HOME


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


dev = "cuda" if th.cuda.is_available() else "cpu"
log("device", dev)
from transformers import AutoTokenizer, AutoModel
from dictionary_learning.dictionary import BatchTopKCrossCoder

MODELS = ["Rostlab/prot_bert", "Exscientia/IgBert_unpaired", "Exscientia/IgBert"]
LAYER = 20


def space_tok(s):
    return " ".join(list(s))


df = pd.read_parquet(GP)
rng = np.random.default_rng(0)
if len(df) > 8000:
    df = df.iloc[rng.choice(len(df), 8000, replace=False)].reset_index(drop=True)
seqs = [s[:512] for s in df["seq"].tolist()]
log(f"general proteins: {len(seqs)}")
with h5py.File(f"{ACTS}/full_paired_L20.h5", "r") as h:
    tmean = h["train_mean"][:]
    tstd = h["train_std"][:]
ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_paired_L20.pt", device=dev, dtype=th.float32)
ae.eval()
cls = pd.read_csv(f"{RES}/classification_paired_L20.csv")
active_ids = cls["latent_id"].to_numpy().astype(np.int64)
A = len(active_ids)
category = cls["category"].to_numpy()
active_t = th.tensor(active_ids, device=dev)
tmean_t = th.tensor(tmean, device=dev, dtype=th.float32)
tstd_t = th.tensor(tstd, device=dev, dtype=th.float32)
log(f"crosscoder {A} active latents")
toks = [AutoTokenizer.from_pretrained(m, do_lower_case=False) for m in MODELS]
mdls = [AutoModel.from_pretrained(m, torch_dtype=th.float16).eval().to(dev) for m in MODELS]
gp_fire = np.zeros(A, dtype=np.int64)
gp_tok = 0
kcheck = []
B = 16
t0 = time.time()
for s in range(0, len(seqs), B):
    batch = seqs[s : s + B]
    per_model = []
    for tok, mdl in zip(toks, mdls):
        enc = tok([space_tok(x) for x in batch], return_tensors="pt", padding=True).to(dev)
        with th.no_grad():
            out = mdl(**enc, output_hidden_states=True)
        hs = out.hidden_states[LAYER].float()
        mask = enc["attention_mask"].bool()
        rows = [hs[bi, 1 : int(mask[bi].sum().item()) - 1, :] for bi in range(len(batch))]
        per_model.append(rows)
    for bi in range(len(batch)):
        x = th.stack([per_model[mi][bi] for mi in range(3)], dim=1)
        xn = (x - tmean_t) / tstd_t
        with th.no_grad():
            f = ae.encode(xn, use_threshold=False, normalize_activations=False)  # TRUE top-k
        code = f.index_select(1, active_t)
        fm = code > 0
        gp_fire += fm.sum(0).cpu().numpy().astype(np.int64)
        gp_tok += code.shape[0]
        kcheck.append(float(fm.sum(1).float().mean().cpu()))
    if (s // B) % 60 == 0:
        log(
            f"  {s+len(batch)}/{len(seqs)} {gp_tok}res {time.time()-t0:.0f}s meank~{np.mean(kcheck[-40:]):.1f}"
        )
gp_freq = gp_fire / max(gp_tok, 1)
log(f"general done {gp_tok} res, mean_k/tok={np.mean(kcheck):.2f}")
with h5py.File(f"{ACTS}/full_paired_L20.h5", "r") as h:
    split = h["split"][:]
    acts = h["acts"]
    testidx = np.where(split == 2)[0]
    rng2 = np.random.default_rng(1)
    testidx = np.sort(rng2.choice(testidx, min(300000, len(testidx)), replace=False))
    ab_fire = np.zeros(A, dtype=np.int64)
    ab_tok = 0
    kc2 = []
    BB = 8192
    for i in range(0, len(testidx), BB):
        rows = testidx[i : i + BB].tolist()
        x = th.tensor(acts[rows].astype(np.float32), device=dev)
        xn = (x - tmean_t) / tstd_t
        with th.no_grad():
            f = ae.encode(xn, use_threshold=False, normalize_activations=False)
        code = f.index_select(1, active_t)
        fm = code > 0
        ab_fire += fm.sum(0).cpu().numpy().astype(np.int64)
        ab_tok += code.shape[0]
        kc2.append(float(fm.sum(1).float().mean().cpu()))
ab_freq = ab_fire / max(ab_tok, 1)
log(f"antibody done {ab_tok} res, mean_k/tok={np.mean(kc2):.2f}")
out = pd.DataFrame(
    {
        "latent_id": active_ids,
        "category": category,
        "freq_general": gp_freq,
        "freq_antibody": ab_freq,
    }
)
out["log2_ratio_ab_over_gp"] = np.log2((ab_freq + 1e-6) / (gp_freq + 1e-6))
out.to_parquet(f"{RES}/genprot_passthrough_perlatent.parquet")


def summ(mask, name):
    sub = out[mask]
    return dict(
        name=name,
        n=int(mask.sum()),
        median_freq_general=float(np.median(sub.freq_general)),
        median_freq_antibody=float(np.median(sub.freq_antibody)),
        mean_freq_general=float(sub.freq_general.mean()),
        mean_freq_antibody=float(sub.freq_antibody.mean()),
        frac_active_on_general=float((sub.freq_general > 0.0005).mean()),
        frac_silent_on_general=float((sub.freq_general < 1e-5).mean()),
        median_log2ratio=float(np.median(sub.log2_ratio_ab_over_gp)),
    )


cat = out.category.astype(str)
summary = {
    "general_tokens": int(gp_tok),
    "antibody_tokens": int(ab_tok),
    "mean_k_general": float(np.mean(kcheck)),
    "mean_k_antibody": float(np.mean(kc2)),
    "shared": summ((cat == "shared-all").to_numpy(), "shared-all"),
    "gained_unp": summ(cat.str.startswith("gained-at-unpaired").to_numpy(), "gained-at-unpaired"),
    "gained_pair": summ(cat.str.startswith("gained-at-paired").to_numpy(), "gained-at-paired"),
    "base_only": summ((cat == "base-only").to_numpy(), "base-only"),
}
json.dump(summary, open(f"{RES}/genprot_passthrough_summary.json", "w"), indent=2)
log("SUMMARY " + json.dumps(summary))
log("WROTE corrected outputs")
