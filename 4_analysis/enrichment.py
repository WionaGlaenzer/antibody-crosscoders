#!/usr/bin/env python
"""Per-latent IGHJ6 enrichment (fold-change) vs F1 detection, paired_L20.
Enrichment = mean(latent activation | IGHJ6) / mean(latent activation | not IGHJ6),
computed sequence-level (mean-of-nonzero-residue activation per sequence, Haque V/J convention).
Compare to the F1-based detection to see if the two metrics agree on which latents are IGHJ6 detectors.
"""

import h5py, numpy as np, pandas as pd, torch as th, json, os, sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

DEV = "cuda" if th.cuda.is_available() else "cpu"
sys.path.insert(0, BASE + "/code/dictionary_learning_jkminder")
from dictionary_learning.dictionary import BatchTopKCrossCoder

COL_PAIRED = 2
LAYER = 20


def log(*a):
    print(*a, flush=True)


ae = BatchTopKCrossCoder.from_pretrained(BASE + "/checkpoints/cc_paired_L20.pt", device=DEV)
ae.eval()
log("loaded crosscoder")

# test-split sequences + labels
df = pd.read_parquet(BASE + "/data/oas_full_annotated.parquet")
test = df[df.split == "test"].reset_index(drop=True)
j6 = (test["j_family_heavy"] == "IGHJ6").values
log(f"test seqs {len(test)}  IGHJ6 {j6.sum()} ({j6.mean()*100:.1f}%)")

# activations store: full_paired_L20.h5 has acts (N_res,3,1024) aligned to residues; need seq boundaries.
# We recompute per-sequence latent means by streaming the store with the manifest's residue index.
h5 = h5py.File(BASE + "/activations/full_paired_L20.h5", "r")
acts = h5["acts"]
N = acts.shape[0]
seq_id = h5["seq_id"][:]
split = h5["split"][:]
chain = h5["chain"][:]
tmean = th.tensor(h5["train_mean"][:], device=DEV)
tstd = th.tensor(h5["train_std"][:], device=DEV)
# split codes: 0 train, 1 val, 2 test (match extraction). keep TEST heavy residues.
SPLIT_TEST = 2
mask = (split == SPLIT_TEST) & (chain == b"H")
idx = np.where(mask)[0]
log(f"test heavy residues: {len(idx)}")

# stream residues in blocks, encode, accumulate per-sequence sum + count of latent activations
DICT = ae.dict_size if hasattr(ae, "dict_size") else 32768
useq = np.unique(seq_id[idx])
seq2pos = {s: i for i, s in enumerate(useq)}
S = len(useq)
sum_act = th.zeros(S, DICT, device=DEV)  # sum of nonzero latent acts per seq
cnt_res = th.zeros(S, 1, device=DEV)  # residues per seq
BLK = 8192
for b0 in range(0, len(idx), BLK):
    rows = idx[b0 : b0 + BLK]
    x = th.tensor(acts[rows[0] : rows[-1] + 1].astype(np.float32), device=DEV)  # contiguous slab
    # subselect exact rows (slab may include non-test rows)
    sel = rows - rows[0]
    x = x[sel]
    # exact convention from concept_labeling.py line 143 (the validated classification encode)
    f = ae.encode(x, use_threshold=True, normalize_activations=True).detach()  # (b,dict)
    sids = seq_id[rows]
    pos = th.tensor([seq2pos[s] for s in sids], device=DEV)
    sum_act.index_add_(0, pos, f.float())
    cnt_res.index_add_(0, pos, th.ones(len(rows), 1, device=DEV))
    if b0 % (BLK * 40) == 0:
        log(f"  {b0}/{len(idx)}")
mean_act = (sum_act / cnt_res.clamp(min=1)).cpu().numpy()  # (S,dict) per-seq mean latent activation
log("encoded per-seq means", mean_act.shape)

# map useq -> IGHJ6 label via the test dataframe (seq_id indexes into the FULL df order used at extraction)
# seq_id is the global row index in oas_full_annotated; build label lookup
j6_full = (df["j_family_heavy"] == "IGHJ6").values
lab = j6_full[useq]
log(f"seqs {S}  IGHJ6 {lab.sum()} ({lab.mean()*100:.1f}%)")

pos_mean = mean_act[lab].mean(0)  # per-latent mean activation on IGHJ6 seqs
neg_mean = mean_act[~lab].mean(0)  # on non-IGHJ6
enrich = pos_mean / np.maximum(neg_mean, 1e-9)

# bring in F1 for IGHJ6 from the germline matrix
gm = pd.read_parquet(BASE + "/results/concept_f1_germline_matrix.parquet")
gm = gm[gm.config == "paired_L20"].set_index("latent_id")
f1_j6 = gm["F1_IGHJ6"].reindex(range(DICT)).values

out = {
    "n_test_seq": int(S),
    "n_j6": int(lab.sum()),
    "enrich_top20_latents": [int(i) for i in np.argsort(-enrich)[:20]],
    "enrich_top20_values": [float(enrich[i]) for i in np.argsort(-enrich)[:20]],
    "f1_of_enrich_top20": [
        float(f1_j6[i]) if not np.isnan(f1_j6[i]) else None for i in np.argsort(-enrich)[:20]
    ],
}
# also: for the known best F1 detector (latent 356), its enrichment
out["latent356_enrich"] = float(enrich[356])
out["latent356_f1"] = float(f1_j6[356]) if not np.isnan(f1_j6[356]) else None
# correlation between enrichment and F1 over active latents
active = ~np.isnan(f1_j6) & (neg_mean > 0)
from scipy.stats import spearmanr

rho, _ = spearmanr(enrich[active], f1_j6[active])
out["spearman_enrich_vs_f1"] = float(rho)
out["n_active_compared"] = int(active.sum())
json.dump(out, open(BASE + "/results/enrichment_vs_f1.json", "w"), indent=2)
np.savez(
    BASE + "/results/enrichment_j6.npz",
    enrich=enrich,
    f1_j6=f1_j6,
    pos_mean=pos_mean,
    neg_mean=neg_mean,
)
log("DONE")
log(json.dumps(out, indent=2))
