#!/usr/bin/env python3
"""Per-residue activation of each conserved-residue detector along one example heavy chain,
with the residue identities marked (Cys/Trp/Gly positions). Writes conserved_profile.json."""

import json, time, numpy as np, h5py, torch as th, pyarrow.parquet as pq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/cc_paired_L20.pt", device=dev, dtype=th.float32)
ae.eval()
# top latent per conserved concept (from conserved_residues.json)
TOPLAT = {"Cys_framework": 17937, "Trp_FR2": 5372, "Gly_FR4": 17992}
REGION_MAP = {0: "FR1", 1: "CDR1", 2: "FR2", 3: "CDR2", 4: "FR3", 5: "CDR3", 6: "FR4"}
tab = pq.read_table(f"{BASE}/data/oas_full_annotated.parquet", columns=["heavy_aa"]).to_pydict()[
    "heavy_aa"
]
h5 = f"{ACTS}/full_paired_L20.h5"
with h5py.File(h5, "r") as h:
    split = h["split"][:]
    seqid = h["seq_id"][:]
    region = h["region_label"][:]
    chain = h["chain"][:]
is_h = chain == b"H"
mask = (split == 2) & is_h
sids = np.unique(seqid[mask])
lats = sorted(set(TOPLAT.values()))
best = None
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    for sid in sids[:4000]:
        gi = np.nonzero((seqid == sid) & is_h & (split == 2))[0]
        if len(gi) < 110 or len(gi) > 140:
            continue
        rl = region[gi]
        if len(set(rl.tolist())) < 7:
            continue
        seqaa = tab[int(sid)]
        if seqaa is None or len(seqaa) < len(gi):
            continue
        aa = list(seqaa[: len(gi)])
        # need Cys, Trp, Gly present
        if "C" not in aa or "W" not in aa or "G" not in aa:
            continue
        x = acts[gi.min() : gi.max() + 1]
        take = gi - gi.min()
        xb = th.tensor(x[take].astype(np.float32), device=dev)
        with th.no_grad():
            f = ae.encode(xb, use_threshold=True, normalize_activations=True)
        prof = {k: f[:, l].cpu().numpy().tolist() for k, l in TOPLAT.items()}
        sc = sum(max(prof[k]) for k in TOPLAT)
        cand = {
            "seq_id": int(sid),
            "len": int(len(gi)),
            "aa": aa,
            "regions": [REGION_MAP[int(v)] for v in rl],
            "profiles": prof,
            "score": float(sc),
            "cys_pos": [i for i, a in enumerate(aa) if a == "C"],
            "trp_pos": [i for i, a in enumerate(aa) if a == "W"],
            "gly_pos": [i for i, a in enumerate(aa) if a == "G"],
        }
        if best is None or sc > best["score"]:
            best = cand
log(f"picked seq {best['seq_id']} len {best['len']} cys@{best['cys_pos']} trp@{best['trp_pos']}")
json.dump({"toplat": TOPLAT, "example": best}, open(f"{RES}/conserved_profile.json", "w"))
log("DONE")
print("cys", best["cys_pos"], "trp", best["trp_pos"], "gly(n)", len(best["gly_pos"]))
