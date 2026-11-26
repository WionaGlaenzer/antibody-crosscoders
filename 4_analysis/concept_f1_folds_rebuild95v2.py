#!/usr/bin/env python3
"""Well-powered 5-fold Swiss-Prot concept F1 on the REBUILT size-matched MIXED crosscoder
(cc_mixed_small_s0_L20), TEST general residues only (kind==1 & split==2).

Rebuilds the residue-level concept-label matrix from FRESH UniProt fetches for the general
reps ACTUALLY in exp_mixed_small_L20.h5 (2,655 general-test seqs), replacing the label-thin
old pool (which only covered general_protein_sample.parquet rows 0-39999 -> 209 labeled test
seqs). General seq_id is namespaced +100000; pool row = seq_id-100000 = index into the
accession-sorted filtered.fasta -> UniProt accession (verified by exact sequence match).

Concepts = the ORIGINAL 12 (order == concept_labels.json). Same InterPLM per-latent max-F1
sweep, per-fold, by-sequence folds, Haque floor 2*pi/(1+pi)+0.2.

Inputs staged into workdir: uniprot_annotations_general95.tsv
Reads (absolute): seqmap, h5, checkpoint, concept_labels.json (concept order check).
Writes to D/results/: concept_f1_folds_rebuild95v2.json, concept_f1_folds_rebuild95v2.npz,
concept_f1_per_concept_rebuild95v2.csv, concept_labels_general95.npz.
"""

import json, time, os, re, sys, csv
import numpy as np, h5py, torch as th

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concept_f1_core import sweep_maxf1

ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
RES = f"{BASE}/results"
dev = "cuda" if th.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


NFOLD = 5

# ---------------- concept order + feature parser (validated 30529/30530 vs old labels) ----
CONCEPTS = [
    "Compositional bias",
    "Domain [FT]",
    "Binding site",
    "Active site",
    "Transmembrane",
    "Signal peptide",
    "Coiled coil",
    "Disulfide bond",
    "Region",
    "Helix",
    "Beta strand",
    "Turn",
]
KEYWORD = [
    "COMPBIAS",
    "DOMAIN",
    "BINDING",
    "ACT_SITE",
    "TRANSMEM",
    "SIGNAL",
    "COILED",
    "DISULFID",
    "REGION",
    "HELIX",
    "STRAND",
    "TURN",
]
NC = len(CONCEPTS)
# confirm concept order matches the canonical concept_labels.json
CL = json.load(open(f"{RES}/concept_labels.json"))
assert CL["concepts"] == CONCEPTS, ("concept order mismatch", CL["concepts"])


def _num(t):
    return int(t.strip().lstrip("<"))


def parse_cell(cell, ci):
    kw = KEYWORD[ci]
    pos = set()
    for m in re.finditer(re.escape(kw) + r"\s+([^;]+)", cell):
        loc = m.group(1).strip()
        if "?" in loc or ">" in loc:
            continue
        loc = loc.replace("<", "")
        if ".." in loc:
            parts = loc.split("..")
            if len(parts) != 2:
                continue
            try:
                a = _num(parts[0])
                b = _num(parts[1])
            except ValueError:
                continue
            if ci == 7:
                pos.add(a)
                pos.add(b)  # Disulfide: endpoints only
            else:
                if b >= a:
                    pos.update(range(a, b + 1))
                else:
                    pos.add(a)
                    pos.add(b)
        else:
            try:
                pos.add(_num(loc))
            except ValueError:
                continue
    return pos


# ---------------- 1. parse fetched UniProt TSV -> acc -> {ci: sorted[pos1idx]} -------------
TSV = "uniprot_annotations_general95.tsv"  # staged into workdir
labels_by_acc = {}
with open(TSV) as fh:
    hdr = next(fh).rstrip("\n").split("\t")
    assert hdr[1:] == CONCEPTS, ("TSV header mismatch", hdr)
    for line in fh:
        p = line.rstrip("\n").split("\t")
        if not p or not p[0]:
            continue
        acc = p[0]
        cols = (p[1:] + [""] * NC)[:NC]
        d = {}
        for ci in range(NC):
            ps = parse_cell(cols[ci], ci)
            if ps:
                d[ci] = sorted(ps)
        if d:
            labels_by_acc[acc] = d
log(f"accs parsed with >=1 label: {len(labels_by_acc)}")

# ---------------- 2. seqmap: seq_id -> pool_id, accession -----------------------------------
sm = json.load(open(f"{RES}/general95_test_seqmap.json"))
sid_list = sm["seq_id"]
pid_list = sm["pool_id"]
acc_list = sm["accession"]
# row_labels keyed by pool_row (== seq_id-100000), positions 0-indexed
row_labels = {}
for sid, pid, acc in zip(sid_list, pid_list, acc_list):
    d = labels_by_acc.get(acc)
    if not d:
        continue
    row_labels[int(pid)] = {int(ci): set(p - 1 for p in poss) for ci, poss in d.items()}
log(f"general-test seqs with >=1 labeled residue (pre-store-clip): {len(row_labels)}")

# ---------------- 3. load crosscoder + store metadata ---------------------------------------
sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")
from dictionary_learning.dictionary import BatchTopKCrossCoder

ae = BatchTopKCrossCoder.from_pretrained(
    f"{CKPT}/cc_mixed_small_s0_L20.pt", device=dev, dtype=th.float32
)
ae.eval()
DICT = ae.dict_size if hasattr(ae, "dict_size") else 32768
h5 = f"{ACTS}/exp_mixed_small_L20.h5"
with h5py.File(h5, "r") as h:
    sid = h["seq_id"][:].astype(np.int64)
    split = h["split"][:]
    kind = h["kind"][:]
    N = sid.shape[0]
gen_test = (kind == 1) & (split == 2)
prow = sid - 100000
test_seqs = np.unique(prow[gen_test])
rng = np.random.default_rng(0)
perm = rng.permutation(len(test_seqs))
seqfold_arr = (perm % NFOLD).astype(np.int8)
fold_of = np.full(N, -1, dtype=np.int8)
pos = np.searchsorted(test_seqs, prow)
pos = np.clip(pos, 0, len(test_seqs) - 1)
is_gt = gen_test & (test_seqs[pos] == prow)
fold_of[is_gt] = seqfold_arr[pos[is_gt]]

# ---------------- 4. per-residue concept bitmask (position = offset in contiguous block) -----
u, first, cnt = np.unique(sid, return_index=True, return_counts=True)
cmask = np.zeros(N, dtype=np.int32)
# per-seq clip stats
n_lab_test_seqs = 0
clipped = 0
for f0, c, s in zip(first, cnt, u):
    rl = row_labels.get(int(s) - 100000)
    if not rl:
        continue
    hit = False
    for ci, poss in rl.items():
        bit = 1 << ci
        for p in poss:
            if 0 <= p < c:
                cmask[f0 + p] |= bit
                hit = True
            elif p >= c:
                clipped += 1
    if hit and gen_test[f0]:
        n_lab_test_seqs += 1
log(
    f"labeled general-test seqs with >=1 in-store residue: {n_lab_test_seqs} | clipped positions: {clipped}"
)

# ---------------- 4b. deliverable: per general-test-residue concept boolean matrix -----------
gt_idx = np.nonzero(gen_test)[0]
gt_labels = np.zeros((len(gt_idx), NC), dtype=bool)
bits = np.array([1 << i for i in range(NC)], dtype=np.int32)
cm_gt = cmask[gt_idx]
for i in range(NC):
    gt_labels[:, i] = ((cm_gt >> i) & 1).astype(bool)
labeled_res_per_concept = {CONCEPTS[i]: int(gt_labels[:, i].sum()) for i in range(NC)}
seqs_with_label_per_concept = {}
gt_sid = sid[gt_idx]
for i in range(NC):
    sids_i = np.unique(gt_sid[gt_labels[:, i]])
    seqs_with_label_per_concept[CONCEPTS[i]] = int(len(sids_i))
n_test_seqs_with_any = int(len(np.unique(gt_sid[gt_labels.any(1)])))
np.savez(
    f"{RES}/concept_labels_general95.npz",
    gt_residue_index=gt_idx.astype(np.int64),
    gt_seq_id=gt_sid.astype(np.int64),
    labels=gt_labels,
    concepts=np.array(CONCEPTS),
)
log(
    f"n general-test residues={len(gt_idx)} | seqs with >=1 label(any concept)={n_test_seqs_with_any}"
)

# ---------------- 5. per-fold positives + Haque floor ---------------------------------------
Pc_fold = np.zeros((NFOLD, NC), dtype=np.int64)
Nfold = np.zeros(NFOLD, dtype=np.int64)
for fk in range(NFOLD):
    fm = fold_of == fk
    Nfold[fk] = int(fm.sum())
    cmf = cmask[fm]
    Pc_fold[fk] = [(int(((cmf >> i) & 1).sum())) for i in range(NC)]
prev_fold = Pc_fold / np.maximum(Nfold[:, None], 1)
floor_fold = 2 * prev_fold / (1 + prev_fold) + 0.2
log(f"gen-test residues {int(gen_test.sum())} | fold sizes {Nfold.tolist()}")

# ---------------- 6. encode TEST general residues, collect nonzeros --------------------------
cmask_t = th.tensor(cmask, device=dev)
foldt = th.tensor(fold_of, device=dev)
BLK = 200000
ENC = 16384
lat_buf = []
act_buf = []
cm_buf = []
fd_buf = []
t0 = time.time()
tot = 0
with h5py.File(h5, "r") as h:
    acts = h["acts"]
    for s in range(0, N, BLK):
        e = min(s + BLK, N)
        x = acts[s:e]
        fblk = fold_of[s:e]
        for b0 in range(0, e - s, ENC):
            b1 = min(b0 + ENC, e - s)
            keep = np.nonzero(fblk[b0:b1] >= 0)[0]
            if len(keep) == 0:
                continue
            xb = th.tensor(x[b0:b1][keep].astype(np.float32), device=dev)
            with th.no_grad():
                f = ae.encode(xb, use_threshold=True, normalize_activations=True)
            r_i, l_i = f.nonzero(as_tuple=True)
            a_v = f[r_i, l_i]
            gi = th.tensor(s + b0 + keep, device=dev)[r_i]
            lat_buf.append(l_i.to(th.int32).cpu().numpy())
            act_buf.append(a_v.cpu().numpy())
            cm_buf.append(cmask_t[gi].cpu().numpy())
            fd_buf.append(foldt[gi].cpu().numpy())
            tot += len(r_i)
        if (s // BLK) % 20 == 0:
            log(f"  encode {e}/{N} nz={tot} {time.time()-t0:.0f}s")
lat = np.concatenate(lat_buf)
act = np.concatenate(act_buf)
cm = np.concatenate(cm_buf)
fd = np.concatenate(fd_buf)
del lat_buf, act_buf, cm_buf, fd_buf
active = np.unique(lat)
log(f"nz={len(lat)} active={len(active)}")

# ---------------- 7. per-fold per-latent per-concept max-F1 sweep ----------------------------
maxf1 = np.zeros((NFOLD, DICT, NC), dtype=np.float32)
for fk in range(NFOLD):
    sel = fd == fk
    maxf1[fk], _ = sweep_maxf1(lat[sel], act[sel], cm[sel], DICT, NC, Pc_fold[fk], bits=bits)
    log(f"fold {fk} swept {time.time()-t0:.0f}s")

# ---------------- 8. per-concept detection stability ----------------------------------------
best_per_fold = np.zeros((NFOLD, NC))
detect_folds = np.zeros(NC, dtype=int)
ndet_per_fold = np.zeros((NFOLD, NC), dtype=int)
for fk in range(NFOLD):
    for ci in range(NC):
        col = maxf1[fk, active, ci]
        best_per_fold[fk, ci] = col.max()
        ndet_per_fold[fk, ci] = int((col > floor_fold[fk, ci]).sum())
for ci in range(NC):
    detect_folds[ci] = int(sum(best_per_fold[fk, ci] > floor_fold[fk, ci] for fk in range(NFOLD)))

summ = {
    "concepts": CONCEPTS,
    "n_folds": NFOLD,
    "fold_sizes": Nfold.tolist(),
    "checkpoint": "cc_mixed_small_s0_L20.pt",
    "store": "exp_mixed_small_L20.h5",
    "label_source": "fresh UniProt fetch (general_pool_95nr reps in mixed store), 2655 general-test seqs",
    "n_gen_test_seqs": int(len(test_seqs)),
    "n_labeled_test_seqs": int(n_test_seqs_with_any),
    "n_labeled_test_seqs_OLD": 209,
    "labeled_residues_per_concept": labeled_res_per_concept,
    "seqs_with_label_per_concept": seqs_with_label_per_concept,
    "concept_positives_total": {CONCEPTS[i]: int(Pc_fold[:, i].sum()) for i in range(NC)},
    "best_f1_mean": {CONCEPTS[i]: float(best_per_fold[:, i].mean()) for i in range(NC)},
    "best_f1_sd": {CONCEPTS[i]: float(best_per_fold[:, i].std(ddof=1)) for i in range(NC)},
    "floor_mean": {CONCEPTS[i]: float(floor_fold[:, i].mean()) for i in range(NC)},
    "detect_in_k_of_5": {CONCEPTS[i]: int(detect_folds[i]) for i in range(NC)},
    "ndet_mean": {CONCEPTS[i]: float(ndet_per_fold[:, i].mean()) for i in range(NC)},
    "ndet_sd": {CONCEPTS[i]: float(ndet_per_fold[:, i].std(ddof=1)) for i in range(NC)},
    "n_concepts_detected_ge3of5": int(sum(1 for ci in range(NC) if detect_folds[ci] >= 3)),
}
np.savez(
    f"{RES}/concept_f1_folds_rebuild95v2.npz",
    maxf1=maxf1.astype(np.float32),
    active=active,
    floor_fold=floor_fold.astype(np.float32),
    best_per_fold=best_per_fold,
    ndet_per_fold=ndet_per_fold,
    concepts=np.array(CONCEPTS),
)
json.dump(summ, open(f"{RES}/concept_f1_folds_rebuild95v2.json", "w"), indent=1)
# per-concept CSV
with open(f"{RES}/concept_f1_per_concept_rebuild95v2.csv", "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(
        [
            "concept",
            "best_f1_mean",
            "best_f1_sd",
            "floor_mean",
            "detect_in_k_of_5",
            "labeled_residues",
            "seqs_with_label",
            "concept_positives_total_5fold",
        ]
    )
    for ci in range(NC):
        cn = CONCEPTS[ci]
        w.writerow(
            [
                cn,
                f"{best_per_fold[:,ci].mean():.4f}",
                f"{best_per_fold[:,ci].std(ddof=1):.4f}",
                f"{floor_fold[:,ci].mean():.4f}",
                int(detect_folds[ci]),
                labeled_res_per_concept[cn],
                seqs_with_label_per_concept[cn],
                int(Pc_fold[:, ci].sum()),
            ]
        )
log("DONE")
print("SUMMARY_JSON_START")
print(json.dumps(summ, indent=1))
