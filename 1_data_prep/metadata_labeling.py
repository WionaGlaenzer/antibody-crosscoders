#!/usr/bin/env python3
"""
TASK B: label paired_L20 crosscoder latents by sequence-level metadata categories
(Isotype, BType consolidated, Disease, Vaccine) on the TEST split.
Method: aggregate each latent's activation to sequence level (mean of nonzero, per crosscoder,
using the igbert_paired encode as in the main analysis) then per-latent per-class max-F1 sweep
(InterPLM) + Haque criterion. BATCH CONFOUND: for Disease/Vaccine report top-detector positive-seq
Author purity; flag concept batch-confounded if best detector >80% from one Author.
"""

import os, sys, json, time
import numpy as np, h5py, torch as th

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

RES = f"{BASE}/results"
ACTS = f"{BASE}/activations"
CKPT = f"{BASE}/checkpoints"
CODE = f"{BASE}/code/dictionary_learning_jkminder"
sys.path.insert(0, CODE)
LABELS = f"{BASE}/data/oas_full_annotated.parquet"
MARGIN = 0.2


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def crit(pi):
    return 2.0 * pi / (1.0 + pi) + MARGIN


def consolidate_btype(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).lower()
    if "naive" in t:
        return "Naive"
    if "plasmablast" in t or "asc" in t or "plasma cell" in t or "plasma-b" in t:
        return "Plasmablast-ASC"
    if "memory" in t:
        return "Memory"
    return "Other"


def build_seq_labels(test_seq):
    import pandas as pd

    df = pd.read_parquet(LABELS, columns=["Isotype", "BType", "Disease", "Vaccine", "Author"])
    out = {}
    # Isotype: only IGHM/IGHA/IGHG/IGHD
    iso = df["Isotype"].astype("object").to_numpy()
    for cls in ["IGHM", "IGHA", "IGHG", "IGHD"]:
        out[("Isotype", cls)] = iso[test_seq] == cls
    # BType consolidated
    bt = np.array([consolidate_btype(x) for x in df["BType"].to_numpy()], dtype=object)
    for cls in ["Naive", "Memory", "Plasmablast-ASC"]:
        out[("BType", cls)] = bt[test_seq] == cls
    # Disease (drop None)
    dis = df["Disease"].astype("object").to_numpy()
    for cls in sorted(set(x for x in df["Disease"].dropna().unique() if str(x) != "None")):
        out[("Disease", cls)] = dis[test_seq] == cls
    # Vaccine (drop None)
    vac = df["Vaccine"].astype("object").to_numpy()
    for cls in sorted(set(x for x in df["Vaccine"].dropna().unique() if str(x) != "None")):
        out[("Vaccine", cls)] = vac[test_seq] == cls
    author = df["Author"].astype("object").to_numpy()[test_seq]
    return out, author


def main():
    import pandas as pd

    dev = "cuda" if th.cuda.is_available() else "cpu"
    log("device", dev)
    from dictionary_learning.dictionary import BatchTopKCrossCoder

    ae = BatchTopKCrossCoder.from_pretrained(
        f"{CKPT}/cc_paired_L20.pt", device=dev, dtype=th.float32
    )
    ae.eval()
    cls = pd.read_csv(f"{RES}/classification_paired_L20.csv")
    active_ids = cls["latent_id"].to_numpy().astype(np.int64)
    A = len(active_ids)
    category = cls["category"].to_numpy()
    is_g = np.array([str(x).startswith("gained") for x in category])
    art = cls["artifact_flag"]
    art_b = art.map({True: True, "True": True, False: False, "False": False})
    survive = np.where(is_g, ~(art_b.fillna(False).to_numpy().astype(bool)), True)
    active_t = th.tensor(active_ids, device=dev)
    with h5py.File(f"{ACTS}/full_paired_L20.h5", "r") as h:
        split = h["split"][:]
        sid = h["seq_id"][:].astype(np.int64)
        N = split.shape[0]
    test = split == 2
    test_seq = np.unique(sid[test])
    n_seq = len(test_seq)
    seq2c = -np.ones(int(sid.max()) + 1, dtype=np.int64)
    seq2c[test_seq] = np.arange(n_seq)
    log(f"active latents={A} test seqs={n_seq}")
    seqlabels, author = build_seq_labels(test_seq)
    log(f"metadata concepts={len(seqlabels)}")

    # sequence-level mean-of-nonzero aggregation (both chains combined, paired encode)
    S = th.zeros(n_seq, A, device=dev)
    C = th.zeros(n_seq, A, device=dev)
    Nt = 0
    BLK = 262144
    ENC = 8192
    t0 = time.time()
    with h5py.File(f"{ACTS}/full_paired_L20.h5", "r") as h:
        acts = h["acts"]
        for s in range(0, N, BLK):
            e = min(s + BLK, N)
            m = test[s:e]
            if not m.any():
                continue
            loc = np.nonzero(m)[0]
            blk = acts[s:e]
            sub = blk[loc]
            comp = seq2c[sid[s:e][loc]]
            comp_t = th.tensor(comp, device=dev)
            for b0 in range(0, sub.shape[0], ENC):
                b1 = min(b0 + ENC, sub.shape[0])
                x = th.tensor(sub[b0:b1].astype(np.float32), device=dev)
                with th.no_grad():
                    f = ae.encode(x, use_threshold=True, normalize_activations=True)
                code = f.index_select(1, active_t)
                cf = code.float()
                nz = (code > 0).float()
                cc = comp_t[b0:b1]
                S.index_add_(0, cc, cf)
                C.index_add_(0, cc, nz)
                Nt += b1 - b0
            if (s // BLK) % 12 == 0:
                log(f"  {Nt} {time.time()-t0:.0f}s")
    score = (S / th.clamp(C, min=1.0)).cpu().numpy()
    del S, C
    th.cuda.empty_cache()
    log(f"aggregated {Nt} residues -> {n_seq} seq vectors {time.time()-t0:.0f}s")

    # per-class max-F1 sweep, capturing best latent + its positive-seq author purity
    sc = th.tensor(score, device=dev)
    order = th.argsort(sc, 0, descending=True)
    del sc
    ns = order.shape[0]
    cumP = th.arange(1, ns + 1, device=dev, dtype=th.float32).unsqueeze(1)
    order_np = order.cpu().numpy()
    rows = []
    summary = {}
    for (cat, cls_name), y in seqlabels.items():
        P = int(y.sum())
        pi = P / ns
        if P < 15:
            continue
        yt = th.tensor(y.astype(np.float32), device=dev)
        ys = yt[order]
        cum = th.cumsum(ys, 0)
        pr = cum / cumP
        rc = cum / P
        f1 = 2 * pr * rc / th.clamp(pr + rc, min=1e-9)
        fm, fa = f1.max(0)
        bc = int(fm.argmax().item())
        tt = int(fa[bc].item())
        bestF1 = float(fm[bc])
        bestP = float(pr[tt, bc])
        bestR = float(rc[tt, bc])
        # author purity of the best detector's PREDICTED-POSITIVE seqs (top tt+1 by score)
        top_seqs = order_np[: tt + 1, bc]
        # among predicted positives that are TRUE positives, author distribution
        tp_mask = y[top_seqs]
        tp_authors = author[top_seqs][tp_mask]
        if len(tp_authors):
            vals, counts = np.unique(tp_authors.astype(str), return_counts=True)
            top_auth = str(vals[counts.argmax()])
            top_purity = float(counts.max() / counts.sum())
        else:
            top_auth = ""
            top_purity = 0.0
        confound = bool(top_purity > 0.8 and cat in ("Disease", "Vaccine"))
        rows.append(
            dict(
                category=cat,
                cls=cls_name,
                best_latent=int(active_ids[bc]),
                F1=round(bestF1, 4),
                precision=round(bestP, 4),
                recall=round(bestR, 4),
                prevalence=round(pi, 4),
                latent_category=str(category[bc]),
                survive=bool(survive[bc]),
                meets_haque=bool(bestF1 > crit(pi)),
                top_Author=top_auth,
                top_Author_purity=round(top_purity, 4),
                batch_confounded=confound,
            )
        )
        del yt, ys, cum, pr, rc, f1, fm, fa
    del order
    th.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/crosscoder_metadata_labels.csv", index=False)
    # per-category summary
    for cat in ["Isotype", "BType", "Disease", "Vaccine"]:
        sub = df[df.category == cat]
        det = sub[sub.F1 > 0.5]
        best = sub.sort_values("F1", ascending=False).iloc[0] if len(sub) else None
        confounded_classes = sub[sub.batch_confounded].cls.tolist()
        # a category is 'clean concept' if its best detector meets Haque AND not batch-confounded
        summary[cat] = dict(
            n_classes_tested=len(sub),
            n_classes_with_detector_F1_gt_0p5=int(len(det)),
            best_class=(str(best.cls) if best is not None else None),
            best_F1=(round(float(best.F1), 4) if best is not None else None),
            best_latent=(int(best.best_latent) if best is not None else None),
            best_latent_category=(str(best.latent_category) if best is not None else None),
            best_top_Author_purity=(
                round(float(best.top_Author_purity), 4) if best is not None else None
            ),
            batch_confounded_classes=confounded_classes,
            verdict=(
                "batch-confounded"
                if (
                    best is not None
                    and cat in ("Disease", "Vaccine")
                    and float(best.top_Author_purity) > 0.8
                )
                else (
                    "clean-concept"
                    if (best is not None and float(best.F1) > 0.5)
                    else "no-detector"
                )
            ),
        )
    with open(f"{RES}/crosscoder_metadata_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote crosscoder_metadata_labels.csv + crosscoder_metadata_summary.json")
    log("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
