#!/usr/bin/env python3
"""
TASK A: label the 3 per-model squared-fix SAEs (dict 16384, k=32, layer 20) by germline V/J
family and IMGT region, on the TEST split. Question: does ProtBert's OWN single-model SAE already
contain germline V/J detectors (F1>0.5)? Do the antibody SAEs (unp, pair) have MORE?
Method: per-latent per-family max-F1 threshold sweep (InterPLM); Haque 2026 detection criterion.
SAE encode uses TRUE batch-top-k: encode(xn, use_threshold=False, normalize_activations=False)
with xn manually normalized by h5 per-model train_mean/train_std[col].
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
REGION_MAP = {"FR1": 0, "CDR1": 1, "FR2": 2, "CDR2": 3, "FR3": 4, "CDR3": 5, "FR4": 6}
REGION_NAMES = [k for k, _ in sorted(REGION_MAP.items(), key=lambda kv: kv[1])]
MARGIN = 0.2
SAES = [
    ("prot", 0, "sae_prot_L20_sq.pt"),
    ("unp", 1, "sae_unp_L20_sq.pt"),
    ("pair", 2, "sae_pair_L20_sq.pt"),
]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def crit(pi):
    return 2.0 * pi / (1.0 + pi) + MARGIN


def load_sae(fn, dev):
    from dictionary_learning.dictionary import BatchTopKSAE

    sae = BatchTopKSAE(activation_dim=1024, dict_size=16384, k=32)
    sd = th.load(f"{CKPT}/{fn}", map_location=dev)
    sae.load_state_dict(sd, strict=False)
    sae.to(dev).eval()
    return sae


def build_germline_labels():
    import pandas as pd

    df = pd.read_parquet(
        LABELS, columns=["v_family_heavy", "j_call_heavy", "v_family_light", "j_call_light"]
    )
    labels = {}

    def add(col, chain):
        arr = df[col].astype("object").to_numpy()
        for val in sorted(set([x for x in df[col].dropna().unique()])):
            labels[f"{col}::{val}"] = (arr, chain)

    add("v_family_heavy", "H")
    add("j_call_heavy", "H")
    add("v_family_light", "L")
    add("j_call_light", "L")
    return labels


def main():
    import pandas as pd

    dev = "cuda" if th.cuda.is_available() else "cpu"
    log("device", dev)
    gl = build_germline_labels()
    log("germline concepts", len(gl))
    with h5py.File(f"{ACTS}/full_paired_L20.h5", "r") as h:
        split = h["split"][:]
        region = h["region_label"][:]
        chain = h["chain"][:]
        sid = h["seq_id"][:].astype(np.int64)
        tmean = h["train_mean"][:]
        tstd = h["train_std"][:]
        N = split.shape[0]
    test = split == 2
    test_seq = np.unique(sid[test])
    n_seq = len(test_seq)
    seq2c = -np.ones(int(sid.max()) + 1, dtype=np.int64)
    seq2c[test_seq] = np.arange(n_seq)
    log(f"test residues={int(test.sum())} test seqs={n_seq}")
    A = 16384
    tm = th.tensor(tmean, device=dev)
    ts = th.tensor(tstd, device=dev)

    # germline membership per family (indexed by compact test seq)
    memb = {}
    chain_of = {}
    for name, (arr, ch) in gl.items():
        col, target = name.split("::")
        v = arr[test_seq] == target
        memb[name] = v
        chain_of[name] = ch

    all_rows = []
    summary = {}
    for tag, col, fn in SAES:
        log("==== SAE", tag, "====")
        sae = load_sae(fn, dev)
        fire_count = th.zeros(A, dtype=th.float64, device=dev)
        far = th.zeros(A, 7, dtype=th.float64, device=dev)
        rt = th.zeros(7, dtype=th.float64, device=dev)
        sH = th.zeros(n_seq, A, device=dev)
        cH = th.zeros(n_seq, A, device=dev)
        sL = th.zeros(n_seq, A, device=dev)
        cL = th.zeros(n_seq, A, device=dev)
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
                reg = region[s:e][loc]
                ch = chain[s:e][loc]
                sd2 = sid[s:e][loc]
                comp = seq2c[sd2]
                reg_t = th.tensor(reg.astype(np.int64), device=dev)
                comp_t = th.tensor(comp, device=dev)
                isH = th.tensor(ch == b"H", device=dev)
                for b0 in range(0, sub.shape[0], ENC):
                    b1 = min(b0 + ENC, sub.shape[0])
                    x = th.tensor(sub[b0:b1, col, :].astype(np.float32), device=dev)
                    xn = (x - tm[col]) / ts[col]
                    with th.no_grad():
                        f = sae.encode(
                            xn, use_threshold=False, normalize_activations=False
                        )  # (b,16384) true top-k
                    fire = (f > 0).to(th.float64)
                    fire_count += fire.sum(0)
                    oh = th.zeros(b1 - b0, 7, dtype=th.float64, device=dev)
                    oh.scatter_(1, reg_t[b0:b1].unsqueeze(1), 1.0)
                    rt += oh.sum(0)
                    far += fire.T @ oh
                    cf = f.float()
                    nz = (f > 0).float()
                    cc = comp_t[b0:b1]
                    hm = isH[b0:b1]
                    if hm.any():
                        sH.index_add_(0, cc[hm], cf[hm])
                        cH.index_add_(0, cc[hm], nz[hm])
                    lm = ~hm
                    if lm.any():
                        sL.index_add_(0, cc[lm], cf[lm])
                        cL.index_add_(0, cc[lm], nz[lm])
                    Nt += b1 - b0
                if (s // BLK) % 12 == 0:
                    log(f"  {tag} {Nt} {time.time()-t0:.0f}s")
        log(f"{tag} encode done Nt={Nt} {time.time()-t0:.0f}s median-fire/latent-check")
        # sparsity check
        med_freq = float(np.median((fire_count.cpu().numpy() / max(Nt, 1))))
        log(f"{tag} median latent fire-frac={med_freq:.5f} (expect ~{32/16384:.5f})")
        # region F1 (fixed binary)
        fc = fire_count.cpu().numpy()
        farr = far.cpu().numpy()
        rtn = rt.cpu().numpy()
        Ntot = float(Nt)
        prec = farr / np.maximum(fc[:, None], 1e-9)
        rec = farr / np.maximum(rtn[None, :], 1e-9)
        f1_reg = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        pi_reg = rtn / Ntot
        best_reg_f1 = f1_reg.max(1)
        n_region_gt5 = int((best_reg_f1 > 0.5).sum())
        n_region_haque = int((f1_reg > crit(pi_reg)[None, :]).any(1).sum())
        # germline max-F1 sweep
        scoreH = (sH / th.clamp(cH, min=1.0)).cpu().numpy()
        scoreL = (sL / th.clamp(cL, min=1.0)).cpu().numpy()
        del sH, cH, sL, cL
        th.cuda.empty_cache()

        def sweep(score_np, members, min_pos=15):
            sc = th.tensor(score_np, device=dev)
            order = th.argsort(sc, 0, descending=True)
            del sc
            out = {}
            pis = {}
            best = {}
            ns = order.shape[0]
            cumP = th.arange(1, ns + 1, device=dev, dtype=th.float32).unsqueeze(1)
            for name, y in members.items():
                P = int(y.sum())
                pi = P / ns
                pis[name] = pi
                if P < min_pos:
                    out[name] = None
                    continue
                yt = th.tensor(y.astype(np.float32), device=dev)
                ys = yt[order]
                cum = th.cumsum(ys, 0)
                pr = cum / cumP
                rc = cum / P
                f1 = 2 * pr * rc / th.clamp(pr + rc, min=1e-9)
                fm, fa = f1.max(0)
                out[name] = fm.cpu().numpy()
                bc = int(fm.argmax().item())
                tt = int(fa[bc].item())
                best[name] = dict(
                    latent=bc,
                    F1=float(fm[bc]),
                    precision=float(pr[tt, bc]),
                    recall=float(rc[tt, bc]),
                )
                del yt, ys, cum, pr, rc, f1, fm, fa
            del order
            th.cuda.empty_cache()
            return out, pis, best

        mH = {n: memb[n] for n in memb if chain_of[n] == "H"}
        mL = {n: memb[n] for n in memb if chain_of[n] == "L"}
        f1H, piH, bH = sweep(scoreH, mH)
        f1L, piL, bL = sweep(scoreL, mL)
        f1a = {**f1H, **f1L}
        pia = {**piH, **piL}
        ba = {**bH, **bL}
        gnames = [n for n in f1a if f1a[n] is not None]

        # per-latent best V / best J
        def isV(fs):
            return len(fs) > 3 and fs[3] == "V"

        def isJ(fs):
            return len(fs) > 3 and fs[3] == "J"

        # per-latent detection: (a) naive F1>0.5 (prevalence-confounded), (b) Haque criterion (prevalence-corrected)
        bestV = np.zeros(A)
        bestJ = np.zeros(A)  # naive best-F1
        detV_h = np.zeros(A, bool)
        detJ_h = np.zeros(A, bool)  # Haque-criterion detection (any family)
        for n in gnames:
            fs = n.split("::")[1]
            cth = crit(pia[n])
            dh = f1a[n] > cth
            if isV(fs):
                bestV = np.maximum(bestV, f1a[n])
                detV_h |= dh
            elif isJ(fs):
                bestJ = np.maximum(bestJ, f1a[n])
                detJ_h |= dh
        n_det_V = int((bestV > 0.5).sum())
        n_det_J = int((bestJ > 0.5).sum())  # confounded
        n_germ_det = int(((bestV > 0.5) | (bestJ > 0.5)).sum())  # confounded
        n_det_V_h = int(detV_h.sum())
        n_det_J_h = int(detJ_h.sum())  # prevalence-corrected
        n_germ_det_h = int((detV_h | detJ_h).sum())
        # per-family rows + counts
        for n in gnames:
            fs = n.split("::")[1]
            b = ba[n]
            f1v = f1a[n]
            all_rows.append(
                dict(
                    config=tag,
                    family=fs,
                    best_latent=int(b["latent"]),
                    F1=round(b["F1"], 4),
                    precision=round(b["precision"], 4),
                    recall=round(b["recall"], 4),
                    prevalence=round(pia[n], 4),
                    n_latents_F1_gt_0p5=int((f1v > 0.5).sum()),
                    meets_haque=bool(b["F1"] > crit(pia[n])),
                )
            )
        # summary
        ighv = [n for n in gnames if n.split("::")[1].startswith("IGHV")]
        ighj6 = [n for n in gnames if n.split("::")[1] == "IGHJ6"]
        best_ighv = max((ba[n]["F1"] for n in ighv), default=0.0)
        best_ighj6 = ba[ighj6[0]]["F1"] if ighj6 else 0.0
        summary[tag] = dict(
            col=col,
            n_test_residues=Nt,
            median_fire_frac=med_freq,
            n_germline_detectors_F1gt0p5_confounded=n_germ_det,
            n_detectors_V_F1gt0p5_confounded=n_det_V,
            n_detectors_J_F1gt0p5_confounded=n_det_J,
            n_germline_detectors_haque=n_germ_det_h,
            n_detectors_V_haque=n_det_V_h,
            n_detectors_J_haque=n_det_J_h,
            n_region_detectors_F1_gt_0p5=n_region_gt5,
            n_region_detectors_haque=n_region_haque,
            best_IGHV_F1=round(float(best_ighv), 4),
            best_IGHJ6_F1=round(float(best_ighj6), 4),
            region_best_F1_max=round(float(best_reg_f1.max()), 4),
        )
        log(f"{tag} SUMMARY {json.dumps(summary[tag])}")
        del sae
        th.cuda.empty_cache()

    pd.DataFrame(all_rows).to_csv(f"{RES}/sae_concept_germline.csv", index=False)
    with open(f"{RES}/sae_concept_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote sae_concept_germline.csv + sae_concept_summary.json")
    # head-to-head print
    log(
        "HEAD-TO-HEAD germline detectors (Haque criterion): "
        + " | ".join(
            f"{t}: V={summary[t]['n_detectors_V_haque']} J={summary[t]['n_detectors_J_haque']} "
            f"IGHJ6={summary[t]['best_IGHJ6_F1']} IGHV={summary[t]['best_IGHV_F1']}"
            for t, _, _ in SAES
        )
    )


if __name__ == "__main__":
    main()
