#!/usr/bin/env python
"""
Classify-only driver: load a TRAINED crosscoder checkpoint, run the decoder-norm
classification + Minder et al. (2504.02922) Latent Scaling diagnostic on the FULL
val split, and write classification_{tag}.csv / summary_{tag}.json / decnorm_{tag}.npz
to an absolute output dir. No training. Reuses the exact analysis math from
train_crosscoder.py.

STALE-CHECKPOINT GUARD: refuses to run on a smoke / undertrained checkpoint.
A checkpoint is accepted only if BOTH hold:
  (1) its progress sidecar (cc_{tag}_progress.json) reports step >= MIN_STEPS, AND
  (2) the resulting n_active < dict_size  (a 400-step smoke has ALL latents active).
"""

import argparse, os, json, time, csv
import numpy as np
import h5py
import torch as th

MIN_STEPS = 30000


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_val_memmap(h5path, tmpdir, tag):
    f = h5py.File(h5path, "r")
    split = f["split"][:]
    idx = np.nonzero(split == 1)[0]  # val split == 1
    n = idx.size
    acts = f["acts"]
    N = acts.shape[0]
    out_path = os.path.join(tmpdir, f"{tag}_val.f16")
    mm = np.memmap(out_path, dtype=np.float16, mode="w+", shape=(n, 3, 1024))
    BLK = 262144
    wp = 0
    for start in range(0, N, BLK):
        stop = min(start + BLK, N)
        bm = split[start:stop] == 1
        k = int(bm.sum())
        if k == 0:
            continue
        mm[wp : wp + k] = acts[start:stop][bm]
        wp += k
    assert wp == n, (wp, n)
    mm.flush()
    tm = f["train_mean"][:]
    ts = f["train_std"][:]
    f.close()
    return n, mm, tm, ts


def classify_decoder_norms(Wdec):
    n = th.linalg.norm(Wdec, dim=2).transpose(0, 1)  # (dict,3): [prot,unp,pair]
    s = n.sum(dim=1, keepdim=True)
    rel = n / (s + 1e-12)
    rp, ru, rpr = rel[:, 0], rel[:, 1], rel[:, 2]
    HI, LO = 0.5, 0.2
    cat = th.full((rel.shape[0],), 5, dtype=th.int64)
    shared = (rp > LO) & (ru > LO) & (rpr > LO)
    base_only = (rp > HI) & (ru < LO) & (rpr < LO)
    gain_unp = (rp < LO) & (ru > LO) & (rpr > LO)
    gain_pair = (rp < LO) & (ru < LO) & (rpr > HI)
    transient = (ru > HI) & (rpr < LO)
    cat[shared] = 0
    cat[base_only] = 1
    cat[gain_unp] = 2
    cat[transient] = 4
    cat[gain_pair] = 3
    return n, rel, cat


CAT_NAMES = {
    0: "shared-all",
    1: "base-only",
    2: "gained-at-unpaired",
    3: "gained-at-paired",
    4: "transient",
    5: "other",
    -1: "dead",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jkminder_repo", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--progress", default=None, help="progress sidecar json for stale guard")
    ap.add_argument("--dict_size", type=int, default=32768)
    ap.add_argument("--k", type=int, default=32)
    args = ap.parse_args()

    # ---- stale-checkpoint guard (1): progress sidecar step >= MIN_STEPS ----
    prog_step = None
    if args.progress and os.path.exists(args.progress):
        p = json.load(open(args.progress))
        prog_step = p.get("step")
        assert (
            prog_step is None or prog_step >= MIN_STEPS - 1
        ), f"STALE CHECKPOINT: progress step {prog_step} < {MIN_STEPS} — refusing (smoke/undertrained)."
        log(f"stale-guard(1) OK: progress step {prog_step} >= {MIN_STEPS}")
    else:
        log("WARN: no progress sidecar; relying on guard (2) n_active<dict_size")

    import sys

    sys.path.insert(0, args.jkminder_repo)
    from dictionary_learning import BatchTopKCrossCoder

    dev = "cuda" if th.cuda.is_available() else "cpu"
    tag = args.tag
    tmp = os.environ.get("TMPDIR", "/tmp")
    log("loading val split from", args.h5)
    n_val, val_mm, tm, ts = build_val_memmap(args.h5, tmp, tag)
    log(f"val {n_val} rows")

    ae = BatchTopKCrossCoder.from_pretrained(args.ckpt, device=dev, code_normalization="CROSSCODER")
    ae.eval()
    # sanity: checkpoint dims + k
    assert ae.dict_size == args.dict_size, (ae.dict_size, args.dict_size)
    log(f"loaded {args.ckpt} dict={ae.dict_size} k={int(ae.k.item())}")

    dict_size = args.dict_size
    Wdec = ae.decoder.weight.detach()
    dnorm, rel, cat = classify_decoder_norms(Wdec)

    ABS = 8192
    n_active_counts = th.zeros(dict_size, device=dev)
    fv_e = th.zeros(3, device=dev)
    fv_var = th.zeros(3, device=dev)
    fv_l0 = 0.0
    with th.no_grad():
        for s in range(0, n_val, ABS):
            xb = th.from_numpy(np.asarray(val_mm[s : s + ABS], dtype=np.float32)).to(dev)
            xn = ae.normalize_activations(xb, inplace=False)
            f = ae.encode(xn, normalize_activations=False, use_threshold=False)
            xhat = ae.decode(f, denormalize_activations=False)
            e = xn - xhat
            fv_e += e.pow(2).sum(-1).sum(0)
            fv_var += (xn - xn.mean(0, keepdim=True)).pow(2).sum(-1).sum(0)
            fv_l0 += (f != 0).float().sum(-1).sum().item()
            n_active_counts += (f > 0).float().sum(0)
    val_act_freq = (n_active_counts / n_val).cpu().numpy()
    active_mask = val_act_freq > 1e-4
    full_ve = (1.0 - fv_e / (fv_var + 1e-12)).tolist()
    full_mse = (fv_e / n_val).tolist()
    full_l0 = fv_l0 / n_val
    log(f"FULL val: VE[p/u/pr] {full_ve[0]:.3f}/{full_ve[1]:.3f}/{full_ve[2]:.3f} L0 {full_l0:.1f}")

    # ---- stale-checkpoint guard (2): n_active < dict_size ----
    n_active = int(active_mask.sum())
    assert (
        n_active < dict_size
    ), f"STALE CHECKPOINT: n_active={n_active} == dict_size {dict_size} (all latents active = 400-step smoke). Refusing."
    log(f"stale-guard(2) OK: n_active {n_active} < dict_size {dict_size}")

    cat_np = cat.cpu().numpy().copy()
    cat_np[~active_mask] = -1
    gained_idx = np.nonzero(((cat_np == 2) | (cat_np == 3)))[0]
    log(
        f"active latents {n_active} | gained {gained_idx.size} (unp {(cat_np==2).sum()}, pair {(cat_np==3).sum()})"
    )

    # ---- Latent Scaling (identical to train_crosscoder.py) ----
    G = gained_idx.size
    betas = None
    if G > 0:
        gidx_t = th.tensor(gained_idx, device=dev, dtype=th.long)
        Dg = Wdec[:, gidx_t, :]
        Dg_norm = th.linalg.norm(Dg, dim=2)
        Dg_hat = Dg / (Dg_norm.unsqueeze(2) + 1e-12)
        num_r = th.zeros(3, G, device=dev)
        num_e = th.zeros(3, G, device=dev)
        den = th.zeros(G, device=dev)
        with th.no_grad():
            for s in range(0, n_val, ABS):
                xb = th.from_numpy(np.asarray(val_mm[s : s + ABS], dtype=np.float32)).to(dev)
                xn = ae.normalize_activations(xb, inplace=False)
                f = ae.encode(xn, normalize_activations=False, use_threshold=False)
                xhat = ae.decode(f, denormalize_activations=False)
                fg = f[:, gidx_t]
                den += (fg * fg).sum(0)
                err = xn - xhat
                for m in range(3):
                    Dh = Dg_hat[m]
                    proj_recon = xhat[:, m, :] @ Dh.t()
                    num_r[m] += (proj_recon * fg).sum(0)
                    proj_err = err[:, m, :] @ Dh.t()
                    proj_err = proj_err + fg * Dg_norm[m].unsqueeze(0)
                    num_e[m] += (proj_err * fg).sum(0)
        beta_r = (num_r / (den.unsqueeze(0) + 1e-12)).cpu().numpy()
        beta_e = (num_e / (den.unsqueeze(0) + 1e-12)).cpu().numpy()
        target_m = np.where(cat_np[gained_idx] == 3, 2, 1)
        eps = 1e-9
        br_base = beta_r[0]
        be_base = beta_e[0]
        br_tgt = beta_r[target_m, np.arange(G)]
        be_tgt = beta_e[target_m, np.arange(G)]
        nu_r = np.clip(br_base / (np.abs(br_tgt) + eps), -1, 5)
        nu_e = np.clip(be_base / (np.abs(be_tgt) + eps), -1, 5)
        artifact_flag = (nu_r >= 0.5) | (nu_e >= 0.5)
        betas = dict(
            gained_idx=gained_idx,
            beta_r=beta_r,
            beta_e=beta_e,
            nu_r=nu_r,
            nu_e=nu_e,
            target_m=target_m,
            artifact_flag=artifact_flag,
        )
        log(
            f"latent scaling: gained {G}, flagged {int(artifact_flag.sum())}, survive {int((~artifact_flag).sum())}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    act_idx = np.nonzero(active_mask)[0]
    nu_r_map = {}
    nu_e_map = {}
    flag_map = {}
    if betas is not None:
        for kk, j in enumerate(betas["gained_idx"]):
            nu_r_map[int(j)] = float(betas["nu_r"][kk])
            nu_e_map[int(j)] = float(betas["nu_e"][kk])
            flag_map[int(j)] = bool(betas["artifact_flag"][kk])
    dnorm_np = dnorm.cpu().numpy()
    rel_np = rel.cpu().numpy()
    csv_path = os.path.join(args.out_dir, f"classification_{tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "latent_id",
                "n_prot",
                "n_unp",
                "n_pair",
                "rel_prot",
                "rel_unp",
                "rel_pair",
                "category",
                "val_act_freq",
                "nu_r",
                "nu_e",
                "artifact_flag",
            ]
        )
        for j in act_idx:
            j = int(j)
            w.writerow(
                [
                    j,
                    f"{dnorm_np[j,0]:.6f}",
                    f"{dnorm_np[j,1]:.6f}",
                    f"{dnorm_np[j,2]:.6f}",
                    f"{rel_np[j,0]:.6f}",
                    f"{rel_np[j,1]:.6f}",
                    f"{rel_np[j,2]:.6f}",
                    CAT_NAMES[int(cat_np[j])],
                    f"{val_act_freq[j]:.3e}",
                    (f"{nu_r_map[j]:.4f}" if j in nu_r_map else ""),
                    (f"{nu_e_map[j]:.4f}" if j in nu_e_map else ""),
                    (str(flag_map[j]) if j in flag_map else ""),
                ]
            )
    log("wrote", csv_path, f"({act_idx.size} active)")
    np.savez(
        os.path.join(args.out_dir, f"decnorm_{tag}.npz"),
        rel=rel_np,
        dnorm=dnorm_np,
        cat=cat_np,
        val_act_freq=val_act_freq,
    )

    counts = {CAT_NAMES[c]: int((cat_np == c).sum()) for c in [-1, 0, 1, 2, 3, 4, 5]}
    # derive regime/layer from tag (e.g. "single_L12" -> ("single",12), "paired_L20" -> ("paired",20))
    import re

    m = re.match(r"(.+)_L(\d+)$", tag)
    reg = m.group(1) if m else args.__dict__.get("regime", tag)
    lay = int(m.group(2)) if m else -1
    summary = dict(
        tag=tag,
        regime=reg,
        layer=lay,
        dict_size=dict_size,
        k=args.k,
        steps=int(prog_step) + 1 if prog_step else None,
        n_val=n_val,
        reclassified=True,
        source_ckpt=args.ckpt,
        n_active=n_active,
        category_counts=counts,
        gained_total=int(G),
        gained_at_paired=int((cat_np == 3).sum()),
        gained_at_unpaired=int((cat_np == 2).sum()),
        final_var_expl={
            "prot_bert": full_ve[0],
            "igbert_unpaired": full_ve[1],
            "igbert_paired": full_ve[2],
        },
        final_recon_mse={
            "prot_bert": full_mse[0],
            "igbert_unpaired": full_mse[1],
            "igbert_paired": full_mse[2],
        },
        final_l0=float(full_l0),
        final_dead=int((~active_mask).sum()),
    )
    if betas is not None:
        summary["latent_scaling"] = {
            "pi_threshold": 0.5,
            "gained_evaluated": int(G),
            "flagged_artifacts_pi0.5": int(betas["artifact_flag"].sum()),
            "survive_pi0.5": int((~betas["artifact_flag"]).sum()),
            "flagged_artifacts_pi0.3": int(((betas["nu_r"] >= 0.3) | (betas["nu_e"] >= 0.3)).sum()),
            "gained_at_paired_flagged": int(betas["artifact_flag"][betas["target_m"] == 2].sum()),
            "gained_at_unpaired_flagged": int(betas["artifact_flag"][betas["target_m"] == 1].sum()),
        }
    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log("wrote summary; CATEGORY COUNTS:", counts)
    log("ALL DONE", tag)


if __name__ == "__main__":
    main()
