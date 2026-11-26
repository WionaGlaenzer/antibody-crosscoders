#!/usr/bin/env python
"""
Train a 2-MODEL BatchTopK crosscoder on IgBERT-unpaired vs IgBERT-paired activations
(columns [1,2] of the paired-regime store; both models evaluated on the correctly
paired heavy-[SEP]-light input). Concentrates all dictionary capacity on the
unpaired<->paired difference to surface paired-specific features that a 3-model
crosscoder absorbs into 'shared'. Classifies latents by the 2-model decoder-norm
pair and runs the Minder et al. (2504.02922) Latent Scaling diagnostic on the
paired-only ('gained') set. Model axis (num_layers=2): 0=igbert_unpaired, 1=igbert_paired.
"""

import argparse, os, sys, json, time
import numpy as np, h5py, torch as th


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


COLS = [1, 2]  # unpaired, paired columns in the 3-model store


def build_memmap(h5path, split_val, tmpdir, tag, limit_rows=None, in_ram=False):
    f = h5py.File(h5path, "r")
    split = f["split"][:]
    idx = np.nonzero(split == split_val)[0]
    n = idx.size
    if limit_rows is not None and n > limit_rows:
        cutoff = idx[limit_rows]
        split = split.copy()
        split[cutoff:] = 99
        n = limit_rows
    acts = f["acts"]
    N = acts.shape[0]
    if in_ram:
        out_path = None
        mm = np.empty((n, 2, 1024), dtype=np.float16)
    else:
        out_path = os.path.join(tmpdir, f"{tag}.f16")
        mm = np.memmap(out_path, dtype=np.float16, mode="w+", shape=(n, 2, 1024))
    BLK = 262144
    write_ptr = 0
    for start in range(0, N, BLK):
        stop = min(start + BLK, N)
        block_mask = split[start:stop] == split_val
        k = int(block_mask.sum())
        if k == 0:
            continue
        chunk = acts[start:stop][block_mask][:, COLS, :]  # (k,2,1024) -> slice cols
        mm[write_ptr : write_ptr + k] = chunk
        write_ptr += k
        if (start // BLK) % 8 == 0:
            log(f"  {tag}: filled {write_ptr}/{n} ({100*write_ptr/n:.0f}%)")
    assert write_ptr == n, (write_ptr, n)
    if hasattr(mm, "flush"):
        mm.flush()
    f.close()
    return out_path, n, mm


def classify_decoder_norms(Wdec):
    """Wdec: (2,dict,act). rel=(r_unp,r_pair) sum to 1. HI=0.5,LO=0.2.
    0 shared, 1 unpaired-only, 2 paired-only, 5 other."""
    n = th.linalg.norm(Wdec, dim=2).transpose(0, 1)  # (dict,2)
    s = n.sum(dim=1, keepdim=True)
    rel = n / (s + 1e-12)
    ru, rpr = rel[:, 0], rel[:, 1]
    HI, LO = 0.5, 0.2
    cat = th.full((rel.shape[0],), 5, dtype=th.int64)
    cat[(ru > LO) & (rpr > LO)] = 0  # shared
    cat[(ru > HI) & (rpr < LO)] = 1  # unpaired-only
    cat[(rpr > HI) & (ru < LO)] = 2  # paired-only (gained)
    return n, rel, cat


CAT_NAMES = {0: "shared", 1: "unpaired-only", 2: "paired-only", 5: "other", -1: "dead"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--dict_size", type=int, default=32768)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=32000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_subset", type=int, default=150000)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--wandb_project", default="antibody-crosscoders")
    ap.add_argument("--wandb_entity", default="wiona-glaenzer-eth-z-rich")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_rows", type=int, default=None)
    args = ap.parse_args()
    tag = f"{args.regime}_L{args.layer}"
    dev = "cuda" if th.cuda.is_available() else "cpu"
    log("device", dev, "| torch", th.__version__)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    from dictionary_learning.trainers.crosscoder import BatchTopKCrossCoderTrainer

    tmpdir = os.environ.get("TMPDIR", "/tmp")
    log("TMPDIR", tmpdir)
    with h5py.File(args.h5, "r") as f:
        train_mean = th.tensor(f["train_mean"][:], dtype=th.float32)[COLS]  # (2,1024)
        train_std = th.tensor(f["train_std"][:], dtype=th.float32)[COLS]
        model_order = json.loads(f.attrs["model_order"])
    log("model_order(full)", model_order, "| using cols", COLS)
    t0 = time.time()
    train_path, n_train, train_mm = build_memmap(
        args.h5, 0, tmpdir, f"train_{tag}", args.limit_rows, in_ram=True
    )
    log(f"train RAM buffer {n_train} rows ({train_mm.nbytes/1e9:.1f}GB) in {time.time()-t0:.0f}s")
    t0 = time.time()
    vlimit = None if args.limit_rows is None else max(20000, args.limit_rows // 4)
    val_path, n_val, val_mm = build_memmap(args.h5, 1, tmpdir, f"val_{tag}", vlimit, in_ram=False)
    log(f"val memmap {n_val} rows in {time.time()-t0:.0f}s")
    os.environ.setdefault("WANDB_MODE", "online")
    use_wandb = True
    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"cc2_{tag}",
            config=dict(
                regime=args.regime,
                layer=args.layer,
                dict_size=args.dict_size,
                k=args.k,
                batch_size=args.batch_size,
                steps=args.steps,
                n_train=n_train,
                n_val=n_val,
                models="unpaired_vs_paired",
            ),
        )
        log("W&B run URL:", getattr(run, "url", "?"))
    except Exception as e:
        log("wandb unavailable:", repr(e)[:150])
        use_wandb = False
        run = None

    def wlog(d, step):
        if use_wandb:
            try:
                wandb.log(d, step=step)
            except Exception:
                pass

    decay_start = int(0.8 * args.steps)
    trainer = BatchTopKCrossCoderTrainer(
        steps=args.steps,
        activation_dim=1024,
        dict_size=args.dict_size,
        k=args.k,
        layer=args.layer,
        lm_name=f"2model_{tag}",
        num_layers=2,
        lr=None,
        warmup_steps=args.warmup,
        decay_start=decay_start,
        seed=args.seed,
        device=dev,
        wandb_name=f"cc2_{tag}",
        activation_mean=train_mean,
        activation_std=train_std,
        target_rms=1.0,
        dict_class_kwargs={"code_normalization": "CROSSCODER"},
    )
    ae = trainer.ae
    log("trainer lr", trainer.lr, "| dict", args.dict_size, "| k", args.k)
    stage = th.empty((args.batch_size, 2, 1024), dtype=th.float16, pin_memory=(dev == "cuda"))

    def draw_batch(mm, n, bs):
        idx = np.random.randint(0, n, size=bs)
        if bs == stage.shape[0]:
            stage.numpy()[:] = mm[idx]
            return stage.to(dev, non_blocking=True).float()
        x = np.asarray(mm[idx], dtype=np.float16)
        return th.from_numpy(x).to(dev).float()

    n_eval = min(args.eval_subset, n_val)
    rng = np.random.default_rng(1234)
    eval_idx = np.sort(rng.choice(n_val, size=n_eval, replace=False))
    val_eval_gpu = th.from_numpy(np.asarray(val_mm[eval_idx], dtype=np.float16)).to(dev)
    log(f"val-subset eval buffer: {n_eval} rows on {dev}")

    def eval_val_subset(step):
        ae.eval()
        EB = 16384
        tot_e = th.zeros(2, device=dev)
        tot_var = th.zeros(2, device=dev)
        l0_sum = 0.0
        nb = 0
        act_any = th.zeros(args.dict_size, device=dev)
        with th.no_grad():
            for s in range(0, n_eval, EB):
                xb = val_eval_gpu[s : s + EB].float()
                xn = ae.normalize_activations(xb, inplace=False)
                f = ae.encode(xn, normalize_activations=False, use_threshold=False)
                xhat = ae.decode(f, denormalize_activations=False)
                e = xn - xhat
                tot_e += e.pow(2).sum(-1).sum(0)
                tot_var += (xn - xn.mean(0, keepdim=True)).pow(2).sum(-1).sum(0)
                l0_sum += (f != 0).float().sum(-1).sum().item()
                nb += xb.shape[0]
                act_any += (f > 0).float().sum(0)
        ve = (1.0 - tot_e / (tot_var + 1e-12)).tolist()
        mse = (tot_e / nb).tolist()
        l0 = l0_sum / nb
        n_active = int((act_any / nb > 1e-4).sum())
        _, _, cat = classify_decoder_norms(ae.decoder.weight.detach())
        catn = cat.clone()
        catn[(act_any / nb) <= 1e-4] = -1
        counts = {CAT_NAMES[c]: int((catn == c).sum()) for c in [-1, 0, 1, 2, 5]}
        ae.train()
        return ve, mse, l0, n_active, counts

    os.makedirs(args.ckpt_dir, exist_ok=True)
    last_path = os.path.join(args.ckpt_dir, f"cc2_{tag}_last.pt")
    best_path = os.path.join(args.ckpt_dir, f"cc2_{tag}_best.pt")
    side_path = os.path.join(args.ckpt_dir, f"cc2_{tag}_progress.json")
    best_ve = -1e9

    def write_ckpt(path, step, ve_mean):
        th.save(ae.state_dict(), path)
        with open(side_path, "w") as fh:
            json.dump(
                dict(
                    tag=tag,
                    step=step,
                    total_steps=args.steps,
                    val_ve_mean=ve_mean,
                    best_val_ve_mean=best_ve,
                    ts=time.strftime("%Y-%m-%d %H:%M:%S"),
                ),
                fh,
                indent=2,
            )

    t0 = time.time()
    for step in range(args.steps):
        x = draw_batch(train_mm, n_train, args.batch_size)
        loss = trainer.update(step, x)
        if step % args.eval_every == 0 or step == args.steps - 1:
            ve, mse, l0, n_active, counts = eval_val_subset(step)
            ve_mean = float(np.mean(ve))
            if step % args.ckpt_every == 0 or step == args.steps - 1:
                write_ckpt(last_path, step, ve_mean)
            if ve_mean > best_ve and step >= args.warmup:
                best_ve = ve_mean
                write_ckpt(best_path, step, ve_mean)
            dmet = {
                "train/loss": loss,
                "val/l0": l0,
                "val/dead_running": trainer.running_deads,
                "val/n_active": n_active,
                "val/var_expl_unpaired": ve[0],
                "val/var_expl_paired": ve[1],
                "lr": trainer.scheduler.get_last_lr()[0],
            }
            dmet.update({f"cat/{k}": v for k, v in counts.items()})
            wlog(dmet, step)
            if step % (args.eval_every * 10) == 0 or step == args.steps - 1:
                log(
                    f"step {step}/{args.steps} loss {loss:.4f} l0 {l0:.1f} deads {trainer.running_deads} "
                    f"nact {n_active} valVE[u/pr] {ve[0]:.3f}/{ve[1]:.3f} shared {counts['shared']} "
                    f"unp-only {counts['unpaired-only']} pair-only {counts['paired-only']} "
                    f"({(time.time()-t0)/(step+1)*1000:.0f} ms/step)"
                )
    log(f"training done in {(time.time()-t0)/60:.1f} min")
    del val_eval_gpu
    if dev == "cuda":
        th.cuda.empty_cache()
    ckpt_path = os.path.join(args.ckpt_dir, f"cc2_{tag}.pt")
    th.save(ae.state_dict(), ckpt_path)
    log("saved checkpoint", ckpt_path, f"({os.path.getsize(ckpt_path)/1e6:.0f} MB)")
    # ANALYSIS
    ae.eval()
    dict_size = args.dict_size
    Wdec = ae.decoder.weight.detach()  # (2,dict,act)
    dnorm, rel, cat = classify_decoder_norms(Wdec)
    ABS = 8192
    n_active_counts = th.zeros(dict_size, device=dev)
    fv_e = th.zeros(2, device=dev)
    fv_var = th.zeros(2, device=dev)
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
    log(f"FULL val: VE[u/pr] {full_ve[0]:.3f}/{full_ve[1]:.3f} L0 {full_l0:.1f}")
    cat_np = cat.cpu().numpy().copy()
    cat_np[~active_mask] = -1
    gained_idx = np.nonzero(cat_np == 2)[0]  # paired-only
    log(
        f"active {active_mask.sum()} | paired-only(gained) {gained_idx.size} | unpaired-only {(cat_np==1).sum()} | shared {(cat_np==0).sum()}"
    )
    # Latent scaling on paired-only: base=unpaired(0), target=paired(1)
    G = gained_idx.size
    betas = None
    if G > 0:
        gidx_t = th.tensor(gained_idx, device=dev, dtype=th.long)
        Dg = Wdec[:, gidx_t, :]
        Dg_norm = th.linalg.norm(Dg, dim=2)
        Dg_hat = Dg / (Dg_norm.unsqueeze(2) + 1e-12)
        num_r = th.zeros(2, G, device=dev)
        num_e = th.zeros(2, G, device=dev)
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
                for m in range(2):
                    Dh = Dg_hat[m]
                    num_r[m] += ((xhat[:, m, :] @ Dh.t()) * fg).sum(0)
                    proj_err = err[:, m, :] @ Dh.t() + fg * Dg_norm[m].unsqueeze(0)
                    num_e[m] += (proj_err * fg).sum(0)
        beta_r = (num_r / (den.unsqueeze(0) + 1e-12)).cpu().numpy()
        beta_e = (num_e / (den.unsqueeze(0) + 1e-12)).cpu().numpy()
        eps = 1e-9
        nu_r = beta_r[0] / (np.abs(beta_r[1]) + eps)
        nu_e = beta_e[0] / (np.abs(beta_e[1]) + eps)
        nu_r = np.clip(nu_r, -1, 5)
        nu_e = np.clip(nu_e, -1, 5)
        artifact_flag = (nu_r >= 0.5) | (nu_e >= 0.5)
        betas = dict(gained_idx=gained_idx, nu_r=nu_r, nu_e=nu_e, artifact_flag=artifact_flag)
        log(
            f"latent scaling: paired-only {G}, flagged artifacts(pi=0.5) {int(artifact_flag.sum())}, survive {int((~artifact_flag).sum())}"
        )
    import csv

    act_idx = np.nonzero(active_mask)[0]
    nu_r_map = {}
    nu_e_map = {}
    flag_map = {}
    if betas is not None:
        for k, j in enumerate(betas["gained_idx"]):
            nu_r_map[int(j)] = float(betas["nu_r"][k])
            nu_e_map[int(j)] = float(betas["nu_e"][k])
            flag_map[int(j)] = bool(betas["artifact_flag"][k])
    dnorm_np = dnorm.cpu().numpy()
    rel_np = rel.cpu().numpy()
    with open(f"classification_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "latent_id",
                "n_unp",
                "n_pair",
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
                    f"{rel_np[j,0]:.6f}",
                    f"{rel_np[j,1]:.6f}",
                    CAT_NAMES[int(cat_np[j])],
                    f"{val_act_freq[j]:.3e}",
                    (f"{nu_r_map[j]:.4f}" if j in nu_r_map else ""),
                    (f"{nu_e_map[j]:.4f}" if j in nu_e_map else ""),
                    (str(flag_map[j]) if j in flag_map else ""),
                ]
            )
    np.savez(
        f"decnorm_{tag}.npz", rel=rel_np, dnorm=dnorm_np, cat=cat_np, val_act_freq=val_act_freq
    )
    counts = {CAT_NAMES[c]: int((cat_np == c).sum()) for c in [-1, 0, 1, 2, 5]}
    summary = dict(
        tag=tag,
        regime=args.regime,
        layer=args.layer,
        models="igbert_unpaired_vs_paired",
        dict_size=dict_size,
        k=args.k,
        steps=args.steps,
        n_train=n_train,
        n_val=n_val,
        hyperparams=dict(
            batch_size=args.batch_size,
            k=args.k,
            dict_size=dict_size,
            num_layers=2,
            expansion_factor=dict_size // 1024,
            total_steps=args.steps,
            epochs_equiv=round(args.steps * args.batch_size / max(n_train, 1), 2),
            lr=trainer.lr,
            warmup_steps=args.warmup,
            decay_start=decay_start,
            code_normalization="CROSSCODER",
            target_rms=1.0,
            auxk_alpha=trainer.auxk_alpha,
            eval_every=args.eval_every,
            eval_subset=int(min(args.eval_subset, n_val)),
            seed=args.seed,
        ),
        n_active=int(active_mask.sum()),
        category_counts=counts,
        paired_only=int((cat_np == 2).sum()),
        unpaired_only=int((cat_np == 1).sum()),
        final_var_expl={"igbert_unpaired": full_ve[0], "igbert_paired": full_ve[1]},
        final_recon_mse={"igbert_unpaired": full_mse[0], "igbert_paired": full_mse[1]},
        final_l0=float(full_l0),
        final_dead=int((~active_mask).sum()),
        dead_running_end=int(trainer.running_deads),
        ckpt_path=ckpt_path,
        best_val_ve_mean=float(best_ve),
    )
    if betas is not None:
        summary["latent_scaling"] = {
            "pi_threshold": 0.5,
            "gained_evaluated": int(G),
            "flagged_artifacts_pi0.5": int(betas["artifact_flag"].sum()),
            "survive_pi0.5": int((~betas["artifact_flag"]).sum()),
            "flagged_artifacts_pi0.3": int(((betas["nu_r"] >= 0.3) | (betas["nu_e"] >= 0.3)).sum()),
        }
    with open(f"summary_{tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log("CATEGORY COUNTS:", counts)
    if use_wandb:
        try:
            wandb.log({f"final_cat/{k}": v for k, v in counts.items()})
            wandb.finish()
        except Exception:
            pass
    log("ALL DONE", tag)


if __name__ == "__main__":
    main()
