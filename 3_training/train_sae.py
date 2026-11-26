#!/usr/bin/env python
"""
Train one per-model BatchTopK SAE (InterPLM/Haque-et-al. style) on a single model's
column of the joint activation store. Baseline against the 3-model joint crosscoder.

Column axis of h5 'acts' (N,3,1024): 0=prot_bert, 1=igbert_unpaired, 2=igbert_paired.
Reuses the SAME jkminder dictionary_learning repo as the crosscoder:
  dictionary_learning.trainers.batch_top_k.BatchTopKTrainer  (dict_class BatchTopKSAE)
Per-model activation normalization by train mean/std (shape (1024,)), Adam, lr auto
(=2e-4 for dict 16384), warmup, target_rms 1.0, auxk_alpha=1/32 for dead revival.
VE/MSE computed in the SAME normalized space as the crosscoder report (apples-to-apples).
"""

import argparse, os, sys, json, time
import numpy as np
import h5py
import torch as th


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ----------------------------------------------------------------------------
def build_col_buffer(h5path, split_val, col, tag, limit_rows=None):
    """Copy rows with split==split_val, ONE model column, into a RAM fp16 array (n,1024)."""
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
    mm = np.empty((n, 1024), dtype=np.float16)
    BLK = 262144
    write_ptr = 0
    for start in range(0, N, BLK):
        stop = min(start + BLK, N)
        block_mask = split[start:stop] == split_val
        k = int(block_mask.sum())
        if k == 0:
            continue
        block = acts[start:stop]  # (blk,3,1024) fp16
        chunk = block[block_mask][:, col, :]  # (k,1024)
        mm[write_ptr : write_ptr + k] = chunk
        write_ptr += k
        if (start // BLK) % 8 == 0:
            log(f"  {tag}: filled {write_ptr}/{n} ({100*write_ptr/max(n,1):.0f}%)")
    assert write_ptr == n, (write_ptr, n)
    f.close()
    return n, mm


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--col", type=int, required=True)  # 0 prot / 1 unp / 2 pair
    ap.add_argument("--name", required=True)  # sae_prot_L20 etc.
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--dict_size", type=int, default=16384)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=32000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--auxk_alpha", type=float, default=1 / 32)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_subset", type=int, default=150000)
    ap.add_argument("--ckpt_every", type=int, default=3000)
    ap.add_argument("--wandb_project", default="antibody-crosscoders")
    ap.add_argument("--wandb_entity", default="wiona-glaenzer-eth-z-rich")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_rows", type=int, default=None)
    ap.add_argument("--recon_loss", choices=["norm", "squared"], default="norm")
    ap.add_argument(
        "--dead_threshold_tokens",
        type=int,
        default=None,
        help="override dead-feature threshold (tokens since fired); default repo=1e7",
    )
    args = ap.parse_args()

    MODEL_NAMES = ["prot_bert", "igbert_unpaired", "igbert_paired"]
    model_name = MODEL_NAMES[args.col]
    dev = "cuda" if th.cuda.is_available() else "cpu"
    log("device", dev, "| torch", th.__version__, "| CUDA", th.cuda.is_available())
    log("SAE", args.name, "| column", args.col, model_name, "| layer", args.layer)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer
    from dictionary_learning.dictionary import BatchTopKSAE
    from collections import namedtuple

    class ConfigurableSAETrainer(BatchTopKTrainer):
        """BatchTopKTrainer with a selectable reconstruction-loss form and an
        adjustable dead-feature threshold. 'norm' matches the crosscoder recipe
        (l2_loss = ||e||) which does NOT collapse on these activations; 'squared'
        is the stock BatchTopKSAE form (l2_loss = ||e||^2) that collapsed."""

        def __init__(self, *a, recon_loss="norm", dead_threshold_tokens=None, **kw):
            super().__init__(*a, **kw)
            self.recon_loss = recon_loss
            if dead_threshold_tokens is not None:
                self.dead_feature_threshold = int(dead_threshold_tokens)
            self.last_auxk_loss = -1.0
            self.last_l2_loss = -1.0

        def update(self, step, x):
            # FIX: normalize BEFORE computing the geometric-median b_dec init, matching
            # BatchTopKCrossCoderTrainer.update(). The stock BatchTopKTrainer computes the
            # median on RAW x then normalizes, so b_dec lands at raw scale (norm ~ mean-norm)
            # but is used in normalized space -> near-constant encoder input -> collapse to k.
            from dictionary_learning.trainers.trainer import (
                remove_gradient_parallel_to_decoder_directions,
                set_decoder_norm_to_unit_norm,
            )

            x = x.to(self.device)
            x = self.ae.normalize_activations(x, inplace=True)  # normalize FIRST
            if step == 0:
                median = self.geometric_median(x)  # median on NORMALIZED x
                median = median.to(self.ae.b_dec.dtype)
                self.ae.b_dec.data = median
            loss = self.loss(x, step=step, normalize_activations=False)
            loss.backward()
            self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
                self.ae.decoder.weight,
                self.ae.decoder.weight.grad,
                self.ae.activation_dim,
                self.ae.dict_size,
            )
            th.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
                self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
            )
            return loss.item()

        def loss(
            self,
            x,
            step=None,
            logging=False,
            use_threshold=False,
            normalize_activations=True,
            inplace_normalize=True,
            **kwargs,
        ):
            x = (
                self.ae.normalize_activations(x, inplace=inplace_normalize)
                if normalize_activations
                else x
            )
            f, active_indices_F, post_relu_acts_BF = self.ae.encode(
                x, return_active=True, use_threshold=use_threshold, normalize_activations=False
            )
            if step > self.threshold_start_step:
                self.update_threshold(f)
            x_hat = self.ae.decode(f, denormalize_activations=False)
            e = x - x_hat
            self.effective_l0 = self.k
            num_tokens_in_step = x.size(0)
            did_fire = th.zeros_like(self.num_tokens_since_fired, dtype=th.bool)
            did_fire[active_indices_F] = True
            self.num_tokens_since_fired += num_tokens_in_step
            self.num_tokens_since_fired[did_fire] = 0
            if self.recon_loss == "norm":
                l2_loss = th.linalg.norm(e, dim=-1).mean()  # crosscoder-matched
            else:
                l2_loss = e.pow(2).sum(dim=-1).mean()  # stock squared
            auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)
            loss = l2_loss + self.auxk_alpha * auxk_loss
            self.last_l2_loss = float(l2_loss.item())
            self.last_auxk_loss = (
                float(auxk_loss.item()) if th.is_tensor(auxk_loss) else float(auxk_loss)
            )
            if not logging:
                return loss
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": self.last_auxk_loss,
                    "loss": loss.item(),
                    "deads": ~did_fire,
                },
            )

    # --- per-column normalization stats ---
    with h5py.File(args.h5, "r") as f:
        train_mean = th.tensor(f["train_mean"][args.col], dtype=th.float32)  # (1024,)
        train_std = th.tensor(f["train_std"][args.col], dtype=th.float32)  # (1024,)
        model_order = json.loads(f.attrs["model_order"])
    assert model_order[args.col] == model_name, (model_order, args.col)
    log(
        "norm stats", model_name, "mean|", float(train_mean.norm()), "std|", float(train_std.norm())
    )

    # --- build RAM buffers (one column) ---
    t0 = time.time()
    n_train, train_mm = build_col_buffer(
        args.h5, 0, args.col, f"train_{args.name}", args.limit_rows
    )
    log(f"train RAM buffer {n_train} rows ({train_mm.nbytes/1e9:.1f}GB) in {time.time()-t0:.0f}s")
    t0 = time.time()
    vlimit = None if args.limit_rows is None else max(20000, args.limit_rows // 4)
    n_val, val_mm = build_col_buffer(args.h5, 1, args.col, f"val_{args.name}", vlimit)
    log(f"val RAM buffer {n_val} rows ({val_mm.nbytes/1e9:.1f}GB) in {time.time()-t0:.0f}s")

    # --- W&B (WANDB dirs under node-local TMPDIR; home quota is exceeded) ---
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    for k_ in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"):
        os.environ.setdefault(k_, tmpdir)
    os.environ.setdefault("WANDB_MODE", "online")
    use_wandb = True
    run = None
    try:
        import wandb

        try:
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.name,
                config=dict(
                    model=model_name,
                    col=args.col,
                    layer=args.layer,
                    dict_size=args.dict_size,
                    k=args.k,
                    batch_size=args.batch_size,
                    steps=args.steps,
                    auxk_alpha=args.auxk_alpha,
                    n_train=n_train,
                    n_val=n_val,
                    baseline="per_model_SAE",
                ),
            )
            log(
                "W&B URL:",
                getattr(run, "url", "?"),
                "| entity",
                run.entity,
                "| project",
                run.project,
            )
        except Exception as e:
            log("wandb online failed -> offline:", repr(e)[:200])
            os.environ["WANDB_MODE"] = "offline"
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.name,
                config=dict(model=model_name, k=args.k),
            )
    except Exception as e:
        log("wandb unavailable:", repr(e)[:200])
        use_wandb = False
    wandb_url = getattr(run, "url", None) if run is not None else None

    def wlog(d, step):
        if use_wandb:
            try:
                import wandb

                wandb.log(d, step=step)
            except Exception:
                pass

    # --- trainer ---
    decay_start = int(0.8 * args.steps)
    trainer = ConfigurableSAETrainer(
        steps=args.steps,
        activation_dim=1024,
        dict_size=args.dict_size,
        k=args.k,
        layer=args.layer,
        lm_name=f"{model_name}_L{args.layer}",
        dict_class=BatchTopKSAE,
        lr=None,
        auxk_alpha=args.auxk_alpha,
        warmup_steps=args.warmup,
        decay_start=decay_start,
        seed=args.seed,
        device=dev,
        wandb_name=args.name,
        activation_mean=train_mean,
        activation_std=train_std,
        target_rms=1.0,
        recon_loss=args.recon_loss,
        dead_threshold_tokens=args.dead_threshold_tokens,
    )
    ae = trainer.ae
    log(
        "trainer lr",
        trainer.lr,
        "| dict",
        args.dict_size,
        "| k",
        args.k,
        "| auxk_alpha",
        args.auxk_alpha,
        "| recon_loss",
        args.recon_loss,
        "| dead_thresh_tokens",
        trainer.dead_feature_threshold,
    )

    # --- fast RAM minibatch drawer via pinned buffer ---
    stage = th.empty((args.batch_size, 1024), dtype=th.float16, pin_memory=(dev == "cuda"))

    def draw_batch(mm, n, bs):
        idx = np.random.randint(0, n, size=bs)
        if bs == stage.shape[0]:
            stage.numpy()[:] = mm[idx]
            return stage.to(dev, non_blocking=True).float()
        x = np.asarray(mm[idx], dtype=np.float16)
        return th.from_numpy(x).to(dev).float()

    # --- fixed val subset on GPU for frequent eval ---
    n_eval = min(args.eval_subset, n_val)
    rng = np.random.default_rng(1234)
    eval_idx = np.sort(rng.choice(n_val, size=n_eval, replace=False))
    val_eval_gpu = th.from_numpy(np.asarray(val_mm[eval_idx], dtype=np.float16)).to(dev)
    log(f"val-subset eval buffer {n_eval} rows on {dev}")

    def eval_val_subset():
        ae.eval()
        EB = 16384
        tot_e = 0.0
        tot_var = 0.0
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
                tot_e += e.pow(2).sum().item()
                tot_var += (xn - xn.mean(0, keepdim=True)).pow(2).sum().item()
                l0_sum += (f != 0).float().sum(-1).sum().item()
                nb += xb.shape[0]
                act_any += (f > 0).float().sum(0)
        ve = 1.0 - tot_e / (tot_var + 1e-12)
        mse = tot_e / (nb * 1024)  # per-element MSE
        mse_persample = tot_e / nb  # per-token summed MSE
        l0 = l0_sum / nb
        n_active = int((act_any / nb > 1e-4).sum())
        ae.train()
        return ve, mse, mse_persample, l0, n_active

    # --- checkpoints ---
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    last_path = os.path.join(args.ckpt_dir, f"{args.name}_last.pt")
    best_path = os.path.join(args.ckpt_dir, f"{args.name}_best.pt")
    final_path = os.path.join(args.ckpt_dir, f"{args.name}.pt")
    side_path = os.path.join(args.results_dir, f"{args.name}_progress.json")
    best_ve = -1e9

    def write_ckpt(path, step, ve):
        th.save(ae.state_dict(), path)
        with open(side_path, "w") as fh:
            json.dump(
                dict(
                    name=args.name,
                    step=step,
                    total_steps=args.steps,
                    val_ve=ve,
                    best_val_ve=best_ve,
                    last_ckpt=last_path,
                    best_ckpt=best_path,
                    ts=time.strftime("%Y-%m-%d %H:%M:%S"),
                ),
                fh,
                indent=2,
            )

    # --- training loop ---
    t0 = time.time()
    for step in range(args.steps):
        x = draw_batch(train_mm, n_train, args.batch_size)
        loss = trainer.update(step, x)
        if step % args.eval_every == 0 or step == args.steps - 1:
            ve, mse, mse_ps, l0, n_active = eval_val_subset()
            if step % args.ckpt_every == 0 or step == args.steps - 1:
                write_ckpt(last_path, step, ve)
            if ve > best_ve and step >= args.warmup:
                best_ve = ve
                write_ckpt(best_path, step, ve)
            wlog(
                {
                    "train/loss": loss,
                    "train/l2_loss": trainer.last_l2_loss,
                    "train/auxk_loss": trainer.last_auxk_loss,
                    "val/var_expl": ve,
                    "val/recon_mse": mse,
                    "val/recon_mse_persample": mse_ps,
                    "val/l0": l0,
                    "val/n_active": n_active,
                    "val/dead": args.dict_size - n_active,
                    "val/dead_running": int(trainer.dead_features),
                    "lr": trainer.scheduler.get_last_lr()[0],
                },
                step,
            )
            if step % (args.eval_every * 10) == 0 or step == args.steps - 1:
                log(
                    f"step {step}/{args.steps} loss {loss:.4f} valVE {ve:.4f} l0 {l0:.1f} "
                    f"nact {n_active} dead {args.dict_size-n_active} "
                    f"({(time.time()-t0)/(step+1)*1000:.0f} ms/step)"
                )
    train_min = (time.time() - t0) / 60
    log(f"training done in {train_min:.1f} min")
    del val_eval_gpu
    if dev == "cuda":
        th.cuda.empty_cache()
    th.save(ae.state_dict(), final_path)
    log("saved final checkpoint", final_path, f"({os.path.getsize(final_path)/1e6:.0f} MB)")

    # ---- FULL val pass: final VE / MSE / L0 / dead ----
    ae.eval()
    ABS = 8192
    fv_e = 0.0
    fv_var = 0.0
    fv_l0 = 0.0
    n_active_counts = th.zeros(args.dict_size, device=dev)
    with th.no_grad():
        for s in range(0, n_val, ABS):
            xb = th.from_numpy(np.asarray(val_mm[s : s + ABS], dtype=np.float32)).to(dev)
            xn = ae.normalize_activations(xb, inplace=False)
            f = ae.encode(xn, normalize_activations=False, use_threshold=False)
            xhat = ae.decode(f, denormalize_activations=False)
            e = xn - xhat
            fv_e += e.pow(2).sum().item()
            fv_var += (xn - xn.mean(0, keepdim=True)).pow(2).sum().item()
            fv_l0 += (f != 0).float().sum(-1).sum().item()
            n_active_counts += (f > 0).float().sum(0)
    full_ve = 1.0 - fv_e / (fv_var + 1e-12)
    full_mse = fv_e / (n_val * 1024)
    full_mse_ps = fv_e / n_val
    full_l0 = fv_l0 / n_val
    val_act_freq = (n_active_counts / n_val).cpu().numpy()
    n_active_full = int((val_act_freq > 1e-4).sum())
    dead_full = args.dict_size - n_active_full
    log(
        f"FULL val: VE {full_ve:.4f} MSE {full_mse:.5f} L0 {full_l0:.1f} "
        f"n_active {n_active_full} dead {dead_full} ({dead_full/args.dict_size:.4f})"
    )

    stats = dict(
        name=args.name,
        model=model_name,
        col=args.col,
        layer=args.layer,
        dict_size=args.dict_size,
        k=args.k,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=trainer.lr,
        auxk_alpha=args.auxk_alpha,
        recon_loss=args.recon_loss,
        dead_threshold_tokens=trainer.dead_feature_threshold,
        warmup=args.warmup,
        decay_start=decay_start,
        n_train=n_train,
        n_val=n_val,
        val_variance_explained=full_ve,
        val_recon_mse=full_mse,
        val_recon_mse_persample=full_mse_ps,
        val_l0=full_l0,
        n_active=n_active_full,
        dead_count=dead_full,
        dead_fraction=dead_full / args.dict_size,
        best_val_ve=best_ve,
        train_minutes=train_min,
        wandb_url=wandb_url,
        checkpoint=final_path,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        ts=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    stats_path = os.path.join(args.results_dir, f"{args.name}_stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    log("wrote stats", stats_path)
    # also save activation-frequency vector for downstream probe analysis
    np.savez(os.path.join(args.results_dir, f"{args.name}_actfreq.npz"), val_act_freq=val_act_freq)
    if use_wandb:
        try:
            import wandb

            wandb.summary.update(
                {
                    "final/val_ve": full_ve,
                    "final/val_mse": full_mse,
                    "final/val_l0": full_l0,
                    "final/dead": dead_full,
                    "final/train_minutes": train_min,
                }
            )
            wandb.finish()
        except Exception:
            pass
    log("DONE", args.name)


if __name__ == "__main__":
    main()
