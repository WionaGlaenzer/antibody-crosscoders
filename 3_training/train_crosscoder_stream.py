#!/usr/bin/env python
"""
Train a joint 3-model BatchTopK crosscoder on one (regime, layer) activation store,
then classify latents by decoder-norm triple and run the Minder et al. (2504.02922)
Latent Scaling diagnostic on 'gained' latents.

Models axis (num_layers=3): 0=prot_bert, 1=igbert_unpaired, 2=igbert_paired.
"""

import argparse, os, sys, json, time, math
import numpy as np
import h5py
import torch as th


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ----------------------------------------------------------------------------
def build_memmap(h5path, split_val, tmpdir, tag, limit_rows=None, in_ram=False):
    """Copy the rows with split==split_val into a buffer. If in_ram, a plain RAM
    numpy array (fast random access for the training loop); else a node-local
    disk memmap (fine for the one-time sequential val pass). Returns (path, n, arr)."""
    f = h5py.File(h5path, "r")
    split = f["split"][:]
    idx = np.nonzero(split == split_val)[0]
    n = idx.size
    if limit_rows is not None and n > limit_rows:
        # deterministic contiguous cap for smoke test: keep the first `limit_rows` split rows
        cutoff = idx[limit_rows]
        split = split.copy()
        split[cutoff:] = 99  # mask out everything past the cap for this split
        n = limit_rows
    acts = f["acts"]
    N = acts.shape[0]
    if in_ram:
        out_path = None
        mm = np.empty((n, 3, 1024), dtype=np.float16)  # ~40.7GB for the full train split
    else:
        out_path = os.path.join(tmpdir, f"{tag}.f16")
        mm = np.memmap(out_path, dtype=np.float16, mode="w+", shape=(n, 3, 1024))
    # stream in contiguous file blocks; select the split rows within each block
    BLK = 262144
    write_ptr = 0
    for start in range(0, N, BLK):
        stop = min(start + BLK, N)
        block_mask = split[start:stop] == split_val
        k = int(block_mask.sum())
        if k == 0:
            continue
        chunk = acts[start:stop][block_mask]  # (k,3,1024) fp16
        mm[write_ptr : write_ptr + k] = chunk
        write_ptr += k
        if (start // BLK) % 8 == 0:
            log(f"  {tag}: filled {write_ptr}/{n} ({100*write_ptr/n:.0f}%)")
    assert write_ptr == n, (write_ptr, n)
    if hasattr(mm, "flush"):
        mm.flush()
    f.close()
    return out_path, n, mm


# ----------------------------------------------------------------------------
def stream_epoch_batches(h5path, batch_size, block_rows, epochs, seed):
    """Yield shuffled fp16 minibatches covering EVERY train row (split==0) once per
    epoch, streaming sequentially from disk so the full 2.5TB train split never has
    to fit in RAM. Rows are buffered in `block_rows`-sized blocks, shuffled within a
    block, and the sub-batch remainder is carried into the next block so only the
    final <batch of each epoch is dropped."""
    f = h5py.File(h5path, "r")
    split = f["split"][:]
    acts = f["acts"]
    N = acts.shape[0]
    rng = np.random.default_rng(seed)
    BLK = 262144
    try:
        for ep in range(epochs):
            buf = np.empty((block_rows, 3, 1024), dtype=np.float16)
            fill = 0
            for start in range(0, N, BLK):
                stop = min(start + BLK, N)
                m = split[start:stop] == 0
                k = int(m.sum())
                if k == 0:
                    continue
                chunk = acts[start:stop][m]  # (k,3,1024) fp16
                pos = 0
                while pos < k:
                    take = min(block_rows - fill, k - pos)
                    buf[fill : fill + take] = chunk[pos : pos + take]
                    fill += take
                    pos += take
                    if fill == block_rows:
                        perm = rng.permutation(fill)
                        nb = fill // batch_size
                        for b in range(nb):
                            yield buf[perm[b * batch_size : (b + 1) * batch_size]]
                        rem = fill - nb * batch_size
                        if rem:
                            buf[:rem] = buf[perm[nb * batch_size :]]
                        fill = rem
            if fill >= batch_size:  # flush tail of the epoch
                perm = rng.permutation(fill)
                nb = fill // batch_size
                for b in range(nb):
                    yield buf[perm[b * batch_size : (b + 1) * batch_size]]
    finally:
        f.close()


# ----------------------------------------------------------------------------
def classify_decoder_norms(Wdec):
    """
    Wdec: (3, dict, act) tensor. Returns per-latent per-model decoder norms (dict,3),
    the sum-normalized relative triple (dict,3) and integer category codes (dict,).
    Category codes:
      0 shared-all, 1 base-only, 2 gained-at-unpaired, 3 gained-at-paired,
      4 transient, 5 other  (dead handled separately by the caller)
    Thresholds (documented in report): HI=0.5, LO=0.2 on the sum-normalized triple
    (r_prot, r_unp, r_pair) where each r in [0,1] and they sum to 1.
    """
    n = th.linalg.norm(Wdec, dim=2).transpose(0, 1)  # (dict,3): [prot, unp, pair]
    s = n.sum(dim=1, keepdim=True)
    rel = n / (s + 1e-12)
    rp, ru, rpr = rel[:, 0], rel[:, 1], rel[:, 2]
    HI, LO = 0.5, 0.2
    cat = th.full((rel.shape[0],), 5, dtype=th.int64)  # default 'other'
    # shared-all: all three reasonably present (none dominant, none negligible)
    shared = (rp > LO) & (ru > LO) & (rpr > LO)
    # base-only: prot high, both igbert low
    base_only = (rp > HI) & (ru < LO) & (rpr < LO)
    # gained-at-unpaired: prot low, unpaired high, paired high (present in both igbert)
    gain_unp = (rp < LO) & (ru > LO) & (rpr > LO)
    # gained-at-paired: prot low AND unpaired low, paired high
    gain_pair = (rp < LO) & (ru < LO) & (rpr > HI)
    # transient: unpaired high, paired low (present at unpaired but lost at paired)
    transient = (ru > HI) & (rpr < LO)
    # assignment order: most specific last wins where overlaps could occur
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


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--dict_size", type=int, default=32768)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=35000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--cat_every", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_subset", type=int, default=150000)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--wandb_project", default="antibody-crosscoders")
    ap.add_argument("--wandb_entity", default="wiona-glaenzer-eth-z-rich")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_rows", type=int, default=None, help="cap rows per split (smoke test)")
    ap.add_argument(
        "--stream",
        action="store_true",
        help="stream all train residues from disk (true epochs, no RAM cap)",
    )
    ap.add_argument("--epochs", type=int, default=1, help="number of full epochs when --stream")
    ap.add_argument(
        "--stream_block", type=int, default=20000000, help="rows held in RAM per streaming block"
    )
    args = ap.parse_args()

    tag = f"{args.regime}_L{args.layer}"
    dev = "cuda" if th.cuda.is_available() else "cpu"
    log("device", dev, "| torch", th.__version__, "| CUDA", th.cuda.is_available())
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    from dictionary_learning.trainers.crosscoder import BatchTopKCrossCoderTrainer

    tmpdir = os.environ.get("TMPDIR", "/tmp")
    log("TMPDIR", tmpdir)

    # --- load normalization stats ---
    with h5py.File(args.h5, "r") as f:
        train_mean = th.tensor(f["train_mean"][:], dtype=th.float32)  # (3,1024)
        train_std = th.tensor(f["train_std"][:], dtype=th.float32)
        model_order = json.loads(f.attrs["model_order"])
    log("model_order", model_order)

    # --- build node-local memmaps ---
    t0 = time.time()
    if args.stream:
        # streaming mode: no RAM train buffer; just count train rows to size the schedule
        with h5py.File(args.h5, "r") as _f:
            n_train = int((_f["split"][:] == 0).sum())
        train_mm = None
        steps_per_epoch = n_train // args.batch_size
        args.steps = steps_per_epoch * args.epochs
        log(
            f"STREAM mode: n_train {n_train} | steps/epoch {steps_per_epoch} | epochs {args.epochs} | total steps {args.steps}"
        )
    else:
        # train buffer in RAM for fast random minibatch indexing (40.7GB full split)
        train_path, n_train, train_mm = build_memmap(
            args.h5, 0, tmpdir, f"train_{tag}", args.limit_rows, in_ram=True
        )
        log(
            f"train RAM buffer {n_train} rows ({train_mm.nbytes/1e9:.1f}GB) in {time.time()-t0:.0f}s"
        )
    t0 = time.time()
    # val on disk memmap (single sequential pass; no random access)
    # stream mode: cap val to a large representative subset (~10M rows) so the
    # node-local memmap and the final full-val analysis pass stay tractable.
    if args.stream:
        vlimit = 2_000_000
    else:
        vlimit = None if args.limit_rows is None else max(20000, args.limit_rows // 4)
    val_path, n_val, val_mm = build_memmap(args.h5, 1, tmpdir, f"val_{tag}", vlimit, in_ram=False)
    log(f"val memmap {n_val} rows in {time.time()-t0:.0f}s")

    # --- W&B ---
    os.environ.setdefault("WANDB_MODE", "online")
    use_wandb = True
    try:
        import wandb

        try:
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"cc_{tag}",
                config=dict(
                    regime=args.regime,
                    layer=args.layer,
                    dict_size=args.dict_size,
                    k=args.k,
                    batch_size=args.batch_size,
                    steps=args.steps,
                    n_train=n_train,
                    n_val=n_val,
                ),
            )
            log(
                "W&B run URL:",
                getattr(run, "url", "?"),
                "| entity:",
                run.entity,
                "| project:",
                run.project,
            )
        except Exception as e:
            log("wandb online init failed -> offline:", repr(e)[:200])
            os.environ["WANDB_MODE"] = "offline"
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"cc_{tag}",
                config=dict(regime=args.regime, layer=args.layer, k=args.k),
            )
    except Exception as e:
        log("wandb unavailable, continuing without:", repr(e)[:200])
        use_wandb = False
        run = None

    def wlog(d, step):
        if use_wandb:
            try:
                wandb.log(d, step=step)
            except Exception:
                pass

    # --- trainer ---
    decay_start = int(0.8 * args.steps)
    trainer = BatchTopKCrossCoderTrainer(
        steps=args.steps,
        activation_dim=1024,
        dict_size=args.dict_size,
        k=args.k,
        layer=args.layer,
        lm_name=f"3model_{tag}",
        num_layers=3,
        lr=None,
        warmup_steps=args.warmup,
        decay_start=decay_start,
        seed=args.seed,
        device=dev,
        wandb_name=f"cc_{tag}",
        activation_mean=train_mean,
        activation_std=train_std,
        target_rms=1.0,
        dict_class_kwargs={"code_normalization": "CROSSCODER"},
    )
    ae = trainer.ae
    log("trainer lr", trainer.lr, "| dict", args.dict_size, "| k", args.k)

    # --- fast RAM minibatch drawer: index in RAM, stage via pinned buffer, async copy ---
    stage = th.empty((args.batch_size, 3, 1024), dtype=th.float16, pin_memory=(dev == "cuda"))

    def draw_batch(mm, n, bs):
        idx = np.random.randint(0, n, size=bs)
        if bs == stage.shape[0]:
            stage.numpy()[:] = mm[idx]  # fancy-index straight into pinned buffer
            return stage.to(dev, non_blocking=True).float()
        x = np.asarray(mm[idx], dtype=np.float16)
        return th.from_numpy(x).to(dev).float()

    # --- fixed val subset held on GPU for frequent in-training eval ---
    n_eval = min(args.eval_subset, n_val)
    rng = np.random.default_rng(1234)
    eval_idx = np.sort(rng.choice(n_val, size=n_eval, replace=False))
    val_eval_gpu = th.from_numpy(np.asarray(val_mm[eval_idx], dtype=np.float16)).to(
        dev
    )  # (n_eval,3,1024)
    log(
        f"val-subset eval buffer: {n_eval} rows ({val_eval_gpu.element_size()*val_eval_gpu.nelement()/1e9:.2f}GB) on {dev}"
    )

    def eval_val_subset(step):
        """Compute val recon (per-model VE + MSE), L0, dead count, decoder-norm categories on the fixed subset."""
        ae.eval()
        EB = 16384
        tot_e = th.zeros(3, device=dev)
        tot_var = th.zeros(3, device=dev)
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
        counts = {CAT_NAMES[c]: int((catn == c).sum()) for c in [-1, 0, 1, 2, 3, 4, 5]}
        ae.train()
        return ve, mse, l0, n_active, counts

    # --- checkpoint helpers ---
    os.makedirs(args.ckpt_dir, exist_ok=True)
    last_path = os.path.join(args.ckpt_dir, f"cc_{tag}_last.pt")
    best_path = os.path.join(args.ckpt_dir, f"cc_{tag}_best.pt")
    side_path = os.path.join(args.ckpt_dir, f"cc_{tag}_progress.json")
    best_ve = -1e9

    def write_ckpt(path, step, ve_mean, is_best=False):
        th.save(ae.state_dict(), path)
        with open(side_path, "w") as fh:
            json.dump(
                dict(
                    tag=tag,
                    step=step,
                    total_steps=args.steps,
                    val_ve_mean=ve_mean,
                    best_val_ve_mean=best_ve,
                    last_ckpt=last_path,
                    best_ckpt=best_path,
                    ts=time.strftime("%Y-%m-%d %H:%M:%S"),
                ),
                fh,
                indent=2,
            )

    # --- training loop ---
    t0 = time.time()
    last_eval = {}
    if args.stream:

        def _stage_np(xnp):
            if xnp.shape[0] == stage.shape[0]:
                stage.numpy()[:] = xnp
                return stage.to(dev, non_blocking=True).float()
            return th.from_numpy(np.asarray(xnp, dtype=np.float16)).to(dev).float()

        batch_source = (
            (i, _stage_np(xnp))
            for i, xnp in enumerate(
                stream_epoch_batches(
                    args.h5, args.batch_size, args.stream_block, args.epochs, args.seed
                )
            )
        )
    else:
        batch_source = (
            (i, draw_batch(train_mm, n_train, args.batch_size)) for i in range(args.steps)
        )
    for step, x in batch_source:
        loss = trainer.update(step, x)
        if step % args.eval_every == 0 or step == args.steps - 1:
            ve, mse, l0, n_active, counts = eval_val_subset(step)
            ve_mean = float(np.mean(ve))
            # periodic + best checkpointing
            if step % args.ckpt_every == 0 or step == args.steps - 1:
                write_ckpt(last_path, step, ve_mean)
            if ve_mean > best_ve and step >= args.warmup:
                best_ve = ve_mean
                write_ckpt(best_path, step, ve_mean, is_best=True)
            dmet = {
                "train/loss": loss,
                "val/l0": l0,
                "val/dead_running": trainer.running_deads,
                "val/n_active": n_active,
                "val/recon_mse_prot_bert": mse[0],
                "val/recon_mse_igbert_unpaired": mse[1],
                "val/recon_mse_igbert_paired": mse[2],
                "val/var_expl_prot_bert": ve[0],
                "val/var_expl_igbert_unpaired": ve[1],
                "val/var_expl_igbert_paired": ve[2],
                "lr": trainer.scheduler.get_last_lr()[0],
            }
            dmet.update({f"cat/{k}": v for k, v in counts.items()})
            wlog(dmet, step)
            last_eval = dict(ve=ve, mse=mse, l0=l0, n_active=n_active, counts=counts)
            if step % (args.eval_every * 10) == 0 or step == args.steps - 1:
                log(
                    f"step {step}/{args.steps} trainloss {loss:.4f} l0 {l0:.1f} "
                    f"deads {trainer.running_deads} nact {n_active} "
                    f"valVE[p/u/pr] {ve[0]:.3f}/{ve[1]:.3f}/{ve[2]:.3f} "
                    f"shared {counts['shared-all']} gpair {counts['gained-at-paired']} "
                    f"gunp {counts['gained-at-unpaired']} ({(time.time()-t0)/(step+1)*1000:.0f} ms/step)"
                )
    log(f"training done in {(time.time()-t0)/60:.1f} min")
    del val_eval_gpu
    if dev == "cuda":
        th.cuda.empty_cache()

    # --- save checkpoint to project dir (absolute; too big to harvest) ---
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"cc_{tag}.pt")
    th.save(ae.state_dict(), ckpt_path)
    log("saved checkpoint", ckpt_path, f"({os.path.getsize(ckpt_path)/1e6:.0f} MB)")

    # ========================= ANALYSIS =========================
    ae.eval()
    dict_size = args.dict_size
    Wdec = ae.decoder.weight.detach()  # (3,dict,act)
    dnorm, rel, cat = classify_decoder_norms(Wdec)  # (dict,3),(dict,3),(dict,)

    # ---- FULL val pass: activation frequency + final per-model VE/MSE + L0 (converged ckpt) ----
    ABS = 8192
    n_active_counts = th.zeros(dict_size, device=dev)  # count of tokens where f>0
    fv_e = th.zeros(3, device=dev)
    fv_var = th.zeros(3, device=dev)
    fv_l0 = 0.0
    with th.no_grad():
        for s in range(0, n_val, ABS):
            xb = th.from_numpy(np.asarray(val_mm[s : s + ABS], dtype=np.float32)).to(dev)
            xn = ae.normalize_activations(xb, inplace=False)
            f = ae.encode(xn, normalize_activations=False, use_threshold=False)  # (b,dict) raw code
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

    # dead = never active on val
    cat_np = cat.cpu().numpy().copy()
    cat_np[~active_mask] = -1  # dead
    # gained latents = gained-at-paired(3) or gained-at-unpaired(2), and active
    gained_idx = np.nonzero(((cat_np == 2) | (cat_np == 3)))[0]
    log(
        f"active latents {active_mask.sum()} | gained {gained_idx.size} "
        f"(unp {(cat_np==2).sum()}, pair {(cat_np==3).sum()})"
    )

    # ---- Latent Scaling on gained latents (Minder et al. Eq.4, 33-36) ----
    # For latent j, model m: beta = <Y d_hat, f_j> / (||f_j||^2), with unit-norm d_hat.
    # beta_r: Y = full recon (denorm off).  beta_eps: Y = error + f_j * d_j  (add back j's own).
    # nu_r[j] = beta_r_base / beta_r_target ; nu_e[j] = beta_e_base / beta_e_target.
    # target model = paired(2) for gained-at-paired, unpaired(1) for gained-at-unpaired.
    G = gained_idx.size
    betas = None
    if G > 0:
        gidx_t = th.tensor(gained_idx, device=dev, dtype=th.long)
        Dg = Wdec[:, gidx_t, :]  # (3,G,act)
        Dg_norm = th.linalg.norm(Dg, dim=2)  # (3,G)
        Dg_hat = Dg / (Dg_norm.unsqueeze(2) + 1e-12)  # (3,G,act) unit
        num_r = th.zeros(3, G, device=dev)
        num_e = th.zeros(3, G, device=dev)
        den = th.zeros(G, device=dev)  # sum f_j^2  (shared across m)
        with th.no_grad():
            for s in range(0, n_val, ABS):
                xb = th.from_numpy(np.asarray(val_mm[s : s + ABS], dtype=np.float32)).to(dev)
                xn = ae.normalize_activations(xb, inplace=False)  # (b,3,act) normalized target
                f = ae.encode(xn, normalize_activations=False, use_threshold=False)  # (b,dict)
                xhat = ae.decode(f, denormalize_activations=False)  # (b,3,act) normalized recon
                fg = f[:, gidx_t]  # (b,G)
                den += (fg * fg).sum(0)
                err = xn - xhat  # (b,3,act)
                for m in range(3):
                    Dh = Dg_hat[m]  # (G,act)
                    proj_recon = xhat[:, m, :] @ Dh.t()  # (b,G)
                    num_r[m] += (proj_recon * fg).sum(0)
                    proj_err = err[:, m, :] @ Dh.t()  # (b,G)
                    # add back latent j's own contribution: f_j * (d_j . d_hat_j) = f_j * ||d_j||
                    proj_err = proj_err + fg * Dg_norm[m].unsqueeze(0)
                    num_e[m] += (proj_err * fg).sum(0)
        beta_r = (num_r / (den.unsqueeze(0) + 1e-12)).cpu().numpy()  # (3,G)
        beta_e = (num_e / (den.unsqueeze(0) + 1e-12)).cpu().numpy()  # (3,G)
        target_m = np.where(cat_np[gained_idx] == 3, 2, 1)  # paired->2, unpaired->1
        eps = 1e-9
        br_base = beta_r[0]
        be_base = beta_e[0]
        br_tgt = beta_r[target_m, np.arange(G)]
        be_tgt = beta_e[target_m, np.arange(G)]
        # clamp tiny/neg target betas
        nu_r = br_base / (np.abs(br_tgt) + eps)
        nu_e = be_base / (np.abs(be_tgt) + eps)
        nu_r = np.clip(nu_r, -1, 5)
        nu_e = np.clip(nu_e, -1, 5)
        # artifact = meaningful base presence (actually-shared): nu_r or nu_e >= pi
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
            f"latent scaling: gained {G}, flagged artifacts (pi=0.5) {int(artifact_flag.sum())}, "
            f"survive {int((~artifact_flag).sum())}"
        )

    # ---- write per-latent classification CSV (active latents) ----
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
    csv_path = f"classification_{tag}.csv"
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
    log("wrote", csv_path, f"({act_idx.size} active latents)")

    # ---- decoder-norm npz for figure (all latents) ----
    np.savez(
        f"decnorm_{tag}.npz", rel=rel_np, dnorm=dnorm_np, cat=cat_np, val_act_freq=val_act_freq
    )

    # ---- summary json ----
    counts = {CAT_NAMES[c]: int((cat_np == c).sum()) for c in [-1, 0, 1, 2, 3, 4, 5]}
    # final training metrics from last logging batch
    summary = dict(
        tag=tag,
        regime=args.regime,
        layer=args.layer,
        dict_size=dict_size,
        k=args.k,
        steps=args.steps,
        n_train=n_train,
        n_val=n_val,
        hyperparams=dict(
            batch_size=args.batch_size,
            k=args.k,
            dict_size=dict_size,
            num_layers=3,
            expansion_factor=dict_size // 1024,
            total_steps=args.steps,
            epochs_equiv=round(args.steps * args.batch_size / max(n_train, 1), 2),
            lr=trainer.lr,
            warmup_steps=args.warmup,
            decay_start=int(0.8 * args.steps),
            code_normalization="CROSSCODER",
            target_rms=1.0,
            auxk_alpha=trainer.auxk_alpha,
            eval_every=args.eval_every,
            eval_subset=int(min(args.eval_subset, n_val)),
            normalization="per-model train_mean/train_std (keep_relative_variance)",
            seed=args.seed,
        ),
        n_active=int(active_mask.sum()),
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
        dead_running_end=int(trainer.running_deads),
        eval_cadence=dict(
            eval_every=args.eval_every, eval_subset=int(min(args.eval_subset, n_val))
        ),
        ckpt_path=ckpt_path,
        ckpt_last=last_path,
        ckpt_best=best_path,
        ckpt_progress=side_path,
        best_val_ve_mean=float(best_ve),
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
    with open(f"summary_{tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log("wrote summary_" + tag + ".json")
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
