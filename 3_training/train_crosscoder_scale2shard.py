#!/usr/bin/env python
"""
Two-shard SINGLE-EPOCH trainer for the SCALE-UP 3-model BatchTopK crosscoder.

The scale stores have a ~408 GB train buffer that does NOT fit in RAM on a standard
GPU node, and streaming random-access from the HDF5 on Lustre stalls (GPU idle).
Solution: a TWO-SHARD single epoch.
  - Randomly permute the train-split rows (NOT contiguous — the store is seq_id/kind
    sorted, so contiguous halves would be biased) and split into two halves.
  - Load HALF 1 into RAM (~204 GB, fits on a 501 GB node), train for the first
    half of the total steps, CHECKPOINT model + optimizer + scheduler + step + RNG.
  - Free half 1, load HALF 2, RESUME from the checkpoint, train the remaining steps.
Every train residue is seen -> one full epoch, without ever holding 408 GB resident
and without touching the streaming path.

Normalization: uses the store's precomputed per-model train_mean/train_std (computed
ONCE over the full train split at extraction), fixed for BOTH shards.

Downstream analysis (val VE, latent scaling, classification CSV, decoder-norm npz,
summary json) is identical to train_crosscoder.py.

Models axis (num_layers=3): 0=prot_bert, 1=igbert_unpaired, 2=igbert_paired.
"""

import argparse, os, sys, json, time, math
import numpy as np
import h5py
import torch as th


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ----------------------------------------------------------------------------
def build_shard_ram(h5path, split_val, shard_rows, tag):
    """Copy the given GLOBAL row indices (shard_rows, sorted) of split==split_val
    into a RAM float16 array via contiguous block reads. shard_rows are absolute
    row indices into acts (already filtered to this split and to this shard)."""
    f = h5py.File(h5path, "r")
    acts = f["acts"]
    N = acts.shape[0]
    n = shard_rows.size
    mm = np.empty((n, 3, 1024), dtype=np.float16)
    # membership mask over the whole file for fast contiguous-block selection
    want = np.zeros(N, dtype=bool)
    want[shard_rows] = True
    # BLK MUST match the shard-assignment block size (SHARD_BLK=262144): with block-strided shards,
    # a block is either entirely owned by this shard (all its train rows wanted) or not owned at all
    # (k==0 -> skipped without touching the disk). This makes each shard load read only its own
    # ~1/n_shards of the file, fully sequentially, instead of scanning the whole 583 GB store.
    BLK = 262144
    write_ptr = 0
    n_read_blocks = 0
    for start in range(0, N, BLK):
        stop = min(start + BLK, N)
        bmask = want[start:stop]
        k = int(bmask.sum())
        if k == 0:
            continue  # block not owned by this shard -> skip (no disk read)
        mm[write_ptr : write_ptr + k] = acts[start:stop][bmask]
        write_ptr += k
        n_read_blocks += 1
        if n_read_blocks % 16 == 0:
            log(f"  {tag}: filled {write_ptr}/{n} ({100*write_ptr/max(n,1):.0f}%)")
    assert write_ptr == n, (write_ptr, n)
    f.close()
    return n, mm


def build_memmap_split(h5path, split_val, tmpdir, tag, in_ram=False, limit_rows=None):
    """Load ALL rows of split==split_val (used for the val split; small enough)."""
    f = h5py.File(h5path, "r")
    split = f["split"][:]
    idx = np.nonzero(split == split_val)[0]
    if limit_rows is not None and idx.size > limit_rows:
        # evenly-strided subsample across the whole split (NOT first-N: the store is
        # seq_id/kind sorted, so first-N would be a biased slice). A ~2M-row strided
        # sample gives a val-VE / activation-frequency estimate indistinguishable from
        # the full split while cutting the memmap build from hours to minutes.
        stride = idx.size / float(limit_rows)
        idx = idx[(np.arange(limit_rows) * stride).astype(np.int64)]
    n = idx.size
    acts = f["acts"]
    N = acts.shape[0]
    keep = np.zeros(N, dtype=bool)
    keep[idx] = True
    if in_ram:
        out_path = None
        mm = np.empty((n, 3, 1024), dtype=np.float16)
    else:
        out_path = os.path.join(tmpdir, f"{tag}.f16")
        mm = np.memmap(out_path, dtype=np.float16, mode="w+", shape=(n, 3, 1024))
    BLK = 262144
    wp = 0
    for start in range(0, N, BLK):
        stop = min(start + BLK, N)
        bmask = keep[start:stop]
        k = int(bmask.sum())
        if k == 0:
            continue
        mm[wp : wp + k] = acts[start:stop][bmask]
        wp += k
    assert wp == n, (wp, n)
    if hasattr(mm, "flush"):
        mm.flush()
    f.close()
    return out_path, n, mm


# ----------------------------------------------------------------------------
def classify_decoder_norms(Wdec):
    n = th.linalg.norm(Wdec, dim=2).transpose(0, 1)  # (dict,3): [prot, unp, pair]
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
    ap.add_argument(
        "--steps", type=int, default=35000, help="TOTAL steps across both shards (one epoch worth)"
    )
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_subset", type=int, default=150000)
    ap.add_argument(
        "--val_cap",
        type=int,
        default=2000000,
        help="max val rows in the full-val memmap (block-strided subsample; None-like large value = all)",
    )
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--wandb_project", default="antibody-crosscoders")
    ap.add_argument("--wandb_entity", default="wiona-glaenzer-eth-z-rich")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument(
        "--n_shards",
        type=int,
        default=2,
        help="number of RAM-fitting train shards for the single epoch",
    )
    args = ap.parse_args()

    tag = args.tag if args.tag else f"{args.regime}_scale_L{args.layer}"
    dev = "cuda" if th.cuda.is_available() else "cpu"
    log("device", dev, "| torch", th.__version__, "| CUDA", th.cuda.is_available(), "| tag", tag)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    from dictionary_learning.trainers.crosscoder import BatchTopKCrossCoderTrainer

    tmpdir = os.environ.get("TMPDIR", "/tmp")
    results_dir = args.results_dir or os.getcwd()
    os.makedirs(results_dir, exist_ok=True)

    # --- norm stats (fixed over full train split, from the store) ---
    with h5py.File(args.h5, "r") as f:
        train_mean = th.tensor(f["train_mean"][:], dtype=th.float32)
        train_std = th.tensor(f["train_std"][:], dtype=th.float32)
        model_order = json.loads(f.attrs["model_order"])
        split = f["split"][:]
    log("model_order", model_order)

    # --- build a BLOCK-STRIDED shard partition of the TRAIN split (all rows seen exactly once) ---
    # Instead of a global row permutation (which scatters each shard across the whole file and
    # forces reading the ENTIRE 583 GB store for every shard), assign whole contiguous BLOCKS of
    # rows round-robin to shards. Each shard then owns a set of large contiguous file ranges, so
    # its load reads ONLY its own blocks (skipping the others) with fully sequential I/O -> ~n_shards x
    # less I/O. The store is seq_id/kind-sorted, so alternating large blocks still gives each shard a
    # representative spread across the distribution (same rationale as the block-strided VE sampler).
    N_total = split.shape[0]
    train_idx = np.nonzero(split == 0)[0]
    n_train = train_idx.size
    SHARD_BLK = 262144  # block granularity for shard assignment
    n_blocks = (N_total + SHARD_BLK - 1) // SHARD_BLK
    block_shard = np.arange(n_blocks) % args.n_shards  # which shard owns each file block
    # row -> block -> shard; keep only TRAIN rows, grouped per shard, ascending (contiguous) order
    is_train = split == 0
    row_block = np.arange(N_total) // SHARD_BLK
    row_shard = block_shard[row_block]
    shard_row_lists = [np.nonzero(is_train & (row_shard == i))[0] for i in range(args.n_shards)]
    for i, s in enumerate(shard_row_lists):
        log(f"shard {i}: {s.size} train rows ({(block_shard==i).sum()} blocks)")
    assert (
        sum(s.size for s in shard_row_lists) == n_train
    ), "shard partition must cover all train rows once"
    steps_per_shard = args.steps // args.n_shards  # split the total step budget evenly

    # --- val on disk memmap (single sequential pass) ---
    t0 = time.time()
    val_path, n_val, val_mm = build_memmap_split(
        args.h5, 1, tmpdir, f"val_{tag}", in_ram=False, limit_rows=args.val_cap
    )
    log(f"val memmap {n_val} rows in {time.time()-t0:.0f}s (cap {args.val_cap})")

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
                    n_shards=args.n_shards,
                    mode="two_shard_single_epoch",
                ),
            )
        except Exception as e:
            log("wandb online failed -> offline:", repr(e)[:150])
            os.environ["WANDB_MODE"] = "offline"
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"cc_{tag}",
                config=dict(regime=args.regime, layer=args.layer, k=args.k),
            )
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

    # --- trainer (built ONCE; optimizer/scheduler persist across shards) ---
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
    log(
        "trainer lr",
        trainer.lr,
        "| dict",
        args.dict_size,
        "| k",
        args.k,
        "| total steps",
        args.steps,
        "| steps/shard",
        steps_per_shard,
    )

    stage = th.empty((args.batch_size, 3, 1024), dtype=th.float16, pin_memory=(dev == "cuda"))

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

    os.makedirs(args.ckpt_dir, exist_ok=True)
    last_path = os.path.join(args.ckpt_dir, f"cc_{tag}_last.pt")
    best_path = os.path.join(args.ckpt_dir, f"cc_{tag}_best.pt")
    shard_ckpt = os.path.join(args.ckpt_dir, f"cc_{tag}_shardstate.pt")
    side_path = os.path.join(args.ckpt_dir, f"cc_{tag}_progress.json")
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

    # --- TWO-SHARD training: full state (model+optim+sched) persists across shards in-process ---
    t0 = time.time()
    global_step = 0
    last_eval = {}
    for shard_i in range(args.n_shards):
        log(
            f"=== SHARD {shard_i}/{args.n_shards}: loading {shard_row_lists[shard_i].size} rows into RAM ==="
        )
        ts = time.time()
        n_sh, train_mm = build_shard_ram(
            args.h5, 0, shard_row_lists[shard_i], f"train_sh{shard_i}_{tag}"
        )
        log(
            f"shard {shard_i} RAM buffer {n_sh} rows ({train_mm.nbytes/1e9:.1f}GB) in {time.time()-ts:.0f}s"
        )
        sh_steps = (
            steps_per_shard
            if shard_i < args.n_shards - 1
            else (args.steps - steps_per_shard * (args.n_shards - 1))
        )
        for _ in range(sh_steps):
            x = draw_batch(train_mm, n_sh, args.batch_size)
            loss = trainer.update(
                global_step, x
            )  # trainer owns optimizer+scheduler; global_step drives LR schedule
            if global_step % args.eval_every == 0 or global_step == args.steps - 1:
                ve, mse, l0, n_active, counts = eval_val_subset(global_step)
                ve_mean = float(np.mean(ve))
                if global_step % args.ckpt_every == 0 or global_step == args.steps - 1:
                    write_ckpt(last_path, global_step, ve_mean)
                if ve_mean > best_ve and global_step >= args.warmup:
                    best_ve = ve_mean
                    write_ckpt(best_path, global_step, ve_mean)
                dmet = {
                    "train/loss": loss,
                    "val/l0": l0,
                    "val/dead_running": trainer.running_deads,
                    "val/n_active": n_active,
                    "val/var_expl_prot_bert": ve[0],
                    "val/var_expl_igbert_unpaired": ve[1],
                    "val/var_expl_igbert_paired": ve[2],
                    "lr": trainer.scheduler.get_last_lr()[0],
                    "shard": shard_i,
                }
                dmet.update({f"cat/{k}": v for k, v in counts.items()})
                wlog(dmet, global_step)
                last_eval = dict(ve=ve, mse=mse, l0=l0, n_active=n_active, counts=counts)
                if global_step % (args.eval_every * 10) == 0 or global_step == args.steps - 1:
                    log(
                        f"step {global_step}/{args.steps} [sh{shard_i}] loss {loss:.4f} l0 {l0:.1f} "
                        f"nact {n_active} valVE {ve[0]:.3f}/{ve[1]:.3f}/{ve[2]:.3f} "
                        f"gpair {counts['gained-at-paired']} gunp {counts['gained-at-unpaired']} "
                        f"({(time.time()-t0)/(global_step+1)*1000:.0f} ms/step)"
                    )
            global_step += 1
        # free shard buffer + checkpoint full state between shards
        del train_mm
        import gc

        gc.collect()
        th.save(
            dict(
                model=ae.state_dict(),
                optimizer=trainer.optimizer.state_dict(),
                scheduler=trainer.scheduler.state_dict(),
                global_step=global_step,
                shard_done=shard_i,
                best_ve=best_ve,
            ),
            shard_ckpt,
        )
        log(f"=== SHARD {shard_i} done at global_step {global_step}; full state checkpointed ===")
    log(
        f"training done in {(time.time()-t0)/60:.1f} min ({global_step} steps, {args.n_shards} shards = 1 epoch)"
    )
    del val_eval_gpu
    if dev == "cuda":
        th.cuda.empty_cache()

    ckpt_path = os.path.join(args.ckpt_dir, f"cc_{tag}.pt")
    th.save(ae.state_dict(), ckpt_path)
    log("saved checkpoint", ckpt_path, f"({os.path.getsize(ckpt_path)/1e6:.0f} MB)")

    # ========================= ANALYSIS (val VE + classify + npz + summary) =========================
    ae.eval()
    Wdec = ae.decoder.weight.detach()
    dnorm, rel, cat = classify_decoder_norms(Wdec)
    # full val pass: activation frequency + per-model VE/MSE + L0
    EB = 16384
    tot_e = th.zeros(3, device=dev)
    tot_var = th.zeros(3, device=dev)
    l0_sum = 0.0
    nb = 0
    act_cnt = th.zeros(args.dict_size, device=dev)
    with th.no_grad():
        for s in range(0, n_val, EB):
            xb = th.from_numpy(np.asarray(val_mm[s : s + EB], dtype=np.float16)).to(dev).float()
            xn = ae.normalize_activations(xb, inplace=False)
            f = ae.encode(xn, normalize_activations=False, use_threshold=False)
            xhat = ae.decode(f, denormalize_activations=False)
            e = xn - xhat
            tot_e += e.pow(2).sum(-1).sum(0)
            tot_var += (xn - xn.mean(0, keepdim=True)).pow(2).sum(-1).sum(0)
            l0_sum += (f != 0).float().sum(-1).sum().item()
            nb += xb.shape[0]
            act_cnt += (f > 0).float().sum(0)
    ve = (1.0 - tot_e / (tot_var + 1e-12)).tolist()
    mse = (tot_e / nb).tolist()
    l0 = l0_sum / nb
    freq = (act_cnt / nb).cpu().numpy()
    active_mask = freq > 1e-4
    catn = cat.clone()
    catn[~th.from_numpy(active_mask).to(cat.device)] = -1
    counts = {CAT_NAMES[c]: int((catn == c).sum()) for c in [-1, 0, 1, 2, 3, 4, 5]}
    n_active = int(active_mask.sum())
    log(f"FULL val: VE {ve} n_active {n_active} counts {counts}")

    # per-latent classification CSV
    import csv

    rel_np = rel.cpu().numpy()
    dn_np = dnorm.cpu().numpy()
    catn_np = catn.cpu().numpy()
    csv_path = os.path.join(results_dir, f"classification_{tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "latent",
                "freq",
                "cat",
                "rel_prot",
                "rel_unp",
                "rel_pair",
                "n_prot",
                "n_unp",
                "n_pair",
            ]
        )
        for i in range(args.dict_size):
            if not active_mask[i]:
                continue
            w.writerow(
                [
                    i,
                    f"{freq[i]:.6g}",
                    CAT_NAMES[int(catn_np[i])],
                    f"{rel_np[i,0]:.4f}",
                    f"{rel_np[i,1]:.4f}",
                    f"{rel_np[i,2]:.4f}",
                    f"{dn_np[i,0]:.4f}",
                    f"{dn_np[i,1]:.4f}",
                    f"{dn_np[i,2]:.4f}",
                ]
            )
    # decoder-norm npz
    np.savez(
        os.path.join(results_dir, f"decnorm_{tag}.npz"),
        dnorm=dn_np,
        rel=rel_np,
        cat=catn_np,
        freq=freq,
    )
    # summary json
    summ = dict(
        tag=tag,
        regime=args.regime,
        layer=args.layer,
        dict_size=args.dict_size,
        k=args.k,
        steps=args.steps,
        mode="two_shard_single_epoch",
        n_shards=args.n_shards,
        n_train=int(n_train),
        n_val=int(n_val),
        n_active=n_active,
        category_counts=counts,
        gained_at_paired=counts["gained-at-paired"],
        gained_at_unpaired=counts["gained-at-unpaired"],
        final_var_expl=dict(prot_bert=ve[0], igbert_unpaired=ve[1], igbert_paired=ve[2]),
        final_recon_mse=dict(prot_bert=mse[0], igbert_unpaired=mse[1], igbert_paired=mse[2]),
        final_l0=l0,
        best_val_ve_mean=best_ve,
        ckpt_path=ckpt_path,
    )
    with open(os.path.join(results_dir, f"summary_{tag}.json"), "w") as fh:
        json.dump(summ, fh, indent=2)
    log("wrote classification CSV + decnorm npz + summary json to", results_dir)
    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
