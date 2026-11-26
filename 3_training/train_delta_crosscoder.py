#!/usr/bin/env python
"""Delta-Crosscoder (arXiv:2603.04426) adapted for IgBERT-unpaired vs -paired.
Implements mechanisms (1) delta-loss and (2) dual-K partitioned dictionary from the
paper, WITHOUT (3) contrastive input pairs (we keep matched same-input activations,
appropriate for MLM encoders). Model axis: 0=unpaired, 1=paired (cols [1,2] of the
paired-regime store).

Dictionary split: first SHARED_FRAC of latents = shared (budget K_shared), remaining =
non-shared/delta (budget K_delta = alpha*K_shared). Loss = recon_unp + recon_pair +
delta_weight * || (b-a) - (Dpair - Dunp) z ||^2  + auxk.
"""

import argparse, os, sys, json, time
import numpy as np, h5py, torch as th
from torch.nn.functional import relu

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, f"{BASE}/code/dictionary_learning_jkminder")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


COLS = [1, 2]


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
    mm = (
        np.empty((n, 2, 1024), np.float16)
        if in_ram
        else np.memmap(
            os.path.join(tmpdir, f"{tag}.f16"), dtype=np.float16, mode="w+", shape=(n, 2, 1024)
        )
    )
    BLK = 262144
    wp = 0
    for s in range(0, N, BLK):
        e = min(s + BLK, N)
        m = split[s:e] == split_val
        k = int(m.sum())
        if k == 0:
            continue
        mm[wp : wp + k] = acts[s:e][m][:, COLS, :]
        wp += k
        if (s // BLK) % 8 == 0:
            log(f"  {tag}: {wp}/{n} ({100*wp/n:.0f}%)")
    assert wp == n, (wp, n)
    if hasattr(mm, "flush"):
        mm.flush()
    f.close()
    return n, mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--dict_size", type=int, default=32768)
    ap.add_argument("--k_shared", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5, help="K_delta = alpha*K_shared")
    ap.add_argument("--shared_frac", type=float, default=0.2)
    ap.add_argument("--delta_weight", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=32000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_subset", type=int, default=150000)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--wandb_project", default="antibody-crosscoders")
    ap.add_argument("--wandb_entity", default="wiona-glaenzer-eth-z-rich")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tag = f"{args.regime}_L{args.layer}"
    dev = "cuda" if th.cuda.is_available() else "cpu"
    log("device", dev, "torch", th.__version__)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    from dictionary_learning.dictionary import CrossCoderEncoder, CrossCoderDecoder

    tmpdir = os.environ.get("TMPDIR", "/tmp")
    with h5py.File(args.h5, "r") as f:
        tmean = th.tensor(f["train_mean"][:], dtype=th.float32)[COLS].to(dev)  # (2,1024)
        tstd = th.tensor(f["train_std"][:], dtype=th.float32)[COLS].to(dev)
        model_order = json.loads(f.attrs["model_order"])
    log("model_order", model_order, "cols", COLS)
    t0 = time.time()
    n_train, train_mm = build_memmap(args.h5, 0, tmpdir, f"train_{tag}", in_ram=True)
    log(f"train {n_train} rows ({train_mm.nbytes/1e9:.1f}GB) {time.time()-t0:.0f}s")
    t0 = time.time()
    n_val, val_mm = build_memmap(args.h5, 1, tmpdir, f"val_{tag}", in_ram=False)
    log(f"val {n_val} rows {time.time()-t0:.0f}s")

    D = args.dict_size
    n_shared = int(D * args.shared_frac)
    n_delta = D - n_shared
    K_shared = args.k_shared
    K_delta = max(1, int(args.alpha * K_shared))
    log(f"dict {D} | shared {n_shared} (K={K_shared}) | delta {n_delta} (K={K_delta})")
    enc = CrossCoderEncoder(activation_dim=1024, dict_size=D, num_layers=2, norm_init_scale=1.0).to(
        dev
    )
    dec = CrossCoderDecoder(
        activation_dim=1024,
        dict_size=D,
        num_layers=2,
        init_with_weight=enc.weight.data.transpose(1, 2).clone(),
    ).to(dev)
    params = list(enc.parameters()) + list(dec.parameters())
    lr = 2e-4 / (D / 16384) ** 0.5  # scale like reference
    opt = th.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    decay_start = int(0.8 * args.steps)

    def lr_at(step):
        if step < args.warmup:
            return step / args.warmup
        if step < decay_start:
            return 1.0
        return max(0.0, (args.steps - step) / (args.steps - decay_start))

    sched = th.optim.lr_scheduler.LambdaLR(opt, lr_at)

    os.environ.setdefault("WANDB_MODE", "online")
    use_wandb = True
    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"delta_{tag}",
            config=dict(
                regime=args.regime,
                layer=args.layer,
                dict_size=D,
                k_shared=K_shared,
                k_delta=K_delta,
                shared_frac=args.shared_frac,
                delta_weight=args.delta_weight,
                alpha=args.alpha,
                steps=args.steps,
                n_train=n_train,
                models="unpaired_vs_paired",
                method="delta-crosscoder",
            ),
        )
        log("W&B:", getattr(run, "url", "?"))
    except Exception as e:
        log("wandb off", repr(e)[:100])
        use_wandb = False

    def wlog(d, s):
        if use_wandb:
            try:
                wandb.log(d, step=s)
            except Exception:
                pass

    def normalize(x):  # x (B,2,1024)
        return (x - tmean) / tstd

    def dual_topk(u):
        """u: (B,D) pre-activation (already relu'd). Apply top-K_shared over shared block,
        top-K_delta over delta block, batchwise (BatchTopK-style over the batch*block)."""
        B = u.shape[0]
        z = th.zeros_like(u)
        # shared block: flatten (B*n_shared), keep top B*K_shared
        us = u[:, :n_shared].reshape(-1)
        ks = B * K_shared
        if ks < us.numel():
            th_s = th.topk(us, ks, sorted=False).values.min()
        else:
            th_s = us.min() - 1
        ms = (u[:, :n_shared] >= th_s) & (u[:, :n_shared] > 0)
        z[:, :n_shared] = u[:, :n_shared] * ms
        ud = u[:, n_shared:].reshape(-1)
        kd = B * K_delta
        if kd < ud.numel():
            th_d = th.topk(ud, kd, sorted=False).values.min()
        else:
            th_d = ud.min() - 1
        md = (u[:, n_shared:] >= th_d) & (u[:, n_shared:] > 0)
        z[:, n_shared:] = u[:, n_shared:] * md
        return z

    def encode(xn):  # returns z (B,D)
        u = enc(xn)  # (B,D) summed+bias+relu = averaged encoding
        return dual_topk(u)

    def decode(z):  # (B,D)->(B,2,1024)
        return dec(z)

    stage = th.empty((args.batch_size, 2, 1024), dtype=th.float16, pin_memory=(dev == "cuda"))

    def draw(mm, n, bs):
        idx = np.random.randint(0, n, size=bs)
        stage.numpy()[:] = mm[idx]
        return stage.to(dev, non_blocking=True).float()

    n_eval = min(args.eval_subset, n_val)
    rng = np.random.default_rng(1234)
    eidx = np.sort(rng.choice(n_val, size=n_eval, replace=False))
    val_gpu = th.from_numpy(np.asarray(val_mm[eidx], dtype=np.float16)).to(dev)

    def rel_delta_norm():
        """per-latent relative decoder norm r_pair = ||d_pair||/(||d_unp||+||d_pair||)."""
        W = dec.weight.detach()  # (2,D,1024)
        nu = th.linalg.norm(W[0], dim=1)
        npr = th.linalg.norm(W[1], dim=1)
        r_pair = npr / (nu + npr + 1e-12)
        return r_pair

    def evaluate():
        with th.no_grad():
            EB = 16384
            te = th.zeros(2, device=dev)
            tv = th.zeros(2, device=dev)
            l0 = 0.0
            nb = 0
            actany = th.zeros(D, device=dev)
            for s in range(0, n_eval, EB):
                xb = val_gpu[s : s + EB].float()
                xn = normalize(xb)
                z = encode(xn)
                xh = decode(z)
                e = xn - xh
                te += e.pow(2).sum(-1).sum(0)
                tv += (xn - xn.mean(0, keepdim=True)).pow(2).sum(-1).sum(0)
                l0 += (z != 0).float().sum(-1).sum().item()
                nb += xb.shape[0]
                actany += (z > 0).float().sum(0)
            ve = (1 - te / (tv + 1e-12)).tolist()
            l0 /= nb
            nact = int((actany / nb > 1e-4).sum())
            # categorize by relative decoder norm within delta block: paired-only if r_pair>0.7, unp-only if <0.3
            r = rel_delta_norm().cpu().numpy()
            live = (actany / nb > 1e-4).cpu().numpy()
            delta_live = live.copy()
            delta_live[:n_shared] = False
            paired_only = int(((r > 0.7) & delta_live).sum())
            unp_only = int(((r < 0.3) & delta_live).sum())
            shared_live = int((live[:n_shared]).sum())
            return ve, l0, nact, paired_only, unp_only, shared_live

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_ve = -1e9
    t0 = time.time()
    for step in range(args.steps):
        xb = draw(train_mm, n_train, args.batch_size)
        xn = normalize(xb)
        z = encode(xn)
        xh = decode(z)
        e = xn - xh
        recon = e.pow(2).sum(-1).mean()  # sum over act dim, mean over batch&layers
        # delta loss: (b-a) vs (Dpair - Dunp) z  -> reconstruct activation difference
        delta_true = xn[:, 1, :] - xn[:, 0, :]
        delta_hat = xh[:, 1, :] - xh[:, 0, :]
        delta_loss = (delta_true - delta_hat).pow(2).sum(-1).mean()
        loss = recon + args.delta_weight * delta_loss
        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            ve, l0, nact, po, uo, sl = evaluate()
            vem = float(np.mean(ve))
            if vem > best_ve and step >= args.warmup:
                best_ve = vem
                th.save(
                    {"enc": enc.state_dict(), "dec": dec.state_dict()},
                    os.path.join(args.ckpt_dir, f"delta_{tag}_best.pt"),
                )
            wlog(
                dict(
                    **{
                        "train/loss": loss.item(),
                        "train/recon": recon.item(),
                        "train/delta_loss": delta_loss.item(),
                        "val/l0": l0,
                        "val/nact": nact,
                        "val/ve_unp": ve[0],
                        "val/ve_pair": ve[1],
                        "cat/paired_only": po,
                        "cat/unp_only": uo,
                        "cat/shared_live": sl,
                        "lr": sched.get_last_lr()[0],
                    }
                ),
                step,
            )
            if step % (args.eval_every * 10) == 0 or step == args.steps - 1:
                log(
                    f"step {step}/{args.steps} loss {loss.item():.3f} recon {recon.item():.3f} dL {delta_loss.item():.3f} "
                    f"l0 {l0:.1f} nact {nact} VE[u/pr] {ve[0]:.3f}/{ve[1]:.3f} paired-only {po} unp-only {uo} ({(time.time()-t0)/(step+1)*1000:.0f}ms)"
                )
    log(f"done {(time.time()-t0)/60:.1f}min")
    th.save(
        {"enc": enc.state_dict(), "dec": dec.state_dict()},
        os.path.join(args.ckpt_dir, f"delta_{tag}.pt"),
    )
    # FINAL full-val analysis
    ve, l0, nact, po, uo, sl = evaluate()
    r = rel_delta_norm().cpu().numpy()
    with th.no_grad():
        actany = th.zeros(D, device=dev)
        for s in range(0, n_val, 16384):
            xb = th.from_numpy(np.asarray(val_mm[s : s + 16384], dtype=np.float32)).to(dev)
            z = encode(normalize(xb))
            actany += (z > 0).float().sum(0)
    freq = (actany / n_val).cpu().numpy()
    live = freq > 1e-4
    delta_live = live.copy()
    delta_live[:n_shared] = False
    summ = dict(
        tag=tag,
        method="delta-crosscoder",
        models="igbert_unpaired_vs_paired",
        layer=args.layer,
        dict_size=D,
        n_shared=n_shared,
        n_delta=n_delta,
        K_shared=K_shared,
        K_delta=K_delta,
        shared_frac=args.shared_frac,
        delta_weight=args.delta_weight,
        steps=args.steps,
        n_train=n_train,
        n_val=n_val,
        final_ve=dict(unp=ve[0], pair=ve[1]),
        n_active=int(live.sum()),
        n_active_shared=int(live[:n_shared].sum()),
        n_active_delta=int(delta_live.sum()),
        paired_only=int(((r > 0.7) & delta_live).sum()),
        unpaired_only=int(((r < 0.3) & delta_live).sum()),
        paired_leaning_delta=int(((r > 0.6) & delta_live).sum()),
        delta_block_rel_pair_norm=dict(
            median=float(np.median(r[delta_live])) if delta_live.any() else None,
            p90=float(np.percentile(r[delta_live], 90)) if delta_live.any() else None,
            max=float(r[delta_live].max()) if delta_live.any() else None,
        ),
        best_val_ve_mean=float(best_ve),
    )
    np.savez(f"delta_decnorm_{tag}.npz", rel_pair=r, freq=freq, n_shared=n_shared)
    json.dump(summ, open(f"delta_summary_{tag}.json", "w"), indent=2, default=float)
    log("SUMMARY", json.dumps(summ, default=float)[:800])
    if use_wandb:
        try:
            wandb.log({f"final/{k}": v for k, v in summ.items() if isinstance(v, (int, float))})
            wandb.finish()
        except Exception:
            pass
    log("ALL DONE", tag)


if __name__ == "__main__":
    main()
