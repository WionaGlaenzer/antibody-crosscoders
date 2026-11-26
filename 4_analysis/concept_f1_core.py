#!/usr/bin/env python3
"""Shared primitives for the concept-F1 analyses (InterPLM-style Swiss-Prot concept
matching on the mixed crosscoder).

Every concept_f1_* script performs the same core computation — encode general-protein
residues through the mixed BatchTopK crosscoder, collect the nonzero
(latent, activation, concept-bitmask) triples, and run the exact InterPLM
sorted-threshold max-F1 sweep per latent against each concept. Those pieces lived
copy-pasted in each script and are factored here VERBATIM so there is one definition of
the sweep (a copy-paste bug in one script but not the others would otherwise produce
inconsistent F1 numbers between figures).

The per-script differences that DO matter stay in the individual scripts and are passed
in as arguments: which activation store / checkpoint, cached vs freshly-fetched concept
labels, whole-set vs test-only vs 5-fold evaluation.
"""

import time
import numpy as np
import torch as th


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def within_seq_positions(sid):
    """Given the store's seq_id column (sorted ascending), return (u, first, cnt, within):
    unique seq ids, their first-row index, per-seq residue count, and a per-row 0-indexed
    within-sequence position. Identical grouped-counts logic used by every script."""
    N = sid.shape[0]
    u, first, cnt = np.unique(sid, return_index=True, return_counts=True)
    within = np.empty(N, dtype=np.int64)
    for f0, c in zip(first, cnt):
        within[f0 : f0 + c] = np.arange(c)
    return u, first, cnt, within


def concept_bits(nc):
    """Bit table [1<<0, 1<<1, ...] for unpacking an int concept bitmask into NC columns."""
    return np.array([1 << i for i in range(nc)], dtype=np.int32)


def encode_collect_triples(ae, h5_acts, N, cmask_t, dev, blk=200000, enc=16384, keep_mask=None):
    """Encode residue activations through the crosscoder in blocks and collect the nonzero
    (latent, activation, concept-mask) triples — the sparse representation the sweep needs.

    ae         : loaded BatchTopKCrossCoder (eval mode)
    h5_acts    : an open h5py dataset ("acts") of shape (N, NM, HID)
    N          : number of residue rows
    cmask_t    : torch int tensor (N,) per-residue concept bitmask (global row indexing)
    keep_mask  : optional bool np.ndarray (N,) — if given, only rows where True are kept
                 (used by the test-only variant; None keeps all rows).

    Returns (lat, act, cm) concatenated numpy arrays over all kept firing residues.
    Uses ae.encode(use_threshold=True, normalize_activations=True) exactly as every script."""
    lat_buf, act_buf, cm_buf = [], [], []
    tot_nz = 0
    t0 = time.time()
    for s in range(0, N, blk):
        e = min(s + blk, N)
        x = h5_acts[s:e]
        km_blk = keep_mask[s:e] if keep_mask is not None else None
        for b0 in range(0, e - s, enc):
            b1 = min(b0 + enc, e - s)
            xb = th.tensor(x[b0:b1].astype(np.float32), device=dev)
            with th.no_grad():
                f = ae.encode(xb, use_threshold=True, normalize_activations=True)
            if km_blk is not None:
                keep_rows = th.tensor(km_blk[b0:b1], device=dev)
                f = f[keep_rows]
                gidx = np.nonzero(km_blk[b0:b1])[0]  # local kept-row -> local block-row map
            nz = f.nonzero(as_tuple=True)
            r_i, l_i = nz
            a_v = f[r_i, l_i]
            if km_blk is not None:
                gm = cmask_t[s + b0 + th.tensor(gidx, device=dev)[r_i]]
            else:
                gm = cmask_t[s + b0 + r_i]
            lat_buf.append(l_i.to(th.int32).cpu().numpy())
            act_buf.append(a_v.cpu().numpy())
            cm_buf.append(gm.cpu().numpy())
            tot_nz += len(r_i)
        if (s // blk) % 20 == 0:
            log(f"  encode {e}/{N} nz={tot_nz} {time.time()-t0:.0f}s")
    return np.concatenate(lat_buf), np.concatenate(act_buf), np.concatenate(cm_buf)


def sweep_maxf1(lat, act, cm, dict_size, nc, Pc, bits=None):
    """Exact per-latent InterPLM max-F1 sweep. For each latent, sort its firing residues by
    activation descending, walk the cumulative TP against each concept, and take the max F1
    over all prefix thresholds. Byte-identical to the inline sweep in every concept_f1 script.

    lat/act/cm : the (latent, activation, concept-bitmask) triples from encode_collect_triples
    dict_size  : crosscoder dictionary size (rows of the output matrix)
    nc         : number of concepts
    Pc         : (NC,) positives per concept (denominator for recall) — computed by the caller
                 over whatever residue population defines this evaluation.
    Returns (maxf1 (dict_size, NC) float32, active latent ids)."""
    if bits is None:
        bits = concept_bits(nc)
    order = np.argsort(lat, kind="stable")
    lat_s = lat[order]
    act_s = act[order]
    cm_s = cm[order]
    la_u, la_first, la_cnt = np.unique(lat_s, return_index=True, return_counts=True)
    maxf1 = np.zeros((dict_size, nc), dtype=np.float32)
    t0 = time.time()
    for k, (L, f0, c) in enumerate(zip(la_u, la_first, la_cnt)):
        a = act_s[f0 : f0 + c]
        m = cm_s[f0 : f0 + c]
        o = np.argsort(-a, kind="stable")
        mo = m[o]
        Ymem = (mo[:, None] & bits[None, :]) > 0
        tp = np.cumsum(Ymem, axis=0).astype(np.float64)
        pp = np.arange(1, c + 1, dtype=np.float64)[:, None]
        prec = tp / pp
        rec = tp / np.maximum(Pc[None, :], 1)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
        maxf1[L] = f1.max(0)
        if k % 2000 == 0:
            log(f"  sweep {k}/{len(la_u)} {time.time()-t0:.0f}s")
    active = np.unique(lat)
    return maxf1, active


def haque_floor(prevalence):
    """The Haque detector criterion floor: max-F1 must exceed 2*pi/(1+pi) + 0.2."""
    return 2 * prevalence / (1 + prevalence) + 0.2
