#!/usr/bin/env python
"""Shared primitives for the 2_extraction activation extractors.

Every extractor in this stage runs the same three frozen 30-layer / 1024-hidden
BERT encoders (ProtBERT, IgBERT-unpaired, IgBERT-paired) over spaced amino-acid
strings, slices per-residue hidden states, stacks the three models on a middle
axis, and accumulates a per-model train-split mean/std. Those primitives lived
copy-pasted in each script; they are factored here VERBATIM (same formulas,
same fp16/fp64 casts) so the extractors are thin wrappers over one core.

Nothing here changes behaviour relative to the previous inline versions — the
per-script differences that DO matter (which layers, which regime, the data
schema, the variance floor) stay in the individual scripts and are passed in as
arguments.
"""

import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Canonical model set + column order used by every extractor. The crosscoder's
# model axis is [prot_bert, igbert_unpaired, igbert_paired]; do not reorder.
MODELS = ["Rostlab/prot_bert", "Exscientia/IgBert_unpaired", "Exscientia/IgBert"]
MODEL_KEYS = ["prot_bert", "igbert_unpaired", "igbert_paired"]
HID = 1024
NM = len(MODELS)


def log(msg, logf=None):
    """Timestamped stdout line, optionally mirrored to an open log file handle."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logf is not None:
        logf.write(line + "\n")
        logf.flush()


def space_tok(seq):
    """ProtBERT/IgBERT expect residues space-separated: 'EVQL' -> 'E V Q L'."""
    return " ".join(list(seq))


def load_tokenizers(models=MODELS, keys=MODEL_KEYS):
    """Load the tokenizers and ASSERT all three are the identical BertTokenizerFast
    (same vocab, same CLS/SEP ids). Identical tokenization is what guarantees that
    residue r maps to the same token row across all three models — the invariant
    the whole crosscoder alignment depends on. Returns (tokenizers, shared_tok)."""
    toks = [AutoTokenizer.from_pretrained(m, do_lower_case=False) for m in models]
    v0 = toks[0].get_vocab()
    for k, t in zip(keys[1:], toks[1:]):
        assert t.get_vocab() == v0, f"tokenizer vocab mismatch for {k}"
        assert t.cls_token_id == toks[0].cls_token_id and t.sep_token_id == toks[0].sep_token_id
    return toks, toks[0]


def load_models(device, models=MODELS):
    """Load the three encoders in fp16, eval mode, on `device`, in canonical order."""
    return [
        AutoModel.from_pretrained(m, torch_dtype=torch.float16).eval().to(device) for m in models
    ]


@torch.no_grad()
def fwd_layers(model, enc, device, layers):
    """Run one model, return {layer: (B, T, HID) fp16 numpy} for the requested layers.
    hidden_states is length 31 (index 0 = embeddings, 1..30 = transformer layers)."""
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states
    return {L: hs[L].to(torch.float16).cpu().numpy() for L in layers}


def stack_residues(hs_per_model, layer, i, sl):
    """Stack the three models' hidden states for sequence `i`, residue slice `sl`,
    at `layer`, into a (n_res, NM, HID) fp16 array — the per-residue crosscoder input.
    hs_per_model is the list (over models) of {layer: (B,T,HID)} dicts from fwd_layers."""
    return np.stack([hs_per_model[m][layer][i, sl] for m in range(NM)], axis=1)


class RunningStats:
    """Streaming per-model mean/std accumulator over the TRAIN split, in float64
    (sum, sum-of-squares, count) exactly as each extractor did inline.

    `finalize(var_floor=...)` reproduces each script's own variance floor:
      - core antibody / mixed extractors used np.maximum(var, 0.0)  -> var_floor=0.0
      - general_extract used np.clip(var, 1e-8, None)               -> var_floor=1e-8
      - all_layers_paired used np.clip(var, 1e-12, None)            -> var_floor=1e-12
    Pass the value matching the script being refactored; the default (0.0) matches
    the antibody core family."""

    def __init__(self, nm=NM, hid=HID):
        self.sum = np.zeros((nm, hid), np.float64)
        self.sq = np.zeros((nm, hid), np.float64)
        self.cnt = 0

    def update(self, acts_train):
        """acts_train: (k, NM, HID) rows from the train split (split==0)."""
        xf = acts_train.astype(np.float64)
        self.sum += xf.sum(0)
        self.sq += (xf * xf).sum(0)
        self.cnt += int(acts_train.shape[0])

    def finalize(self, var_floor=0.0):
        """Return (mean, std) as float32. `var_floor` is the lower clip on variance
        (0.0 -> np.maximum semantics; >0 -> np.clip lower bound)."""
        c = max(self.cnt, 1)
        mean = (self.sum / c).astype(np.float32)
        var = self.sq / c - (self.sum / c) ** 2
        if var_floor == 0.0:
            var = np.maximum(var, 0.0)
        else:
            var = np.clip(var, var_floor, None)
        std = np.sqrt(var).astype(np.float32)
        return mean, std


def sanity_check(h5file, nm=NM, n_sample=8192):
    """Assert no NaN/Inf in a sample of `acts` and (if present) train_mean/train_std.
    Returns a dict of the observed values for logging."""
    acts = h5file["acts"]
    n = acts.shape[0]
    samp = acts[: min(n_sample, n)].astype(np.float32)
    nan = int(np.isnan(samp).sum())
    inf = int(np.isinf(samp).sum())
    mnan = 0
    mean_norms = None
    if "train_mean" in h5file and "train_std" in h5file:
        tm = h5file["train_mean"][:]
        ts = h5file["train_std"][:]
        mnan = int(np.isnan(tm).sum()) + int(np.isnan(ts).sum())
        mean_norms = [round(float(np.linalg.norm(tm[m])), 3) for m in range(nm)]
    assert (
        nan == 0 and inf == 0 and mnan == 0
    ), f"NaN/Inf in store (acts nan={nan} inf={inf} norm_nan={mnan})"
    return dict(shape=tuple(acts.shape), nan=nan, inf=inf, norm_nan=mnan, mean_norms=mean_norms)
