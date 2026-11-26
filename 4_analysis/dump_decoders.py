#!/usr/bin/env python3
"""Dump per-model decoder weight matrices from the antibody paired and mixed crosscoders,
for InterPLM-style UMAP of latent decoder vectors. Writes decoder_paired.npz, decoder_mixed.npz."""

import numpy as np, torch as th, json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

CKPT = f"{BASE}/checkpoints"
from dictionary_learning.dictionary import BatchTopKCrossCoder


def dump(tag, ckpt):
    ae = BatchTopKCrossCoder.from_pretrained(f"{CKPT}/{ckpt}", device="cpu", dtype=th.float32)
    ae.eval()
    # decoder: shape (num_layers, dict_size, activation_dim) typically
    W = ae.decoder.weight.detach().cpu().numpy() if hasattr(ae.decoder, "weight") else None
    if W is None:
        # try common attribute names
        for attr in ["decoder", "W_dec", "decoders"]:
            obj = getattr(ae, attr, None)
            print(tag, "attr", attr, type(obj))
    print(tag, "decoder.weight shape:", W.shape if W is not None else None)
    np.savez_compressed(f"{BASE}/results/decoder_{tag}.npz", W=W)
    return W.shape


print("paired:", dump("paired", "cc_paired_L20.pt"))
print("mixed:", dump("mixed", "cc_mixed_L20.pt"))
