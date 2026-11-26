#!/usr/bin/env python
"""Single (unpaired) regime, layer 20 only — the 96k single-chain store.

Thin wrapper over the consolidated extract_activations.py — identical extraction
core, fixed to LAYERS=[20] and regime=single. Kept as a named entry point so the
provenance of the 96k single store is explicit; equivalent to:

    python extract_activations.py --layers 20 --regime single <args...>

All CLI args are forwarded unchanged.
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_activations

if __name__ == "__main__":
    if not any(a == "--layers" for a in sys.argv):
        sys.argv += ["--layers", "20"]
    if not any(a == "--regime" for a in sys.argv):
        sys.argv += ["--regime", "single"]
    extract_activations.main()
