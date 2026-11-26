#!/usr/bin/env python
"""Scale-up variant: paired regime, layer 20 only.

Thin wrapper over the consolidated extract_activations.py — identical extraction
core, fixed to LAYERS=[20] and regime=paired (what the scale-up paired store was
built with). Kept as a named entry point so the provenance of
scaleup_paired_L20.h5 is explicit; equivalent to:

    python extract_activations.py --layers 20 --regime paired <args...>

All CLI args are forwarded unchanged.
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_activations

if __name__ == "__main__":
    # inject the fixed layer/regime unless the caller overrode them
    if not any(a == "--layers" for a in sys.argv):
        sys.argv += ["--layers", "20"]
    if not any(a == "--regime" for a in sys.argv):
        sys.argv += ["--regime", "paired"]
    extract_activations.main()
