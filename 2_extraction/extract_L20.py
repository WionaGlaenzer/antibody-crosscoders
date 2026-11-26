#!/usr/bin/env python
"""Layer-20-only extraction, both regimes selectable via --regime.

Thin wrapper over the consolidated extract_activations.py, fixed to LAYERS=[20]
(the original extract_L20.py restricted the sweep to L20 to avoid writing the full
4-layer set). --regime {single,paired,both} is forwarded as-is. Equivalent to:

    python extract_activations.py --layers 20 [--regime ...] <args...>
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_activations

if __name__ == "__main__":
    if not any(a == "--layers" for a in sys.argv):
        sys.argv += ["--layers", "20"]
    extract_activations.main()
