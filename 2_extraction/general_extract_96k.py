"""Same as general_extract.py but for the full 96k declustered general-protein pool.
Only the input parquet and output h5 path differ; the extraction is identical.
"""

import os, sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import general_extract


if __name__ == "__main__":
    general_extract.main(
        gp=f"{BASE}/data/general_protein_96k.parquet",
        out=f"{BASE}/activations/general_96k_single_L20.h5",
    )
