"""
Verifies that there is no cluster overlap between the splits of the sequence data and performs other sanity checks.
"""

import pandas as pd, json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

d = pd.read_parquet(
    f"{BASE}/data/oas_scaleup_annotated.parquet",
    columns=[
        "cluster_id",
        "split",
        "heavy_aa",
        "v_family_heavy",
        "Isotype",
        "BType",
        "Disease",
        "region_labels_heavy",
    ],
)
g = d.groupby("cluster_id")["split"].nunique()
out = dict(
    rows=len(d),
    n_clusters=int(len(g)),
    clusters_spanning=int((g > 1).sum()),
    split=d["split"].value_counts().to_dict(),
    vfam=int(d["v_family_heavy"].nunique()),
    iso=d["Isotype"].value_counts().head(6).to_dict(),
    btype=int(d["BType"].nunique()),
    disease=int(d["Disease"].nunique()),
    rl_match=bool(len(d.iloc[0]["region_labels_heavy"]) == len(d.iloc[0]["heavy_aa"])),
)
json.dump(
    out,
    open(f"{BASE}/results/scaleup_verify.json", "w"),
    indent=2,
    default=str,
)
print("OK")
