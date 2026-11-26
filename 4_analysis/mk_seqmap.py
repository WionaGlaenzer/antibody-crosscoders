"""Build the seq-id -> (antibody / general-protein) index map from feature_interp.json, resolving
the +100000 general-protein namespacing back to pool rows. Feeds downstream example-selection
and label-join steps.
"""

import json, pyarrow.parquet as pq, numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import BASE

DD = BASE
fi = json.load(open(DD + "/results/feature_interp.json"))
ab_ids = set()
gen_ids = set()
for f, rec in fi["records"].items():
    for t in rec["top"]:
        if t["kind"] == 0:
            ab_ids.add(t["seq_id"])
        else:
            gen_ids.add(t["seq_id"] - 100000)
ab_ids = sorted(ab_ids)
gen_ids = sorted(gen_ids)
oas = pq.read_table(
    DD + "/data/oas_full_annotated.parquet",
    columns=[
        "id",
        "heavy_aa",
        "light_aa",
        "region_labels_heavy",
        "region_labels_light",
        "v_call_heavy",
        "j_call_heavy",
        "v_call_light",
        "j_call_light",
    ],
).to_pydict()
ab_map = {}
for sid in ab_ids:
    ab_map[str(sid)] = {
        "id": oas["id"][sid],
        "heavy_aa": oas["heavy_aa"][sid],
        "light_aa": oas["light_aa"][sid],
        "reg_h": oas["region_labels_heavy"][sid],
        "reg_l": oas["region_labels_light"][sid],
        "v_h": oas["v_call_heavy"][sid],
        "j_h": oas["j_call_heavy"][sid],
        "v_l": oas["v_call_light"][sid],
        "j_l": oas["j_call_light"][sid],
    }
gp = pq.read_table(
    DD + "/data/general_protein_sample.parquet", columns=["id", "seq", "source"]
).to_pydict()
gen_map = {}
for sid in gen_ids:
    gen_map[str(sid)] = {"id": gp["id"][sid], "seq": gp["seq"][sid], "source": gp["source"][sid]}
json.dump({"ab": ab_map, "gen": gen_map}, open(DD + "/results/interp_seqmap.json", "w"))
print("ab seqs", len(ab_map), "gen seqs", len(gen_map))
