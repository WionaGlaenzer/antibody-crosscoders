"""
Verifies that activations were extracted successfully and written to the HDF5 file.
"""

import h5py, numpy as np, os

f = h5py.File(os.environ["D"] + "/activations/scaleup_paired_L20.h5")
sp = f["split"][:]
print(
    "rows",
    f["acts"].shape[0],
    "train",
    int((sp == 0).sum()),
    "val",
    int((sp == 1).sum()),
    "test",
    int((sp == 2).sum()),
)
print("model_order", f.attrs.get("model_order"), "| train_mean", "train_mean" in f)
