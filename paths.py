"""Project paths used by all pipeline scripts.

Values are read from ``config.yaml`` in the repository root. If that file does not exist,
``config.example.yaml`` is used, which holds the paths of the paper's setup on ETH Euler.
Another config file can be selected with the environment variable ``CROSSCODER_CONFIG``,
and the project root and scratch directory can be overridden with ``CROSSCODER_ROOT`` and
``CROSSCODER_SCRATCH``.
"""

import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent

if os.environ.get("CROSSCODER_CONFIG"):
    CONFIG_FILE = Path(os.environ["CROSSCODER_CONFIG"])
elif (REPO / "config.yaml").exists():
    CONFIG_FILE = REPO / "config.yaml"
else:
    CONFIG_FILE = REPO / "config.example.yaml"

_cfg = yaml.safe_load(CONFIG_FILE.read_text())


def _path(key, env=None):
    value = os.environ.get(env) if env else None
    return str(value or _cfg[key]).rstrip("/")


BASE = _path("root", "CROSSCODER_ROOT")  # data/, activations/, checkpoints/, results/
SCRATCH = _path("scratch", "CROSSCODER_SCRATCH")  # large temporary files
HF_HOME = _path("hf_home")  # Hugging Face model cache
OAS_RAW = _path("oas_raw")  # raw OAS paired download (1_data_prep only)
