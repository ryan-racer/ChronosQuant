"""ChronosQuant: quantization and evaluation of the Chronos-2 time series foundation model."""

import os
from pathlib import Path

__version__ = "0.1.0"

# HF `datasets` reads these environment variables at import time, so they must be set
# before any submodule imports `fev` (which imports `datasets`). Package __init__ runs
# first on any `chronosquant.*` import, making this the one place that guarantees the
# ordering. `ensure_truststore()` sets the same values as a belt-and-braces fallback.
#
# - HF_DATASETS_CACHE: project-local prepared-datasets cache. The cache format is not
#   compatible across `datasets` major versions (4.x writes feature types 3.x cannot
#   read), so sharing the user-global cache with other tools silently poisons runs.
# - HF_DATASETS_TRUST_REMOTE_CODE: `autogluon/chronos_datasets_extra` (ETTh/ETTm)
#   ships a loading script; `datasets` 3.x wants this opt-in (4.x removed support
#   entirely — hence the `datasets<4` pin).
_REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_DATASETS_CACHE", str(_REPO_ROOT / "data" / "hf_datasets_cache"))
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
