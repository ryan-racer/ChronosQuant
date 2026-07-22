"""Shared utilities: TLS handling and run provenance."""

import functools
import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


@functools.cache
def ensure_truststore() -> None:
    """Configure network/HF access (idempotent). Called at every entry point.

    - Trusts the OS certificate store: required behind corporate TLS-intercepting
      proxies for HuggingFace Hub / PyPI downloads. Must run before any HTTP client
      is created.
    - Allows script-based HF datasets: `autogluon/chronos_datasets_extra` (ETTh/ETTm,
      2 of the 27 Benchmark II tasks) ships a loading script, which `datasets` 3.x
      only runs with trust_remote_code enabled (and 4.x not at all — hence the
      `datasets<4` pin in pyproject.toml).
    """
    import os

    import truststore

    truststore.inject_into_ssl()
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    # Project-local prepared-datasets cache: the cache format is not compatible across
    # `datasets` major versions (4.x writes feature types 3.x cannot read), so sharing
    # the user-global cache with other tools silently poisons runs. Must be set before
    # `datasets` is imported.
    os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / "data" / "hf_datasets_cache"))


def get_git_sha() -> str | None:
    """Return the current git commit SHA (with ``+dirty`` suffix if unclean), or None."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}+dirty" if status else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def environment_info() -> dict[str, Any]:
    """Snapshot of the software/hardware environment for run provenance.

    Every evaluation run persists this alongside its results so that any number
    in the paper can be traced back to an exact code + environment state.
    """
    info: dict[str, Any] = {
        "git_sha": get_git_sha(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    for pkg in ["chronosquant", "chronos-forecasting", "fev", "torch", "datasets", "numpy"]:
        try:
            info[f"version_{pkg.replace('-', '_')}"] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            info[f"version_{pkg.replace('-', '_')}"] = None

    info["cpu"] = platform.processor()
    try:
        import psutil

        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
    except ImportError:
        pass

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
            info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    except ImportError:
        info["cuda_available"] = False

    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        info["nvidia_driver_version"] = driver.splitlines()[0] if driver else None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        info["nvidia_driver_version"] = None
    return info


def file_sha256(path) -> str:
    """SHA-256 of a file (provenance for benchmark definitions)."""
    import hashlib
    from pathlib import Path

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
