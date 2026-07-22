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
    """Trust the OS certificate store (idempotent).

    Required behind corporate TLS-intercepting proxies for HuggingFace Hub /
    PyPI downloads. Must be called before any HTTP client is created.
    """
    import truststore

    truststore.inject_into_ssl()


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

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["cuda_available"] = False
    return info
